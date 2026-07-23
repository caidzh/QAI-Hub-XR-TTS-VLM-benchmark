"""Independent static-shape PT2/ONNX export with fail-soft stage isolation."""

from __future__ import annotations

import json
import traceback
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from xrbench.errors import classify_exception
from xrbench.vlm.cache_adapter import cache_tensor_names
from xrbench.vlm.smolvlm_adapter import SmolVLMAdapter, VisionSample


@dataclass(frozen=True)
class ExportedStage:
    stage: str
    variant: str
    status: str
    source_path: str | None
    source_format: str | None
    input_specs: dict[str, tuple[tuple[int, ...], str]]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    error_category: str | None = None
    error_summary: str | None = None


def _dtype_name(tensor: Any) -> str:
    text = str(tensor.dtype).replace("torch.", "")
    return "int32" if text == "bool" else text


def _specs(names: Sequence[str], args: Sequence[Any]) -> dict[str, tuple[tuple[int, ...], str]]:
    return {
        name: (tuple(map(int, tensor.shape)), _dtype_name(tensor))
        for name, tensor in zip(names, args, strict=True)
    }


def build_export_cases(
    adapter: SmolVLMAdapter,
    sample: VisionSample,
    *,
    prompt: str,
    prompt_lengths: Sequence[int],
    context_lengths: Sequence[int],
) -> list[tuple[str, str, Any, tuple[Any, ...], list[str], list[str]]]:
    import torch

    wrappers = adapter.wrappers()
    report = adapter.architecture_report(sample)
    cases: list[tuple[str, str, Any, tuple[Any, ...], list[str], list[str]]] = [
        (
            "vision_encoder",
            f"image-{sample.pixel_values.shape[-2]}x{sample.pixel_values.shape[-1]}",
            wrappers["vision_encoder"],
            (sample.pixel_values, sample.patch_attention_mask),
            ["pixel_values", "patch_attention_mask"],
            ["visual_features"],
        ),
        (
            "vision_projector",
            f"visual-tokens-{sample.visual_features.shape[1]}",
            wrappers["vision_projector"],
            (sample.visual_features,),
            ["visual_features"],
            ["projected_visual_tokens"],
        ),
    ]
    layers = int(report.number_of_layers or 0)
    present_names = cache_tensor_names(layers, "present")
    past_names = cache_tensor_names(layers, "past")
    for length in prompt_lengths:
        ids, mask = adapter.token_inputs(prompt, int(length))
        cases.append(
            (
                "language_prefill",
                f"prompt-{length}",
                wrappers["language_prefill"],
                (ids, mask),
                ["input_ids", "attention_mask"],
                ["final_logits", *present_names],
            )
        )
    for context in context_lengths:
        ids = torch.zeros((1, 1), dtype=torch.int64)
        attention = torch.ones((1, int(context) + 1), dtype=torch.int64)
        cache_position = torch.tensor([int(context)], dtype=torch.int64)
        cache = adapter.empty_cache(int(context))
        cases.append(
            (
                "language_decode",
                f"context-{context}",
                wrappers["language_decode"],
                (ids, attention, cache_position, *cache),
                ["input_ids", "attention_mask", "cache_position", *past_names],
                ["next_token_logits", *present_names],
            )
        )
    return cases


def export_all(
    adapter: SmolVLMAdapter,
    sample: VisionSample,
    output_dir: Path,
    *,
    prompt: str,
    prompt_lengths: Sequence[int],
    context_lengths: Sequence[int],
) -> list[ExportedStage]:
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_dir = output_dir.parents[1] / "failure_reports"
    failures_dir.mkdir(parents=True, exist_ok=True)
    cases = build_export_cases(
        adapter,
        sample,
        prompt=prompt,
        prompt_lengths=prompt_lengths,
        context_lengths=context_lengths,
    )
    results: list[ExportedStage] = []
    for stage, variant, wrapper, args, input_names, output_names in cases:
        base = output_dir / f"{stage}-{variant}"
        try:
            path, source_format = _export_one(
                wrapper, args, base, input_names=input_names, output_names=output_names
            )
            results.append(
                ExportedStage(
                    stage=stage,
                    variant=variant,
                    status="exported",
                    source_path=str(path.resolve()),
                    source_format=source_format,
                    input_specs=_specs(input_names, args),
                    input_names=tuple(input_names),
                    output_names=tuple(output_names),
                )
            )
        except Exception as error:
            trace = traceback.format_exc()
            category = classify_exception(error)
            report_path = failures_dir / f"{stage}-{variant}-export.md"
            report_path.write_text(
                "\n".join(
                    [
                        f"# Export failure: {stage} / {variant}",
                        "",
                        f"- Category: `{category}`",
                        "- Status: `export_failed`",
                        "- PT2 was attempted first and ONNX was attempted as fallback.",
                        "",
                        "```text",
                        trace[-12000:],
                        "```",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            results.append(
                ExportedStage(
                    stage=stage,
                    variant=variant,
                    status="export_failed",
                    source_path=None,
                    source_format=None,
                    input_specs=_specs(input_names, args),
                    input_names=tuple(input_names),
                    output_names=tuple(output_names),
                    error_category=category,
                    error_summary=str(error)[:2000],
                )
            )
    (output_dir / "exports.json").write_text(
        json.dumps([asdict(item) for item in results], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return results


def _export_one(
    wrapper: Any,
    args: tuple[Any, ...],
    base_path: Path,
    *,
    input_names: Sequence[str],
    output_names: Sequence[str],
) -> tuple[Path, str]:
    import torch

    pt2_path = base_path.with_suffix(".pt2")
    pt2_error: Exception | None = None
    try:
        with torch.inference_mode():
            exported = torch.export.export(wrapper, args, strict=False)
        torch.export.save(exported, pt2_path)
        return pt2_path, "pt2"
    except Exception as error:
        pt2_error = error

    onnx_path = base_path.with_suffix(".onnx")
    try:
        with torch.inference_mode():
            actual = wrapper(*args)
            output_count = len(actual) if isinstance(actual, tuple | list) else 1
            names = list(output_names[:output_count])
            if len(names) < output_count:
                names.extend(f"output_{index}" for index in range(len(names), output_count))
            torch.onnx.export(
                wrapper,
                args,
                onnx_path,
                input_names=list(input_names),
                output_names=names,
                opset_version=18,
                do_constant_folding=True,
                dynamo=False,
            )
        return onnx_path, "onnx"
    except Exception as onnx_error:
        raise RuntimeError(
            f"PT2 export failed: {pt2_error}; ONNX fallback failed: {onnx_error}"
        ) from onnx_error


def load_exports(path: Path) -> list[ExportedStage]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    output: list[ExportedStage] = []
    for item in raw:
        item["input_specs"] = {
            name: (tuple(value[0]), value[1]) for name, value in item["input_specs"].items()
        }
        item["input_names"] = tuple(item["input_names"])
        item["output_names"] = tuple(item["output_names"])
        output.append(ExportedStage(**item))
    return output
