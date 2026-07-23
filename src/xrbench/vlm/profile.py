"""Remote VLM component profiling with per-stage failure continuation."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from xrbench.config import BenchConfig
from xrbench.errors import classify_exception
from xrbench.hub.client import AIHubClient, CompileRequest
from xrbench.hub.jobs import PlannedJob
from xrbench.manifest import JobManifest
from xrbench.metrics import StageProfile
from xrbench.validation import compare_arrays
from xrbench.vlm.export import ExportedStage
from xrbench.vlm.local_reference import deterministic_inputs_for_export
from xrbench.vlm.smolvlm_adapter import SmolVLMAdapter, VisionSample

LOGGER = logging.getLogger(__name__)


def vlm_plan(
    exports: Sequence[ExportedStage] | None,
    *,
    model_id: str,
    device_name: str,
    prompt_lengths: Sequence[int],
    context_lengths: Sequence[int],
    skip_inference: bool,
    precision_modes: Sequence[str] = ("float",),
) -> list[PlannedJob]:
    stage_variants: list[tuple[str, str, str, dict[str, object]]]
    if exports is None:
        stage_variants = [
            (
                "vision_encoder",
                "processor-image",
                "pt2-or-onnx",
                {
                    "pixel_values": {
                        "shape": ["1", "3", "processor_height", "processor_width"],
                        "dtype": "float32",
                    },
                    "patch_attention_mask": {
                        "shape": ["1", "patch_rows", "patch_columns"],
                        "dtype": "int32",
                    },
                },
            ),
            (
                "vision_projector",
                "visual-features",
                "pt2-or-onnx",
                {
                    "visual_features": {
                        "shape": ["1", "vision_tokens", "vision_hidden_size"],
                        "dtype": "float32",
                    }
                },
            ),
        ]
        stage_variants.extend(
            [
                (
                    "language_prefill",
                    f"prompt-{length}",
                    "pt2-or-onnx",
                    {"input_ids": [1, length], "attention_mask": [1, length]},
                )
                for length in prompt_lengths
            ]
        )
        stage_variants.extend(
            [
                (
                    "language_decode",
                    f"context-{length}",
                    "pt2-or-onnx",
                    {"input_ids": [1, 1], "context_length": length},
                )
                for length in context_lengths
            ]
        )
    else:
        stage_variants = [
            (
                item.stage,
                item.variant,
                item.source_format or "export_failed",
                {
                    key: {"shape": value[0], "dtype": value[1]}
                    for key, value in item.input_specs.items()
                },
            )
            for item in exports
        ]
    return [
        PlannedJob(
            model=model_id,
            stage=stage,
            variant=variant,
            target_device=device_name,
            source_format=source_format,
            target_runtime="qnn_dlc",
            precision=precision,
            input_shapes=input_specs,
            compile_jobs=0 if source_format == "export_failed" else 1,
            profile_jobs=0 if source_format == "export_failed" else 1,
            inference_jobs=0 if skip_inference or source_format == "export_failed" else 1,
        )
        for stage, variant, source_format, input_specs in stage_variants
        for precision in precision_modes
    ]


def profile_vlm_exports(
    config: BenchConfig,
    run_dir: Path,
    device_name: str,
    hub_client: AIHubClient,
    exports: Sequence[ExportedStage],
    *,
    adapter: SmolVLMAdapter,
    sample: VisionSample,
    device_os: str | None = None,
    hub_device: Any | None = None,
    resume: bool,
    force_resubmit: bool,
    skip_inference: bool,
    skip_download: bool,
) -> tuple[list[StageProfile], list[dict[str, str]]]:
    manifest = JobManifest.load(run_dir / "job_manifest.json")
    section = config.section("vlm")
    model_id = str(section["model_id"])
    prompt = str(section["prompt"])
    precisions = list(map(str, config.section("precision").get("modes", ["float"])))
    profiles: list[StageProfile] = []
    failures: list[dict[str, str]] = []
    for exported in exports:
        base_specs = _report_specs(exported)
        if exported.status != "exported" or not exported.source_path or not exported.source_format:
            for precision in precisions:
                profiles.append(
                    StageProfile(
                        benchmark="vlm",
                        model_id=model_id,
                        stage=exported.stage,
                        device_name=device_name,
                        device_os=device_os,
                        runtime="qnn_dlc",
                        precision=precision,
                        input_variant=exported.variant,
                        input_specs=base_specs,
                        status=exported.status,
                        error_summary=exported.error_summary,
                    )
                )
            failures.append(
                {
                    "stage": f"{exported.stage}/{exported.variant}",
                    "category": exported.error_category or "export_failure",
                    "status": exported.status,
                    "summary": exported.error_summary or "Export failed",
                }
            )
            continue
        args = deterministic_inputs_for_export(
            adapter, sample, exported.stage, exported.variant, prompt
        )
        inference_inputs = {
            name: [_to_numpy(value)]
            for name, value in zip(exported.input_names, args, strict=True)
        }
        local_primary = _local_primary_output(adapter, exported.stage, args)
        for precision in precisions:
            request = CompileRequest(
                stage_name=exported.stage,
                source_model_path=Path(exported.source_path),
                source_format=exported.source_format,
                target_runtime="qnn_dlc",
                precision=precision,
                input_specs=exported.input_specs,
                compile_options="",
                calibration_data=inference_inputs if precision == "int8" else None,
            )
            job_variant = (
                exported.variant if precision == "float" else f"{exported.variant}-{precision}"
            )
            try:
                execution = hub_client.execute(
                    request,
                    benchmark="vlm",
                    variant=job_variant,
                    device_name=device_name,
                    manifest=manifest,
                    run_dir=run_dir,
                    device_os=device_os,
                    device=hub_device,
                    configuration_checksum=config.checksum(),
                    inference_inputs=inference_inputs,
                    resume=resume,
                    force_resubmit=force_resubmit,
                    skip_inference=skip_inference,
                    skip_download=skip_download,
                )
                parsed = execution.parsed_profile
                distribution = parsed.inference_distribution if parsed else None
                validation = _validate_primary_output(
                    exported,
                    local_primary,
                    execution.inference_outputs,
                    atol=float(config.section("validation").get("atol", 1e-3))
                    * (10 if precision == "int8" else 1),
                    rtol=float(config.section("validation").get("rtol", 1e-2))
                    * (10 if precision == "int8" else 1),
                )
                if validation is not None:
                    validation_path = (
                        run_dir
                        / "inference_outputs"
                        / f"{exported.stage}-{job_variant}.validation.json"
                    )
                    validation_path.write_text(
                        json.dumps(validation, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    if not bool(validation["within_tolerance"]):
                        failures.append(
                            {
                                "stage": f"{exported.stage}/{job_variant}",
                                "category": "inference_mismatch",
                                "status": "inference_mismatch",
                                "summary": "Hosted primary output exceeded configured tolerance",
                            }
                        )
                profiles.append(
                    StageProfile(
                        benchmark="vlm",
                        model_id=model_id,
                        stage=exported.stage,
                        device_name=device_name,
                        device_os=device_os,
                        runtime="qnn_dlc",
                        precision=precision,
                        input_variant=exported.variant,
                        input_specs=base_specs,
                        compile_job_id=execution.record.compile_job_id,
                        profile_job_id=execution.record.profile_job_id,
                        inference_job_id=execution.record.inference_job_id,
                        compile_job_url=execution.record.compile_job_url,
                        profile_job_url=execution.record.profile_job_url,
                        status=(
                            "inference_mismatch"
                            if validation is not None
                            and not bool(validation["within_tolerance"])
                            else execution.record.status
                        ),
                        estimated_inference_ms=parsed.estimated_inference_ms if parsed else None,
                        inference_min_ms=distribution.minimum if distribution else None,
                        inference_mean_ms=distribution.mean if distribution else None,
                        inference_median_ms=distribution.median if distribution else None,
                        inference_p90_ms=distribution.p90 if distribution else None,
                        inference_p95_ms=distribution.p95 if distribution else None,
                        inference_p99_ms=distribution.p99 if distribution else None,
                        inference_stddev_ms=distribution.stddev if distribution else None,
                        inference_samples=distribution.samples if distribution else 0,
                        first_load_ms=parsed.first_load_ms if parsed else None,
                        warm_load_ms=parsed.warm_load_ms if parsed else None,
                        compile_ms=parsed.compile_ms if parsed else None,
                        inference_peak_mib_low=parsed.inference_peak_mib[0] if parsed else None,
                        inference_peak_mib_high=parsed.inference_peak_mib[1] if parsed else None,
                        inference_increase_mib_low=(
                            parsed.inference_increase_mib[0] if parsed else None
                        ),
                        inference_increase_mib_high=(
                            parsed.inference_increase_mib[1] if parsed else None
                        ),
                        first_load_peak_mib_low=(
                            parsed.first_load_peak_mib[0] if parsed else None
                        ),
                        first_load_peak_mib_high=(
                            parsed.first_load_peak_mib[1] if parsed else None
                        ),
                        first_load_increase_mib_low=(
                            parsed.first_load_increase_mib[0] if parsed else None
                        ),
                        first_load_increase_mib_high=(
                            parsed.first_load_increase_mib[1] if parsed else None
                        ),
                        warm_load_peak_mib_low=(
                            parsed.warm_load_peak_mib[0] if parsed else None
                        ),
                        warm_load_peak_mib_high=(
                            parsed.warm_load_peak_mib[1] if parsed else None
                        ),
                        warm_load_increase_mib_low=(
                            parsed.warm_load_increase_mib[0] if parsed else None
                        ),
                        warm_load_increase_mib_high=(
                            parsed.warm_load_increase_mib[1] if parsed else None
                        ),
                        compile_peak_mib_low=(
                            parsed.compile_peak_mib[0] if parsed else None
                        ),
                        compile_peak_mib_high=(
                            parsed.compile_peak_mib[1] if parsed else None
                        ),
                        compile_increase_mib_low=(
                            parsed.compile_increase_mib[0] if parsed else None
                        ),
                        compile_increase_mib_high=(
                            parsed.compile_increase_mib[1] if parsed else None
                        ),
                        placement=parsed.placement if parsed else None,
                        cpu_fallback=parsed.cpu_fallback if parsed else None,
                        runtime_layers=list(parsed.runtime_layers) if parsed else None,
                        validation=validation,
                    )
                )
            except Exception as error:
                category = classify_exception(error)
                summary = str(error)[:2000]
                matching = next(
                    (
                        record
                        for record in reversed(manifest.records)
                        if record.stage == exported.stage
                        and record.variant == job_variant
                        and record.precision == precision
                    ),
                    None,
                )
                status = matching.status if matching else "compile_failed"
                failures.append(
                    {
                        "stage": f"{exported.stage}/{job_variant}",
                        "category": category,
                        "status": status,
                        "summary": summary,
                    }
                )
                profiles.append(
                    StageProfile(
                        benchmark="vlm",
                        model_id=model_id,
                        stage=exported.stage,
                        device_name=device_name,
                        device_os=device_os,
                        runtime="qnn_dlc",
                        precision=precision,
                        input_variant=exported.variant,
                        input_specs=base_specs,
                        compile_job_id=matching.compile_job_id if matching else None,
                        profile_job_id=matching.profile_job_id if matching else None,
                        inference_job_id=matching.inference_job_id if matching else None,
                        compile_job_url=matching.compile_job_url if matching else None,
                        profile_job_url=matching.profile_job_url if matching else None,
                        status=status,
                        error_summary=summary,
                    )
                )
                LOGGER.error(
                    "VLM stage %s/%s/%s failed: %s",
                    exported.stage,
                    exported.variant,
                    precision,
                    error,
                )
    write_unsupported_report(run_dir, failures)
    return profiles, failures


def _report_specs(exported: ExportedStage) -> dict[str, object]:
    result: dict[str, object] = {
        key: {"shape": list(value[0]), "dtype": value[1]}
        for key, value in exported.input_specs.items()
    }
    if exported.stage == "language_prefill":
        result["prompt_length"] = int(exported.variant.rsplit("-", 1)[1])
    if exported.stage == "language_decode":
        result["context_length"] = int(exported.variant.rsplit("-", 1)[1])
        result["decode_query_length"] = 1
    return result


def _to_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    # AI Hub does not accept bool tensor IO for generic compile inputs.
    if str(getattr(value, "dtype", "")) == "bool":
        value = value.astype("int32")
    return value


def _local_primary_output(
    adapter: SmolVLMAdapter, stage: str, args: tuple[Any, ...]
) -> Any:
    import torch

    with torch.inference_mode():
        output = adapter.wrappers()[stage](*args)
    primary = output[0] if isinstance(output, tuple | list) else output
    return _to_numpy(primary)


def _validate_primary_output(
    exported: ExportedStage,
    local_primary: Any,
    hosted_outputs: Any,
    *,
    atol: float,
    rtol: float,
) -> dict[str, object] | None:
    if not isinstance(hosted_outputs, Mapping) or not hosted_outputs:
        return None
    primary_name = exported.output_names[0]
    values = hosted_outputs.get(primary_name)
    if values is None:
        values = next(iter(hosted_outputs.values()))
    hosted = values[0] if isinstance(values, list | tuple) else values
    is_logits = "logits" in primary_name
    metrics = compare_arrays(local_primary, hosted, logits=is_logits)
    result: dict[str, object] = asdict(metrics)
    result.update(
        {
            "output": primary_name,
            "absolute_tolerance": atol,
            "relative_tolerance": rtol,
            "within_tolerance": bool(
                metrics.max_absolute_error <= atol
                or metrics.max_relative_error <= rtol
            ),
        }
    )
    return result


def write_unsupported_report(run_dir: Path, failures: Sequence[dict[str, str]]) -> None:
    relevant = [
        failure
        for failure in failures
        if failure["status"] in {"export_failed", "compile_failed", "exported_but_not_compiled"}
    ]
    if not relevant:
        return
    lines = [
        "# VLM unsupported and failed stages",
        "",
        "No latency is reported for these stages. Exported source graphs and available "
        "Workbench logs remain in the run directory.",
        "",
    ]
    for failure in relevant:
        lines.extend(
            [
                f"## {failure['stage']}",
                "",
                f"- Status: `{failure['status']}`",
                f"- Category: `{failure['category']}`",
                "",
                "```text",
                failure["summary"],
                "```",
                "",
            ]
        )
    (run_dir / "failure_reports" / "unsupported_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
