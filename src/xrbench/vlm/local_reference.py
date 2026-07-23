"""Local parity validation for isolated VLM wrappers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from xrbench.validation import compare_arrays
from xrbench.vlm.smolvlm_adapter import SmolVLMAdapter, VisionSample


@dataclass(frozen=True)
class LocalParity:
    stage: str
    variant: str
    passed: bool
    max_absolute_error: float
    mean_absolute_error: float
    cosine_similarity: float | None


def validate_vision_wrappers(
    adapter: SmolVLMAdapter,
    sample: VisionSample,
    output_dir: Path,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> list[LocalParity]:
    import torch

    wrappers = adapter.wrappers()
    with torch.inference_mode():
        wrapped_visual = wrappers["vision_encoder"](
            sample.pixel_values, sample.patch_attention_mask
        )
        direct_vision = adapter.vision_module(
            pixel_values=sample.pixel_values,
            patch_attention_mask=sample.patch_attention_mask.to(dtype=torch.bool),
            return_dict=True,
        ).last_hidden_state
        wrapped_projected = wrappers["vision_projector"](direct_vision)
        direct_projected = adapter.connector_module(direct_vision)

    results: list[LocalParity] = []
    for stage, wrapped, direct in (
        ("vision_encoder", wrapped_visual, direct_vision),
        ("vision_projector", wrapped_projected, direct_projected),
    ):
        validation = compare_arrays(
            direct.detach().cpu().numpy(), wrapped.detach().cpu().numpy()
        )
        passed = bool(torch.allclose(wrapped, direct, atol=atol, rtol=rtol))
        results.append(
            LocalParity(
                stage=stage,
                variant="sample-image",
                passed=passed,
                max_absolute_error=validation.max_absolute_error,
                mean_absolute_error=validation.mean_absolute_error,
                cosine_similarity=validation.cosine_similarity,
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "vlm_local_parity.json").write_text(
        json.dumps([asdict(item) for item in results], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return results


def deterministic_inputs_for_export(
    adapter: SmolVLMAdapter,
    sample: VisionSample,
    stage: str,
    variant: str,
    prompt: str,
) -> tuple[Any, ...]:
    import torch

    if stage == "vision_encoder":
        return sample.pixel_values, sample.patch_attention_mask
    if stage == "vision_projector":
        return (sample.visual_features,)
    if stage == "language_prefill":
        length = int(variant.rsplit("-", 1)[1])
        return adapter.token_inputs(prompt, length)
    if stage == "language_decode":
        context = int(variant.rsplit("-", 1)[1])
        return (
            torch.zeros((1, 1), dtype=torch.int64),
            torch.ones((1, context + 1), dtype=torch.int64),
            torch.tensor([context], dtype=torch.int64),
            *adapter.empty_cache(context),
        )
    raise KeyError(stage)
