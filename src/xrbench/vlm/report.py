"""Derived VLM latency estimates grouped by fixed prompt/context."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from xrbench.metrics import (
    StageProfile,
    component_sum_ttft_ms,
    decode_tokens_per_second,
    estimated_generation_ms,
)


def derive_vlm_metrics(
    profiles: Sequence[StageProfile], output_lengths: Sequence[int]
) -> dict[str, Any]:
    vision = _first_latency(profiles, "vision_encoder")
    projector = _first_latency(profiles, "vision_projector")
    prefill = {
        _as_int(profile.input_specs["prompt_length"]): profile.estimated_inference_ms
        for profile in profiles
        if profile.stage == "language_prefill"
        and "prompt_length" in profile.input_specs
    }
    decode = {
        _as_int(profile.input_specs["context_length"]): profile.estimated_inference_ms
        for profile in profiles
        if profile.stage == "language_decode"
        and "context_length" in profile.input_specs
    }
    prompt_results: dict[str, Any] = {}
    for prompt_length, prefill_ms in sorted(prefill.items()):
        ttft = component_sum_ttft_ms(vision, projector, prefill_ms)
        prompt_results[str(prompt_length)] = {
            "component-sum estimated TTFT": ttft,
            "vision_encoder_ms": vision,
            "vision_projector_ms": projector,
            "language_prefill_ms": prefill_ms,
        }
    decode_results: dict[str, Any] = {}
    for context_length, decode_ms in sorted(decode.items()):
        decode_results[str(context_length)] = {
            "one_token_decode_ms": decode_ms,
            "estimated_decode_tokens_per_second": decode_tokens_per_second(decode_ms),
            "estimated_generation_ms_by_prompt": {
                str(prompt_length): {
                    str(output_length): estimated_generation_ms(
                        prompt_data["component-sum estimated TTFT"],
                        decode_ms,
                        int(output_length),
                    )
                    for output_length in output_lengths
                }
                for prompt_length, prompt_data in prompt_results.items()
            },
        }
    return {
        "prompt_lengths": prompt_results,
        "decode_context_lengths": decode_results,
        "note": "component-sum estimated TTFT is derived from separately profiled subgraphs",
    }


def _first_latency(profiles: Sequence[StageProfile], stage: str) -> float | None:
    return next(
        (
            profile.estimated_inference_ms
            for profile in profiles
            if profile.stage == stage and profile.estimated_inference_ms is not None
        ),
        None,
    )


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("Boolean is not a valid sequence length")
    if isinstance(value, int | str):
        return int(value)
    raise TypeError(f"Expected integer sequence length, got {type(value).__name__}")
