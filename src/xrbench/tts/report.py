"""Derived TTS metrics from measured stage profiles and observed local call counts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, replace
from typing import Any

from xrbench.metrics import StageProfile, estimated_neural_rtf, estimated_tts_neural_ms
from xrbench.tts.local_reference import LocalTTSResult


def derive_tts_metrics(
    profiles: Sequence[StageProfile], local_results: Sequence[LocalTTSResult]
) -> dict[str, Any]:
    latencies = {profile.stage: profile.estimated_inference_ms for profile in profiles}
    by_text: dict[str, Any] = {}
    for local in local_results:
        neural_ms = estimated_tts_neural_ms(latencies, local.invocation_counts)
        by_text[local.variant] = {
            "measurement_domains": {
                "local_cpu": "Local floating-point Python/PyTorch pipeline",
                "hosted_component": "Qualcomm AI Hub separately profiled neural stages",
            },
            "invocation_note": (
                "The official local Python Piper reference uses espeak phonemization, so "
                "Charsiu components are independently profiled but have zero observed calls "
                "in this local synthesis estimate."
            ),
            "text": {
                "raw_character_count": local.raw_character_count,
                "normalized_text_length": local.normalized_text_length,
                "phoneme_count": local.phoneme_count,
                "actual_model_input_sequence_length": local.actual_model_input_sequence_length,
            },
            "audio": {
                "generated_sample_count": local.generated_sample_count,
                "duration_seconds": local.generated_audio_duration_seconds,
            },
            "invocation_counts": dict(local.invocation_counts),
            "local_cpu_timing_ms": dict(local.timing_ms),
            "component_sum_estimated_neural_synthesis_ms": neural_ms,
            "estimated_neural_rtf": estimated_neural_rtf(
                neural_ms, local.generated_audio_duration_seconds
            ),
        }
    return {"texts": by_text, "local_reference": [asdict(item) for item in local_results]}


def expand_profiles_by_text(
    profiles: Sequence[StageProfile], local_results: Sequence[LocalTTSResult]
) -> list[StageProfile]:
    """Associate one fixed-shape measured profile with each observed text variant.

    The duplicate rows intentionally retain the same job IDs, making it clear
    that latency was measured once per fixed component graph while invocation
    counts and sequence/audio metadata vary by text.
    """

    if not local_results:
        return list(profiles)
    expanded: list[StageProfile] = []
    for local in local_results:
        for profile in profiles:
            specs = dict(profile.input_specs)
            specs.update(
                {
                    "actual_sequence_length": local.actual_model_input_sequence_length,
                    "phoneme_count": local.phoneme_count,
                    "generated_sample_count": local.generated_sample_count,
                    "generated_audio_duration_seconds": local.generated_audio_duration_seconds,
                    "profile_reuse_note": (
                        "same fixed-shape component profile job reused across text variants"
                    ),
                }
            )
            expanded.append(
                replace(profile, input_variant=local.variant, input_specs=specs)
            )
    return expanded
