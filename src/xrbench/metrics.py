"""Normalized metrics and derived benchmark calculations."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class StageProfile:
    benchmark: str
    model_id: str
    stage: str
    device_name: str
    device_os: str | None
    runtime: str
    precision: str
    input_variant: str
    input_specs: dict[str, object]
    compile_job_id: str | None = None
    profile_job_id: str | None = None
    inference_job_id: str | None = None
    compile_job_url: str | None = None
    profile_job_url: str | None = None
    status: str = "planned"
    estimated_inference_ms: float | None = None
    inference_min_ms: float | None = None
    inference_mean_ms: float | None = None
    inference_median_ms: float | None = None
    inference_p90_ms: float | None = None
    inference_p95_ms: float | None = None
    inference_p99_ms: float | None = None
    inference_stddev_ms: float | None = None
    inference_samples: int = 0
    first_load_ms: float | None = None
    warm_load_ms: float | None = None
    compile_ms: float | None = None
    inference_peak_mib_low: float | None = None
    inference_peak_mib_high: float | None = None
    inference_increase_mib_low: float | None = None
    inference_increase_mib_high: float | None = None
    first_load_peak_mib_low: float | None = None
    first_load_peak_mib_high: float | None = None
    first_load_increase_mib_low: float | None = None
    first_load_increase_mib_high: float | None = None
    warm_load_peak_mib_low: float | None = None
    warm_load_peak_mib_high: float | None = None
    warm_load_increase_mib_low: float | None = None
    warm_load_increase_mib_high: float | None = None
    compile_peak_mib_low: float | None = None
    compile_peak_mib_high: float | None = None
    compile_increase_mib_low: float | None = None
    compile_increase_mib_high: float | None = None
    placement: dict[str, int] | None = None
    cpu_fallback: bool | None = None
    runtime_layers: list[dict[str, object]] | None = None
    validation: dict[str, object] | None = None
    error_summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Distribution:
    minimum: float
    mean: float
    median: float
    p90: float
    p95: float
    p99: float
    stddev: float
    samples: int


def percentile(values: Iterable[float], quantile: float) -> float:
    """Calculate a linearly interpolated percentile for 0 <= quantile <= 100."""

    if not 0 <= quantile <= 100:
        raise ValueError("quantile must be between 0 and 100")
    ordered = sorted(float(item) for item in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * quantile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def describe(values: Iterable[float]) -> Distribution | None:
    series = [float(value) for value in values]
    if not series:
        return None
    return Distribution(
        minimum=min(series),
        mean=statistics.fmean(series),
        median=statistics.median(series),
        p90=percentile(series, 90),
        p95=percentile(series, 95),
        p99=percentile(series, 99),
        stddev=statistics.pstdev(series),
        samples=len(series),
    )


def estimated_tts_neural_ms(
    component_latency_ms: Mapping[str, float | None],
    invocation_counts: Mapping[str, int],
) -> float | None:
    """Sum component latency multiplied by observed reference invocation counts."""

    total = 0.0
    for component, count in invocation_counts.items():
        if count <= 0:
            continue
        latency = component_latency_ms.get(component)
        if latency is None:
            return None
        total += float(latency) * count
    return total


def estimated_neural_rtf(neural_ms: float | None, audio_duration_seconds: float) -> float | None:
    if neural_ms is None or audio_duration_seconds <= 0:
        return None
    return neural_ms / 1000.0 / audio_duration_seconds


def component_sum_ttft_ms(
    vision_encoder_ms: float | None,
    projector_ms: float | None,
    prefill_ms: float | None,
) -> float | None:
    values = (vision_encoder_ms, projector_ms, prefill_ms)
    return sum(float(value) for value in values if value is not None) if None not in values else None


def decode_tokens_per_second(one_token_decode_ms: float | None) -> float | None:
    if one_token_decode_ms is None or one_token_decode_ms <= 0:
        return None
    return 1000.0 / one_token_decode_ms


def estimated_generation_ms(
    ttft_ms: float | None, one_token_decode_ms: float | None, output_tokens: int
) -> float | None:
    if ttft_ms is None or one_token_decode_ms is None:
        return None
    return ttft_ms + max(output_tokens - 1, 0) * one_token_decode_ms


@dataclass(frozen=True)
class ValidationMetrics:
    max_absolute_error: float
    mean_absolute_error: float
    max_relative_error: float
    cosine_similarity: float | None
    top1_agreement: bool | None
    topk_overlap: float | None
