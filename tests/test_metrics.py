from __future__ import annotations

import pytest

from xrbench.metrics import (
    component_sum_ttft_ms,
    decode_tokens_per_second,
    describe,
    estimated_generation_ms,
    estimated_neural_rtf,
    estimated_tts_neural_ms,
    percentile,
)


def test_percentiles_and_distribution() -> None:
    values = [1, 2, 3, 4, 5]
    assert percentile(values, 90) == pytest.approx(4.6)
    summary = describe(values)
    assert summary is not None
    assert summary.minimum == 1
    assert summary.mean == 3
    assert summary.median == 3
    assert summary.samples == 5


def test_tts_derived_metrics_multiply_repeated_decoder() -> None:
    latency = {"encoder": 10.0, "sdp": 2.0, "flow": 8.0, "decoder": 4.0}
    counts = {"encoder": 1, "sdp": 1, "flow": 1, "decoder": 5}
    total = estimated_tts_neural_ms(latency, counts)
    assert total == 40.0
    assert estimated_neural_rtf(total, 2.0) == 0.02


def test_missing_tts_component_does_not_fabricate_total() -> None:
    assert estimated_tts_neural_ms({"encoder": 1.0}, {"encoder": 1, "flow": 1}) is None


def test_vlm_derived_metrics() -> None:
    ttft = component_sum_ttft_ms(5, 2, 10)
    assert ttft == 17
    assert decode_tokens_per_second(20) == 50
    assert estimated_generation_ms(ttft, 20, 8) == 157
