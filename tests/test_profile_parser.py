from __future__ import annotations

import pytest

from xrbench.hub.profile_parser import parse_profile


def test_parse_current_execution_summary_schema() -> None:
    profile = {
        "execution_summary": {
            "estimated_inference_time": 3652,
            "first_load_time": 1_374_293,
            "warm_load_time": 277_823,
            "compile_time": 12_000,
            "all_inference_times": [3000, 4000, 5000],
            "inference_memory_increase_range": (0, 2_658_496),
            "inference_memory_peak_range": (2_654_208, 39_406_752),
            "first_load_memory_peak_range": (98_230_272, 108_003_248),
            "warm_load_memory_peak_range": (118_784, 43_225_712),
            "compile_memory_peak_range": (0, 1_048_576),
        },
        "execution_detail": [
            {"name": "a", "compute_unit": "CPU"},
            {"name": "b", "compute_unit": "NPU"},
            {"name": "c", "compute_unit": "NPU"},
        ],
    }
    parsed = parse_profile(profile)
    assert parsed.estimated_inference_ms == pytest.approx(3.652)
    assert parsed.first_load_ms == pytest.approx(1374.293)
    assert parsed.warm_load_ms == pytest.approx(277.823)
    assert parsed.all_inference_ms == (3.0, 4.0, 5.0)
    assert parsed.inference_distribution is not None
    assert parsed.inference_distribution.mean == 4.0
    assert parsed.inference_peak_mib[1] == pytest.approx(39_406_752 / 2**20)
    assert parsed.placement == {"CPU": 1, "NPU": 2}
    assert parsed.cpu_fallback is True


def test_parse_top_level_variant() -> None:
    profile = {
        "estimated_inference_time": 10_000,
        "execution_times": [9_000, 11_000],
        "load_time": 20_000,
        "execution_memory_peak_range": {"lower": 1024, "upper": 2048},
        "has_cpu_fallback": "false",
    }
    parsed = parse_profile(profile)
    assert parsed.estimated_inference_ms == 10.0
    assert parsed.first_load_ms == 20.0
    assert parsed.inference_peak_mib == pytest.approx((1024 / 2**20, 2048 / 2**20))
    assert parsed.cpu_fallback is False
