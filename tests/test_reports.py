from __future__ import annotations

import csv
import json
from pathlib import Path

from xrbench.metrics import StageProfile
from xrbench.reports import generate_reports


def test_report_generation_labels_estimates_and_writes_csv(tmp_path: Path) -> None:
    profile = StageProfile(
        benchmark="vlm",
        model_id="model",
        stage="language_decode",
        device_name="Samsung Galaxy S22",
        device_os="13",
        runtime="qnn_dlc",
        precision="float",
        input_variant="context-32",
        input_specs={"context_length": 32},
        status="success",
        estimated_inference_ms=10.0,
        inference_peak_mib_low=1.0,
        inference_peak_mib_high=2.0,
        placement={"NPU": 20},
    )
    generate_reports(
        tmp_path,
        [profile],
        {"date": "today", "python_version": "3.11", "repository_commit": "abc"},
        {"name": "Samsung Galaxy S22"},
        {
            "device": {"requested_name": "Samsung Galaxy S22 5G"},
            "tts": {"model_id": "pipertts_en"},
            "vlm": {"model_id": "model"},
            "precision": {"modes": ["float"]},
        },
        {"vlm": {"component-sum estimated TTFT": 20}},
    )
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["measured_stage_profiles"][0]["estimated_inference_ms"] == 10
    markdown = (tmp_path / "report.md").read_text()
    assert "component-sum estimated TTFT" in markdown
    assert "not an end-to-end" in markdown
    assert "not a Meta Quest measurement" in markdown
    with (tmp_path / "metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["stage"] == "language_decode"
