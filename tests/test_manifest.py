from __future__ import annotations

import json
from pathlib import Path

from xrbench.manifest import JobManifest, JobRecord


def record(**changes: object) -> JobRecord:
    values = {
        "benchmark": "vlm",
        "stage": "vision_encoder",
        "variant": "image",
        "source_model_checksum": "source",
        "stage_config_checksum": "config",
        "device_name": "Samsung Galaxy S22 5G",
        "runtime": "qnn_dlc",
        "precision": "float",
        "input_specs": {"image": {"shape": [1, 3, 224, 224], "dtype": "float32"}},
    }
    values.update(changes)
    return JobRecord(**values)  # type: ignore[arg-type]


def test_manifest_round_trip_and_resume(tmp_path: Path) -> None:
    path = tmp_path / "job_manifest.json"
    manifest = JobManifest(path)
    item = record()
    manifest.upsert(item)
    manifest.update(item, compile_job_id="j123", status="success")
    loaded = JobManifest.load(path)
    found = loaded.find_compatible(record())
    assert found is not None
    assert found.compile_job_id == "j123"
    assert json.loads(path.read_text())["schema_version"] == 2


def test_resume_rejects_changed_shape_device_or_runtime(tmp_path: Path) -> None:
    manifest = JobManifest(tmp_path / "manifest.json", [record(status="success")])
    assert manifest.find_compatible(record(device_name="Other")) is None
    assert manifest.find_compatible(record(runtime="onnx")) is None
    assert (
        manifest.find_compatible(
            record(input_specs={"image": {"shape": [1, 3, 256, 256], "dtype": "float32"}})
        )
        is None
    )


def test_compatible_resubmission_attempts_are_retained(tmp_path: Path) -> None:
    manifest = JobManifest(tmp_path / "manifest.json")
    first = record(status="success")
    second = record(status="planned")
    manifest.upsert(first)
    manifest.upsert(second)
    assert len(manifest.records) == 2
    assert first.record_id != second.record_id
    assert manifest.find_compatible(record()) is first
