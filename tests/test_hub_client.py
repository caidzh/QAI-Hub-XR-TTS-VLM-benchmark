from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from xrbench.hub.client import AIHubClient, CompileRequest, remote_authorized
from xrbench.manifest import JobManifest


class FakeJob:
    def __init__(self, job_id: str, kind: str, success: bool = True) -> None:
        self.job_id = job_id
        self.url = f"https://example.test/jobs/{job_id}"
        self.kind = kind
        self.success = success

    def wait(self, timeout: int | None = None):
        return SimpleNamespace(success=self.success, message="failed" if not self.success else "")

    def get_target_model(self):
        return "target-model" if self.success else None

    def download_target_model(self, filename: str) -> str:
        Path(filename).write_bytes(b"model")
        return filename

    def download_profile(self):
        return {
            "execution_summary": {
                "estimated_inference_time": 2000,
                "all_inference_times": [1000, 2000, 3000],
                "inference_memory_peak_range": [0, 1_048_576],
            }
        }

    def download_output_data(self, filename: str | None = None):
        if filename is None:
            return {"output": [__import__("numpy").array([1.0], dtype="float32")]}
        Path(filename).write_bytes(b"h5")
        return filename

    def download_results(self, output_dir: str) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{self.job_id}.log").write_text("log", encoding="utf-8")

    def download_job_logs(self, output_dir: str) -> list[str]:
        return []


class FakeClient:
    def __init__(self, *, compile_success: bool = True, inference_success: bool = True) -> None:
        self.compile_success = compile_success
        self.inference_success = inference_success
        self.submissions = {"compile": 0, "link": 0, "profile": 0, "inference": 0}
        self.jobs: dict[str, FakeJob] = {}

    def submit_compile_job(self, **kwargs):
        self.submissions["compile"] += 1
        job = FakeJob("compile-1", "compile", self.compile_success)
        self.jobs[job.job_id] = job
        return job

    def submit_link_job(self, **kwargs):
        self.submissions["link"] += 1
        job = FakeJob("link-1", "link")
        self.jobs[job.job_id] = job
        return job

    def submit_profile_job(self, **kwargs):
        self.submissions["profile"] += 1
        job = FakeJob("profile-1", "profile")
        self.jobs[job.job_id] = job
        return job

    def submit_inference_job(self, **kwargs):
        self.submissions["inference"] += 1
        job = FakeJob("inference-1", "inference", self.inference_success)
        self.jobs[job.job_id] = job
        return job

    def get_job(self, job_id: str) -> FakeJob:
        return self.jobs[job_id]


def request(source: Path) -> CompileRequest:
    return CompileRequest(
        stage_name="vision_encoder",
        source_model_path=source,
        source_format="pt2",
        target_runtime="qnn_dlc",
        precision="float",
        input_specs={"image": ((1, 3, 8, 8), "float32")},
    )


def test_execute_success_downloads_and_resumes_without_duplicate_submission(
    tmp_path: Path,
) -> None:
    source = tmp_path / "model.pt2"
    source.write_bytes(b"source")
    fake = FakeClient()
    client = AIHubClient(client=fake, remote_enabled=True, retries=0)
    manifest = JobManifest(tmp_path / "job_manifest.json")
    result = client.execute(
        request(source),
        benchmark="vlm",
        variant="image",
        device_name="Samsung Galaxy S22",
        manifest=manifest,
        run_dir=tmp_path,
        inference_inputs={"image": [object()]},
    )
    assert result.record.status == "inferred"
    assert result.parsed_profile is not None
    assert result.parsed_profile.estimated_inference_ms == 2.0
    assert isinstance(result.inference_outputs, dict)
    assert "output" in result.inference_outputs
    assert fake.submissions == {"compile": 1, "link": 0, "profile": 1, "inference": 1}

    resumed = client.execute(
        request(source),
        benchmark="vlm",
        variant="image",
        device_name="Samsung Galaxy S22",
        manifest=manifest,
        run_dir=tmp_path,
        inference_inputs={"image": [object()]},
        resume=True,
    )
    assert resumed.record.compile_job_id == "compile-1"
    assert fake.submissions == {"compile": 1, "link": 0, "profile": 1, "inference": 1}


def test_voice_ai_request_links_before_profile(tmp_path: Path) -> None:
    source = tmp_path / "model.onnx"
    source.write_bytes(b"source")
    fake = FakeClient()
    client = AIHubClient(client=fake, remote_enabled=True, retries=0)
    manifest = JobManifest(tmp_path / "job_manifest.json")
    compile_request = replace(
        request(source),
        target_runtime="voice_ai",
        compile_options="--target_runtime qnn_dlc",
        link_options="--qnn_sdk_version 2.34",
        profile_options="--qnn_options context_enable_graphs=vision_encoder",
    )
    result = client.execute(
        compile_request,
        benchmark="tts",
        variant="fixed",
        device_name="Samsung Galaxy S22",
        manifest=manifest,
        run_dir=tmp_path,
        skip_inference=True,
    )
    assert result.record.link_job_id == "link-1"
    assert result.record.status == "success"
    assert fake.submissions == {"compile": 1, "link": 1, "profile": 1, "inference": 0}
    assert (tmp_path / "models" / "vision_encoder-fixed-voice-ai.bin").exists()


def test_compile_failure_persists_manifest_and_artifact(tmp_path: Path) -> None:
    source = tmp_path / "model.pt2"
    source.write_bytes(b"source")
    fake = FakeClient(compile_success=False)
    client = AIHubClient(client=fake, remote_enabled=True, retries=0)
    manifest = JobManifest(tmp_path / "job_manifest.json")
    with pytest.raises(Exception, match="compile vision_encoder failed"):
        client.execute(
            request(source),
            benchmark="vlm",
            variant="image",
            device_name="Samsung Galaxy S22",
            manifest=manifest,
            run_dir=tmp_path,
        )
    loaded = JobManifest.load(tmp_path / "job_manifest.json")
    assert loaded.records[0].status == "exported_but_not_compiled"
    assert loaded.records[0].errors
    assert (tmp_path / "logs" / "compile-1" / "compile-1.log").exists()


def test_remote_authorization_supports_either_documented_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QAIHUB_RUN_REMOTE", raising=False)
    assert remote_authorized(True)
    assert not remote_authorized(False)
    monkeypatch.setenv("QAIHUB_RUN_REMOTE", "1")
    assert remote_authorized(False)
    assert not remote_authorized(True, dry_run=True)
