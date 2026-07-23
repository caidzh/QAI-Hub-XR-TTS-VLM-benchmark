"""Reusable, public-API-only Qualcomm AI Hub client layer."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xrbench.config import checksum_file, checksum_mapping
from xrbench.errors import (
    AuthenticationError,
    CompileError,
    InferenceMismatchError,
    JobTimeoutError,
    ProfileError,
    XRBenchError,
    classify_exception,
)
from xrbench.hub.artifacts import (
    download_inference_output,
    download_job_artifacts,
    download_profile,
    download_target_model,
)
from xrbench.hub.profile_parser import ParsedProfile, parse_profile
from xrbench.manifest import JobManifest, JobRecord

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompileRequest:
    stage_name: str
    source_model_path: Path
    source_format: str
    target_runtime: str
    precision: str
    input_specs: dict[str, tuple[tuple[int, ...], str]]
    compile_options: str = ""
    calibration_data: Mapping[str, list[Any]] | str | None = None
    link_options: str | None = None
    profile_options: str = ""

    def options(self) -> str:
        options = self.compile_options.strip()
        runtime_flag = f"--target_runtime {self.target_runtime}"
        if "--target_runtime" not in options:
            options = f"{runtime_flag} {options}".strip()
        if self.precision == "int8" and "quantize" not in options:
            options = f"{options} --quantize_full_type int8".strip()
        return options

    def stage_checksum(self) -> str:
        return checksum_mapping(
            {
                "stage": self.stage_name,
                "source_format": self.source_format,
                "target_runtime": self.target_runtime,
                "precision": self.precision,
                "input_specs": self.input_specs,
                "compile_options": self.options(),
                "link_options": self.link_options,
                "profile_options": self.profile_options,
            }
        )


@dataclass(frozen=True)
class JobExecutionResult:
    record: JobRecord
    parsed_profile: ParsedProfile | None
    raw_profile: dict[str, Any] | None
    inference_outputs: Any | None


def remote_authorized(cli_run_remote: bool, dry_run: bool = False) -> bool:
    """Remote work is authorized by either documented opt-in; dry-run always wins."""

    env_enabled = os.environ.get("QAIHUB_RUN_REMOTE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return not dry_run and (cli_run_remote or env_enabled)


class AIHubClient:
    """Thin integration around current public ``qai_hub.Client`` methods."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        remote_enabled: bool = False,
        timeout_seconds: int = 3600,
        retries: int = 2,
        retry_backoff_seconds: float = 5.0,
    ) -> None:
        if client is None:
            try:
                import qai_hub as hub

                client = hub.Client()
            except Exception as error:
                raise AuthenticationError(
                    "Could not initialize Qualcomm AI Hub client. Run `qai-hub configure`.",
                    cause=error,
                ) from error
        self.client = client
        self.remote_enabled = remote_enabled
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.retry_backoff_seconds = retry_backoff_seconds

    def validate_authentication(self) -> list[Any]:
        try:
            return list(self.client.get_devices())
        except Exception as error:
            raise AuthenticationError(
                "Qualcomm AI Hub authentication or device access failed", cause=error
            ) from error

    def _require_remote(self) -> None:
        if not self.remote_enabled:
            raise XRBenchError(
                "Remote submission is disabled; pass --run-remote or set QAIHUB_RUN_REMOTE=1"
            )

    def _retry(self, operation: Callable[[], Any], operation_name: str) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return operation()
            except Exception as error:
                last_error = error
                if attempt >= self.retries:
                    raise
                delay = self.retry_backoff_seconds * (2**attempt)
                LOGGER.warning("%s failed (%s); retrying in %.1fs", operation_name, error, delay)
                time.sleep(delay)
        raise AssertionError(last_error)

    def _device_object(self, name: str) -> Any:
        try:
            import qai_hub as hub

            return hub.Device(name)
        except ImportError:
            # A fake client used by tests may accept the name directly.
            return name

    def _wait(self, job: Any, label: str) -> Any:
        try:
            status = job.wait(timeout=self.timeout_seconds)
        except TimeoutError as error:
            raise JobTimeoutError(f"{label} timed out after {self.timeout_seconds}s", cause=error) from error
        if not bool(getattr(status, "success", False)):
            message = str(getattr(status, "message", "") or getattr(status, "failure_reason", ""))
            raise XRBenchError(f"{label} failed: {message or status}")
        return status

    def get_job(self, job_id: str) -> Any:
        return self.client.get_job(job_id)

    def upload_model(self, model_path: Path) -> Any:
        self._require_remote()
        return self._retry(
            lambda: self.client.upload_model(model_path),
            f"model upload for {model_path.name}",
        )

    def upload_dataset(
        self, entries: Mapping[str, list[Any]], *, name: str | None = None
    ) -> Any:
        self._require_remote()
        return self._retry(
            lambda: self.client.upload_dataset(entries, name=name),
            f"dataset upload for {name or 'xrbench-inputs'}",
        )

    def wait_for_job(self, job: Any, label: str = "job") -> Any:
        return self._wait(job, label)

    def submit_compile(
        self, request: CompileRequest, device_name: str, device: Any | None = None
    ) -> Any:
        self._require_remote()
        return self._retry(
            lambda: self.client.submit_compile_job(
                model=request.source_model_path,
                device=device or self._device_object(device_name),
                name=f"xrbench-{request.stage_name}",
                input_specs=request.input_specs,
                options=request.options(),
                calibration_data=request.calibration_data,
                retry=True,
            ),
            f"compile submission for {request.stage_name}",
        )

    def submit_link(
        self,
        target_model: Any,
        device_name: str,
        stage_name: str,
        options: str,
        device: Any | None = None,
    ) -> Any:
        self._require_remote()
        return self._retry(
            lambda: self.client.submit_link_job(
                models=[target_model],
                device=device or self._device_object(device_name),
                name=f"xrbench-{stage_name}-voice-ai",
                options=options,
            ),
            f"link submission for {stage_name}",
        )

    def submit_profile(
        self,
        target_model: Any,
        device_name: str,
        stage_name: str,
        options: str = "",
        device: Any | None = None,
    ) -> Any:
        self._require_remote()
        return self._retry(
            lambda: self.client.submit_profile_job(
                model=target_model,
                device=device or self._device_object(device_name),
                name=f"xrbench-{stage_name}",
                options=options,
                retry=True,
            ),
            f"profile submission for {stage_name}",
        )

    def submit_inference(
        self,
        target_model: Any,
        device_name: str,
        stage_name: str,
        inputs: Mapping[str, list[Any]] | str,
        options: str = "",
        device: Any | None = None,
    ) -> Any:
        self._require_remote()
        return self._retry(
            lambda: self.client.submit_inference_job(
                model=target_model,
                device=device or self._device_object(device_name),
                inputs=inputs,
                name=f"xrbench-{stage_name}-validation",
                options=options,
                retry=True,
            ),
            f"inference submission for {stage_name}",
        )

    def execute(
        self,
        request: CompileRequest,
        *,
        benchmark: str,
        variant: str,
        device_name: str,
        manifest: JobManifest,
        run_dir: Path,
        device_os: str | None = None,
        device: Any | None = None,
        configuration_checksum: str | None = None,
        inference_inputs: Mapping[str, list[Any]] | str | None = None,
        resume: bool = False,
        force_resubmit: bool = False,
        skip_inference: bool = False,
        skip_download: bool = False,
    ) -> JobExecutionResult:
        """Compile, profile, optionally infer, and persist after every transition."""

        source_checksum = checksum_file(request.source_model_path)
        candidate = JobRecord(
            benchmark=benchmark,
            stage=request.stage_name,
            variant=variant,
            source_model_checksum=source_checksum,
            stage_config_checksum=request.stage_checksum(),
            device_name=device_name,
            runtime=request.target_runtime,
            precision=request.precision,
            input_specs={
                name: {"shape": list(spec[0]), "dtype": spec[1]}
                for name, spec in request.input_specs.items()
            },
            source_model_path=str(request.source_model_path),
            configuration_checksum=configuration_checksum,
            device_os=device_os,
        )
        existing = None if force_resubmit else manifest.find_compatible(candidate, successful_only=False)
        if resume and existing is not None:
            candidate = existing
        else:
            manifest.upsert(candidate)

        raw_profile: dict[str, Any] | None = None
        parsed: ParsedProfile | None = None
        inference_outputs: Any | None = None
        try:
            compile_job = (
                self.get_job(candidate.compile_job_id)
                if resume and candidate.compile_job_id
                else self.submit_compile(request, device_name, device)
            )
            candidate.compile_job_id = str(compile_job.job_id)
            candidate.compile_job_url = str(getattr(compile_job, "url", "")) or None
            candidate.status = "compile_submitted"
            manifest.upsert(candidate)
            self._wait(compile_job, f"compile {request.stage_name}")
            target_model = compile_job.get_target_model()
            if target_model is None:
                raise CompileError(f"Compile job {compile_job.job_id} returned no target model")
            candidate.status = "compiled"
            manifest.upsert(candidate)
            if not skip_download:
                model_suffix = (
                    ".dlc"
                    if request.target_runtime == "qnn_dlc" or request.link_options is not None
                    else ".bin"
                )
                downloaded = download_target_model(
                    compile_job,
                    run_dir
                    / "models"
                    / f"{request.stage_name}-{variant}-compiled{model_suffix}",
                )
                if downloaded:
                    candidate.artifact_paths.append(str(downloaded))
                    manifest.upsert(candidate)

            if request.link_options is not None:
                link_job = (
                    self.get_job(candidate.link_job_id)
                    if resume and candidate.link_job_id
                    else self.submit_link(
                        target_model,
                        device_name,
                        request.stage_name,
                        request.link_options,
                        device,
                    )
                )
                candidate.link_job_id = str(link_job.job_id)
                candidate.link_job_url = str(getattr(link_job, "url", "")) or None
                candidate.status = "link_submitted"
                manifest.upsert(candidate)
                self._wait(link_job, f"link {request.stage_name}")
                target_model = link_job.get_target_model()
                if target_model is None:
                    raise CompileError(f"Link job {link_job.job_id} returned no target model")
                candidate.status = "linked"
                manifest.upsert(candidate)
                if not skip_download:
                    linked = download_target_model(
                        link_job,
                        run_dir
                        / "models"
                        / f"{request.stage_name}-{variant}-voice-ai.bin",
                    )
                    if linked:
                        candidate.artifact_paths.append(str(linked))
                        manifest.upsert(candidate)

            profile_job = (
                self.get_job(candidate.profile_job_id)
                if resume and candidate.profile_job_id
                else self.submit_profile(
                    target_model,
                    device_name,
                    request.stage_name,
                    request.profile_options,
                    device,
                )
            )
            candidate.profile_job_id = str(profile_job.job_id)
            candidate.profile_job_url = str(getattr(profile_job, "url", "")) or None
            candidate.status = "profile_submitted"
            manifest.upsert(candidate)
            self._wait(profile_job, f"profile {request.stage_name}")
            profile_path = run_dir / "profiles" / f"{request.stage_name}-{variant}.json"
            raw_profile = download_profile(profile_job, profile_path)
            parsed = parse_profile(raw_profile)
            candidate.artifact_paths.append(str(profile_path))
            candidate.status = "profiled"
            manifest.upsert(candidate)

            if inference_inputs is not None and not skip_inference:
                inference_job = (
                    self.get_job(candidate.inference_job_id)
                    if resume and candidate.inference_job_id
                    else self.submit_inference(
                        target_model,
                        device_name,
                        request.stage_name,
                        inference_inputs,
                        request.profile_options,
                        device,
                    )
                )
                candidate.inference_job_id = str(inference_job.job_id)
                candidate.inference_job_url = str(getattr(inference_job, "url", "")) or None
                candidate.status = "inference_submitted"
                manifest.upsert(candidate)
                self._wait(inference_job, f"inference {request.stage_name}")
                download_output = getattr(inference_job, "download_output_data", None)
                if callable(download_output):
                    inference_outputs = download_output()
                    _write_output_metadata(
                        run_dir
                        / "inference_outputs"
                        / f"{request.stage_name}-{variant}.metadata.json",
                        inference_outputs,
                    )
                if skip_download:
                    inference_outputs = inference_outputs if inference_outputs is not None else True
                else:
                    output_path = (
                        run_dir
                        / "inference_outputs"
                        / f"{request.stage_name}-{variant}.h5"
                    )
                    downloaded_output = download_inference_output(inference_job, output_path)
                    if downloaded_output:
                        candidate.artifact_paths.append(str(downloaded_output))
                candidate.status = "inferred"
                manifest.upsert(candidate)
            else:
                candidate.status = "success"
                manifest.upsert(candidate)
        except Exception as error:
            candidate.status = (
                "exported_but_not_compiled"
                if candidate.compile_job_id and not candidate.profile_job_id
                else "failed"
            )
            candidate.errors.append(
                {"category": classify_exception(error), "summary": str(error)[:2000]}
            )
            manifest.upsert(candidate)
            for job_id in (
                candidate.compile_job_id,
                candidate.link_job_id,
                candidate.profile_job_id,
                candidate.inference_job_id,
            ):
                if not job_id or skip_download:
                    continue
                try:
                    job = self.get_job(job_id)
                    paths = download_job_artifacts(job, run_dir / "logs" / job_id)
                    candidate.artifact_paths.extend(map(str, paths))
                except Exception as download_error:
                    candidate.errors.append(
                        {
                            "category": classify_exception(download_error),
                            "summary": str(download_error)[:2000],
                        }
                    )
            manifest.upsert(candidate)
            category = classify_exception(error)
            error_type: type[XRBenchError] = (
                ProfileError if category == "profile_failure" else CompileError
            )
            if category == "inference_mismatch":
                error_type = InferenceMismatchError
            raise error_type(str(error), cause=error) from error

        return JobExecutionResult(candidate, parsed, raw_profile, inference_outputs)


def _write_output_metadata(path: Path, outputs: Any) -> None:
    """Persist names/shapes/dtypes only, never large tensor payloads."""

    import json

    metadata: dict[str, Any] = {}
    if isinstance(outputs, Mapping):
        for name, values in outputs.items():
            entries = values if isinstance(values, list | tuple) else [values]
            metadata[str(name)] = [
                {
                    "shape": list(getattr(value, "shape", ())),
                    "dtype": str(getattr(value, "dtype", type(value).__name__)),
                }
                for value in entries
            ]
    else:
        metadata["result"] = {"type": type(outputs).__name__}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
