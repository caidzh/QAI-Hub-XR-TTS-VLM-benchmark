"""Best-effort download of public AI Hub job artifacts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from xrbench.errors import DownloadError

LOGGER = logging.getLogger(__name__)


def download_job_artifacts(job: Any, output_dir: Path, *, strict: bool = False) -> list[Path]:
    """Download all SDK-supported results and logs, retaining partial downloads."""

    output_dir.mkdir(parents=True, exist_ok=True)
    before = {path.resolve() for path in output_dir.rglob("*") if path.is_file()}
    errors: list[Exception] = []
    download_results = getattr(job, "download_results", None)
    if callable(download_results):
        try:
            download_results(str(output_dir))
        except Exception as error:  # SDK/network errors must not erase prior stage results.
            errors.append(error)
            LOGGER.warning("Could not download complete results for %s: %s", job, error)
    download_logs = getattr(job, "download_job_logs", None)
    if callable(download_logs):
        try:
            download_logs(str(output_dir))
        except Exception as error:
            errors.append(error)
            LOGGER.warning("Could not download job logs for %s: %s", job, error)
    available_method = getattr(job, "get_available_artifacts", None)
    typed_download = getattr(job, "download_artifacts_for_type", None)
    if callable(available_method) and callable(typed_download):
        try:
            for artifact_type in available_method():
                typed_download(str(output_dir), artifact_type)
        except Exception as error:
            errors.append(error)
            LOGGER.warning("Could not download every available artifact for %s: %s", job, error)
    after = {path.resolve() for path in output_dir.rglob("*") if path.is_file()}
    paths = sorted(after - before)
    if strict and errors and not paths:
        raise DownloadError(f"Artifact download failed: {errors[-1]}", cause=errors[-1])
    return paths


def download_target_model(job: Any, output_path: Path) -> Path | None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    method = getattr(job, "download_target_model", None)
    if not callable(method):
        return None
    result = method(str(output_path))
    if result is None:
        return None
    return Path(str(result)).resolve()


def download_profile(job: Any, output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    method = getattr(job, "download_profile", None)
    if not callable(method):
        raise DownloadError("Profile job has no public download_profile method")
    profile = method()
    if not isinstance(profile, dict):
        raise DownloadError(f"Unexpected profile result type: {type(profile).__name__}")
    import json

    output_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return profile


def download_inference_output(job: Any, output_path: Path) -> Path | None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    method = getattr(job, "download_output_data", None)
    if not callable(method):
        return None
    result = method(str(output_path))
    return Path(str(result)).resolve() if result else None
