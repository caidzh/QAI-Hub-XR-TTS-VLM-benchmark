"""Atomic, resumable job-manifest persistence."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from xrbench.config import checksum_mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobRecord:
    benchmark: str
    stage: str
    variant: str
    source_model_checksum: str
    stage_config_checksum: str
    device_name: str
    runtime: str
    precision: str
    input_specs: dict[str, Any]
    source_model_path: str | None = None
    configuration_checksum: str | None = None
    device_os: str | None = None
    compile_job_id: str | None = None
    compile_job_url: str | None = None
    link_job_id: str | None = None
    link_job_url: str | None = None
    profile_job_id: str | None = None
    profile_job_url: str | None = None
    inference_job_id: str | None = None
    inference_job_url: str | None = None
    status: str = "planned"
    artifact_paths: list[str] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    record_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    @property
    def key(self) -> str:
        payload = {
            "benchmark": self.benchmark,
            "stage": self.stage,
            "variant": self.variant,
            "source_model_checksum": self.source_model_checksum,
            "stage_config_checksum": self.stage_config_checksum,
            "device_name": self.device_name,
            "device_os": self.device_os,
            "runtime": self.runtime,
            "precision": self.precision,
            "input_specs": self.input_specs,
        }
        return checksum_mapping(payload)

    def compatible_with(self, other: JobRecord) -> bool:
        return self.key == other.key


class JobManifest:
    """Manifest whose every mutation is flushed with an atomic replace."""

    def __init__(self, path: Path, records: Iterable[JobRecord] = ()) -> None:
        self.path = path
        self.records: list[JobRecord] = list(records)

    @classmethod
    def load(cls, path: Path) -> JobManifest:
        if not path.exists():
            return cls(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        records = [JobRecord(**item) for item in data.get("jobs", [])]
        return cls(path, records)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 2,
            "updated_at": utc_now(),
            "jobs": [asdict(record) for record in self.records],
        }
        fd, temp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def upsert(self, record: JobRecord) -> JobRecord:
        record.updated_at = utc_now()
        for index, existing in enumerate(self.records):
            if existing.record_id == record.record_id:
                self.records[index] = record
                self.save()
                return record
        self.records.append(record)
        self.save()
        return record

    def update(self, record: JobRecord, **changes: Any) -> JobRecord:
        for key, value in changes.items():
            if not hasattr(record, key):
                raise AttributeError(key)
            setattr(record, key, value)
        return self.upsert(record)

    def find_compatible(
        self, request: JobRecord, *, successful_only: bool = True
    ) -> JobRecord | None:
        valid_statuses = {"success", "profiled", "inferred", "downloaded"}
        matches = [
            record for record in reversed(self.records) if record.compatible_with(request)
        ]
        for record in matches:
            if record.status in valid_statuses:
                return record
        if successful_only:
            return None
        resumable_statuses = {
            "planned",
            "compile_submitted",
            "compiled",
            "link_submitted",
            "linked",
            "profile_submitted",
            "inference_submitted",
        }
        for record in matches:
            if record.status in resumable_statuses:
                return record
        return None
