"""Configuration loading, validation, overrides, and stable checksums."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from xrbench.errors import ConfigurationError

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigurationError(f"Configuration root must be a mapping: {path}")
    return data


@dataclass(frozen=True)
class DeviceConfig:
    requested_name: str = "Samsung Galaxy S22 5G"
    allow_s22_family_fallback: bool = True
    allow_unrelated_fallback: bool = False


@dataclass(frozen=True)
class RemoteConfig:
    timeout_seconds: int = 3600
    retries: int = 2
    retry_backoff_seconds: float = 5.0
    skip_inference: bool = False
    skip_download: bool = False


@dataclass(frozen=True)
class BenchConfig:
    """Resolved configuration plus typed accessors for frequently used fields."""

    data: dict[str, Any]
    source_files: tuple[Path, ...]

    @property
    def device(self) -> DeviceConfig:
        raw = self.section("device")
        return DeviceConfig(
            requested_name=str(raw.get("requested_name", "Samsung Galaxy S22 5G")),
            allow_s22_family_fallback=bool(raw.get("allow_s22_family_fallback", True)),
            allow_unrelated_fallback=bool(raw.get("allow_unrelated_fallback", False)),
        )

    @property
    def remote(self) -> RemoteConfig:
        raw = self.section("remote")
        return RemoteConfig(
            timeout_seconds=int(raw.get("timeout_seconds", 3600)),
            retries=int(raw.get("retries", 2)),
            retry_backoff_seconds=float(raw.get("retry_backoff_seconds", 5)),
            skip_inference=bool(raw.get("skip_inference", False)),
            skip_download=bool(raw.get("skip_download", False)),
        )

    def section(self, name: str) -> dict[str, Any]:
        value = self.data.get(name, {})
        if not isinstance(value, dict):
            raise ConfigurationError(f"Configuration section '{name}' must be a mapping")
        return value

    def checksum(self) -> str:
        return checksum_mapping(self.data)

    def dump_yaml(self, path: Path) -> None:
        path.write_text(yaml.safe_dump(self.data, sort_keys=True), encoding="utf-8")


def load_config(
    config_path: str | Path | None = None,
    *,
    requested_device: str | None = None,
    output_dir: str | Path | None = None,
) -> BenchConfig:
    """Load defaults, deep-merge a track config, and apply explicit CLI overrides."""

    default_path = DEFAULT_CONFIG
    data = _read_yaml(default_path)
    sources = [default_path]
    if config_path is not None:
        user_path = Path(config_path).expanduser().resolve()
        if user_path != default_path:
            data = _deep_merge(data, _read_yaml(user_path))
            sources.append(user_path)
    if requested_device:
        data.setdefault("device", {})["requested_name"] = requested_device
    if output_dir:
        data.setdefault("paths", {})["output_dir"] = str(Path(output_dir).expanduser())
    validate_config(data)
    return BenchConfig(data=data, source_files=tuple(sources))


def validate_config(data: Mapping[str, Any]) -> None:
    device = data.get("device")
    if not isinstance(device, Mapping) or not str(device.get("requested_name", "")).strip():
        raise ConfigurationError("device.requested_name must be a non-empty string")
    modes = data.get("precision", {}).get("modes", [])  # type: ignore[union-attr]
    if not isinstance(modes, list) or not modes:
        raise ConfigurationError("precision.modes must contain at least one mode")
    unsupported = set(map(str, modes)) - {"float", "int8"}
    if unsupported:
        raise ConfigurationError(f"Unsupported precision modes: {sorted(unsupported)}")
    for key in ("prompt_lengths", "decode_context_lengths"):
        values = data.get("vlm", {}).get(key, [])  # type: ignore[union-attr]
        if not isinstance(values, list) or any(int(value) <= 0 for value in values):
            raise ConfigurationError(f"vlm.{key} must be a list of positive integers")


def checksum_mapping(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def checksum_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
