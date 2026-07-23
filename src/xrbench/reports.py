"""JSON, CSV, and Markdown report generation."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from xrbench.metrics import StageProfile

LIMITATIONS = (
    "Qualcomm AI Hub profile latency is a model microbenchmark.",
    "Preprocessing and Python orchestration are not part of remote model inference latency.",
    "Separately profiled subgraphs do not measure inter-graph transfer overhead.",
    "Component-sum TTFT is an estimate, not true end-to-end TTFT.",
    "Samsung Galaxy S22 is being used as a similar device and is not a Meta Quest measurement.",
    "Real Quest XR workloads may differ due to OS, firmware, clocks, thermals, memory "
    "pressure, rendering, tracking, and compositor contention.",
)


def package_versions(names: Sequence[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def repository_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def environment_metadata(root: Path) -> dict[str, Any]:
    return {
        "date": datetime.now(timezone.utc).isoformat(),
        "repository_commit": repository_commit(root),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": package_versions(
            ("qai-hub", "qai-hub-models", "torch", "transformers", "onnx", "numpy")
        ),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_metrics_json(path: Path, profiles: Iterable[StageProfile], derived: Mapping[str, Any]) -> None:
    _write_json(
        path,
        {
            "schema_version": 1,
            "measured_stage_profiles": [profile.to_dict() for profile in profiles],
            "derived_estimates": dict(derived),
        },
    )


def write_metrics_csv(path: Path, profiles: Iterable[StageProfile]) -> None:
    rows = [profile.to_dict() for profile in profiles]
    fields = list(asdict(_empty_stage()).keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, dict | list)
                    else value
                    for key, value in row.items()
                }
            )


def _empty_stage() -> StageProfile:
    return StageProfile(
        benchmark="",
        model_id="",
        stage="",
        device_name="",
        device_os=None,
        runtime="",
        precision="",
        input_variant="",
        input_specs={},
    )


def _fmt(value: float | None, suffix: str = "") -> str:
    return "—" if value is None else f"{value:.3f}{suffix}"


def _memory(profile: StageProfile) -> str:
    low, high = profile.inference_peak_mib_low, profile.inference_peak_mib_high
    if low is None and high is None:
        return "—"
    if low == high or low is None:
        return _fmt(high, " MiB")
    return f"{_fmt(low)}-{_fmt(high)} MiB"


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(output)


def write_markdown_report(
    path: Path,
    profiles: Sequence[StageProfile],
    environment: Mapping[str, Any],
    device: Mapping[str, Any] | None,
    config: Mapping[str, Any],
    derived: Mapping[str, Any],
    failures: Sequence[Mapping[str, Any]] = (),
) -> None:
    requested = _mapping(config.get("device")).get("requested_name")
    resolved = device.get("name") if device else None
    lines = [
        "# Qualcomm AI Hub XR Latency Report",
        "",
        "## Environment",
        "",
        f"- Date: {environment.get('date', 'unknown')}",
        f"- Repository commit: {environment.get('repository_commit') or 'not available'}",
        f"- Python: {environment.get('python_version', 'unknown')}",
        f"- Relevant packages: {json.dumps(environment.get('packages', {}), sort_keys=True)}",
        f"- Requested device: {requested or 'unknown'}",
        f"- Resolved device: {resolved or 'not resolved'}",
        f"- Resolved device OS: {device.get('os') if device else 'not resolved'}",
        f"- Device attributes: {json.dumps(device.get('attributes', []) if device else [])}",
        f"- TTS model: {_mapping(config.get('tts')).get('model_id', 'unknown')}",
        f"- TTS runtime: {_mapping(config.get('tts')).get('target_runtime', 'unknown')}",
        f"- VLM model: {_mapping(config.get('vlm')).get('model_id', 'unknown')}",
        f"- VLM runtime: {_mapping(config.get('vlm')).get('target_runtime', 'unknown')}",
        f"- Precision modes: {_mapping(config.get('precision')).get('modes', [])}",
        "",
    ]
    tts = [item for item in profiles if item.benchmark == "tts"]
    vlm = [item for item in profiles if item.benchmark == "vlm"]
    if tts:
        lines.extend(
            [
                "## TTS measured component metrics",
                "",
                _markdown_table(
                    (
                        "text variant",
                        "sequence length",
                        "component",
                        "status",
                        "inference latency",
                        "first load",
                        "warm load",
                        "peak memory",
                        "placement",
                    ),
                    (
                        (
                            item.input_variant,
                            str(item.input_specs.get("actual_sequence_length", "—")),
                            item.stage,
                            item.status,
                            _fmt(item.estimated_inference_ms, " ms"),
                            _fmt(item.first_load_ms, " ms"),
                            _fmt(item.warm_load_ms, " ms"),
                            _memory(item),
                            json.dumps(item.placement or {}, sort_keys=True),
                        )
                        for item in tts
                    ),
                ),
                "",
                "### TTS derived component-sum estimates",
                "",
                "These combine separately measured neural component profiles with invocation "
                "counts observed in the local floating-point reference pipeline.",
                "",
                "```json",
                json.dumps(derived.get("tts", {}), indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    if vlm:
        lines.extend(
            [
                "## VLM measured component metrics",
                "",
                _markdown_table(
                    (
                        "stage",
                        "prompt length",
                        "context length",
                        "status",
                        "inference latency",
                        "first load",
                        "warm load",
                        "peak memory",
                        "placement",
                    ),
                    (
                        (
                            item.stage,
                            str(item.input_specs.get("prompt_length", "—")),
                            str(item.input_specs.get("context_length", "—")),
                            item.status,
                            _fmt(item.estimated_inference_ms, " ms"),
                            _fmt(item.first_load_ms, " ms"),
                            _fmt(item.warm_load_ms, " ms"),
                            _memory(item),
                            json.dumps(item.placement or {}, sort_keys=True),
                        )
                        for item in vlm
                    ),
                ),
                "",
                "### VLM derived estimates",
                "",
                "The value named `component-sum estimated TTFT` is not an end-to-end "
                "measurement. Decode estimates remain separated by KV-cache context length.",
                "",
                "```json",
                json.dumps(derived.get("vlm", {}), indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    if failures:
        lines.extend(
            [
                "## Failures",
                "",
                _markdown_table(
                    ("stage", "category", "status", "summary"),
                    (
                        (
                            str(item.get("stage", "unknown")),
                            str(item.get("category", "unknown")),
                            str(item.get("status", "failed")),
                            str(item.get("summary", "")).replace("|", "\\|"),
                        )
                        for item in failures
                    ),
                ),
                "",
            ]
        )
    lines.extend(["## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in LIMITATIONS)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def generate_reports(
    run_dir: Path,
    profiles: Sequence[StageProfile],
    environment: Mapping[str, Any],
    device: Mapping[str, Any] | None,
    config: Mapping[str, Any],
    derived: Mapping[str, Any],
    failures: Sequence[Mapping[str, Any]] = (),
) -> None:
    write_metrics_json(run_dir / "metrics.json", profiles, derived)
    write_metrics_csv(run_dir / "metrics.csv", profiles)
    write_markdown_report(
        run_dir / "report.md", profiles, environment, device, config, derived, failures
    )
