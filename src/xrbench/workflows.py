"""Top-level track orchestration used by the CLI."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from xrbench.config import BenchConfig
from xrbench.devices import resolve_device
from xrbench.hub.client import AIHubClient
from xrbench.hub.jobs import format_plan
from xrbench.metrics import StageProfile
from xrbench.paths import create_run_dir, project_root
from xrbench.reports import environment_metadata, generate_reports
from xrbench.tts.local_reference import LocalTTSResult, load_sentences, run_local_suite
from xrbench.tts.piper_adapter import PiperAdapter
from xrbench.tts.prepare import prepare_tts
from xrbench.tts.profile import (
    load_local_results,
    profile_tts_components,
    save_tts_failure_report,
    tts_plan,
)
from xrbench.tts.report import derive_tts_metrics, expand_profiles_by_text
from xrbench.vlm.architecture_inspector import write_report as write_architecture_report
from xrbench.vlm.export import export_all, load_exports
from xrbench.vlm.local_reference import validate_vision_wrappers
from xrbench.vlm.profile import profile_vlm_exports, vlm_plan
from xrbench.vlm.report import derive_vlm_metrics
from xrbench.vlm.smolvlm_adapter import SmolVLMAdapter

LOGGER = logging.getLogger(__name__)


def initialize_run(
    config: BenchConfig,
    benchmark: str,
    output_dir: str | Path | None,
) -> tuple[Path, dict[str, Any]]:
    run_dir = create_run_dir(config, benchmark, output_dir)
    config.dump_yaml(run_dir / "config.resolved.yaml")
    environment = environment_metadata(project_root())
    _json_write(run_dir / "environment.json", environment)
    manifest_path = run_dir / "job_manifest.json"
    if not manifest_path.exists():
        _json_write(manifest_path, {"schema_version": 2, "jobs": []})
    return run_dir, environment


def run_tts_workflow(
    action: str,
    config: BenchConfig,
    *,
    output_dir: str | Path | None,
    remote_enabled: bool,
    dry_run: bool,
    resume: bool,
    force_resubmit: bool,
    skip_inference: bool,
    skip_download: bool,
) -> tuple[int, Path]:
    run_dir, environment = initialize_run(config, "tts", output_dir)
    section = config.section("tts")
    model_id = str(section.get("model_id", "pipertts_en"))
    component_names = list(map(str, section.get("required_components", [])))
    requested = config.device.requested_name
    if dry_run or (action in {"profile", "infer", "all"} and not remote_enabled):
        plans = tts_plan(
            component_names,
            model_id=model_id,
            device_name=requested,
            skip_inference=skip_inference or action == "profile",
        )
        print(format_plan(plans))
        print("Dry-run: no Qualcomm AI Hub jobs were submitted.")
        _write_empty_reports(run_dir, environment, config)
        return 0, run_dir

    adapter: PiperAdapter | None = None
    components: Any = None
    local_results: list[LocalTTSResult] = []
    profiles: list[StageProfile] = []
    failures: list[dict[str, str]] = []

    if action in {"prepare", "local", "profile", "infer", "all"}:
        adapter = PiperAdapter.from_pretrained(synthesis_only=action == "local")
    if action in {"prepare", "profile", "infer", "all"}:
        assert adapter is not None
        components = prepare_tts(run_dir / "models" / "sources", adapter)
        LOGGER.info("Prepared %d official PiperTTS components", len(components))
    if action in {"local", "profile", "infer", "all"}:
        assert adapter is not None
        sentence_path = _project_path(str(section["sentences_file"]))
        local_results = run_local_suite(
            load_sentences(sentence_path),
            run_dir / "local_reference",
            adapter=adapter,
        )
    elif (run_dir / "local_reference" / "local_metrics.json").exists():
        local_results = load_local_results(run_dir / "local_reference" / "local_metrics.json")

    device_data: dict[str, Any] | None = None
    if action in {"profile", "infer", "all"}:
        assert adapter is not None
        assert components is not None
        hub = AIHubClient(
            remote_enabled=remote_enabled,
            timeout_seconds=config.remote.timeout_seconds,
            retries=config.remote.retries,
            retry_backoff_seconds=config.remote.retry_backoff_seconds,
        )
        resolved = resolve_device(hub.client, config.device)
        device_data = resolved.to_dict()
        _json_write(run_dir / "device.json", device_data)
        profiles, failures = profile_tts_components(
            config,
            run_dir,
            resolved.name,
            hub,
            components,
            adapter=adapter,
            device_os=resolved.os,
            hub_device=resolved.hub_device,
            resume=resume,
            force_resubmit=force_resubmit,
            skip_inference=skip_inference or action == "profile",
            skip_download=skip_download,
        )
        save_tts_failure_report(run_dir, failures)
    elif (run_dir / "metrics.json").exists() and action == "report":
        profiles = _load_profiles(run_dir / "metrics.json")
        device_data = _read_optional_json(run_dir / "device.json")

    derived = {"tts": derive_tts_metrics(profiles, local_results)}
    already_expanded = any("profile_reuse_note" in item.input_specs for item in profiles)
    report_profiles = (
        list(profiles)
        if already_expanded
        else expand_profiles_by_text(profiles, local_results)
    )
    generate_reports(
        run_dir,
        report_profiles,
        environment,
        device_data,
        config.data,
        derived,
        failures,
    )
    required = set(component_names)
    required_failed = any(
        item.stage in required and item.status not in {"success", "profiled", "inferred", "downloaded"}
        for item in profiles
    )
    return (1 if required_failed else 0), run_dir


def run_vlm_workflow(
    action: str,
    config: BenchConfig,
    *,
    output_dir: str | Path | None,
    remote_enabled: bool,
    dry_run: bool,
    resume: bool,
    force_resubmit: bool,
    skip_inference: bool,
    skip_download: bool,
) -> tuple[int, Path]:
    run_dir, environment = initialize_run(config, "vlm", output_dir)
    section = config.section("vlm")
    model_id = str(section["model_id"])
    prompt_lengths = list(map(int, section["prompt_lengths"]))
    context_lengths = list(map(int, section["decode_context_lengths"]))
    requested = config.device.requested_name
    if dry_run or (action in {"profile", "infer", "all"} and not remote_enabled):
        plans = vlm_plan(
            None,
            model_id=model_id,
            device_name=requested,
            prompt_lengths=prompt_lengths,
            context_lengths=context_lengths,
            skip_inference=skip_inference or action == "profile",
            precision_modes=list(
                map(str, config.section("precision").get("modes", ["float"]))
            ),
        )
        print(format_plan(plans))
        print("Dry-run: no Qualcomm AI Hub jobs were submitted.")
        _write_empty_reports(run_dir, environment, config)
        return 0, run_dir

    adapter: SmolVLMAdapter | None = None
    sample: Any = None
    exports: Any = None
    profiles: list[StageProfile] = []
    failures: list[dict[str, str]] = []
    image_path = _project_path(str(section["sample_image"]))

    if action in {"inspect", "prepare", "export", "local", "profile", "infer", "all"}:
        adapter = SmolVLMAdapter.from_pretrained(model_id)
        sample = adapter.vision_sample(image_path)
        write_architecture_report(
            adapter.architecture_report(sample), run_dir / "local_reference"
        )
    if action in {"export", "profile", "infer", "all"}:
        assert adapter is not None
        assert sample is not None
        exports = export_all(
            adapter,
            sample,
            run_dir / "models" / "sources",
            prompt=str(section["prompt"]),
            prompt_lengths=prompt_lengths,
            context_lengths=context_lengths,
        )
    elif (run_dir / "models" / "sources" / "exports.json").exists():
        exports = load_exports(run_dir / "models" / "sources" / "exports.json")
    if action in {"local", "all"}:
        assert adapter is not None
        assert sample is not None
        parity = validate_vision_wrappers(
            adapter,
            sample,
            run_dir / "local_reference",
            atol=float(config.section("validation").get("atol", 1e-3)),
            rtol=float(config.section("validation").get("rtol", 1e-2)),
        )
        for item in parity:
            if not item.passed:
                failures.append(
                    {
                        "stage": item.stage,
                        "category": "inference_mismatch",
                        "status": "failed",
                        "summary": f"Local wrapper parity max abs error {item.max_absolute_error}",
                    }
                )

    device_data: dict[str, Any] | None = None
    if action in {"profile", "infer", "all"}:
        assert adapter is not None
        assert sample is not None
        assert exports is not None
        hub = AIHubClient(
            remote_enabled=remote_enabled,
            timeout_seconds=config.remote.timeout_seconds,
            retries=config.remote.retries,
            retry_backoff_seconds=config.remote.retry_backoff_seconds,
        )
        resolved = resolve_device(hub.client, config.device)
        device_data = resolved.to_dict()
        _json_write(run_dir / "device.json", device_data)
        remote_profiles, remote_failures = profile_vlm_exports(
            config,
            run_dir,
            resolved.name,
            hub,
            exports,
            adapter=adapter,
            sample=sample,
            device_os=resolved.os,
            hub_device=resolved.hub_device,
            resume=resume,
            force_resubmit=force_resubmit,
            skip_inference=skip_inference or action == "profile",
            skip_download=skip_download,
        )
        profiles.extend(remote_profiles)
        failures.extend(remote_failures)
    elif action == "report" and (run_dir / "metrics.json").exists():
        profiles = _load_profiles(run_dir / "metrics.json")
        device_data = _read_optional_json(run_dir / "device.json")

    output_lengths = list(
        map(int, config.section("report").get("generation_output_tokens", [8, 16, 32, 64]))
    )
    derived = {"vlm": derive_vlm_metrics(profiles, output_lengths)}
    generate_reports(
        run_dir, profiles, environment, device_data, config.data, derived, failures
    )
    mandatory_failed = any(
        item.stage in {"vision_encoder", "vision_projector"}
        and item.status not in {"success", "profiled", "inferred", "downloaded"}
        for item in profiles
    ) or any(
        str(item.get("stage", "")).split("/", 1)[0]
        in {"vision_encoder", "vision_projector"}
        and item.get("status") == "failed"
        for item in failures
    )
    return (1 if mandatory_failed else 0), run_dir


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root() / path).resolve()


def _json_write(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _load_profiles(path: Path) -> list[StageProfile]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [StageProfile(**item) for item in data.get("measured_stage_profiles", [])]


def _write_empty_reports(
    run_dir: Path, environment: dict[str, Any], config: BenchConfig
) -> None:
    generate_reports(run_dir, [], environment, None, config.data, {})
