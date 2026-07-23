"""Fail-soft remote profiling for official PiperTTS components."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from xrbench.config import BenchConfig
from xrbench.errors import classify_exception
from xrbench.hub.client import AIHubClient, CompileRequest
from xrbench.hub.jobs import PlannedJob
from xrbench.manifest import JobManifest
from xrbench.metrics import StageProfile
from xrbench.tts.local_reference import LocalTTSResult
from xrbench.tts.piper_adapter import PiperAdapter, PreparedComponent
from xrbench.validation import compare_arrays

LOGGER = logging.getLogger(__name__)

KNOWN_TTS_INPUT_SPECS: dict[str, dict[str, object]] = {
    "encoder": {
        "x": {"shape": [1, 512], "dtype": "int32"},
        "x_lengths": {"shape": [1], "dtype": "int32"},
    },
    "sdp": {
        "x_encoded": {"shape": [1, 192, 512], "dtype": "float32"},
        "x_mask": {"shape": [1, 1, 512], "dtype": "float32"},
        "length_scale": {"shape": [1], "dtype": "float32"},
        "noise_scale_w": {"shape": [1], "dtype": "float32"},
    },
    "flow": {
        "m_p": {"shape": [1, 192, 512], "dtype": "float32"},
        "logs_p": {"shape": [1, 192, 512], "dtype": "float32"},
        "y_mask": {"shape": [1, 1, 1536], "dtype": "float32"},
        "attn_squeezed": {"shape": [1, 1536, 512], "dtype": "float32"},
        "noise_scale": {"shape": [1], "dtype": "float32"},
    },
    "decoder": {"z": {"shape": [1, 192, 64], "dtype": "float32"}},
    "charsiu_encoder": {
        "input_ids": {"shape": [1, 50], "dtype": "int32"},
        "encoder_attention_mask": {"shape": [1, 50], "dtype": "int32"},
    },
    "charsiu_decoder": {
        "input_ids": {"shape": [1, 1], "dtype": "int32"},
        "encoder_attention_mask": {"shape": [1, 50], "dtype": "int32"},
        "position": {"shape": [1, 1], "dtype": "int32"},
        "flattened_cache": {
            "description": "4 tensors/block: self K/V [1,H,49,D], cross K/V [1,H,50,D]"
        },
    },
}


def tts_plan(
    components: Sequence[PreparedComponent] | Sequence[str],
    *,
    model_id: str,
    device_name: str,
    runtime: str = "voice_ai",
    precision: str = "float",
    skip_inference: bool = False,
) -> list[PlannedJob]:
    plans: list[PlannedJob] = []
    for component in components:
        if isinstance(component, str):
            name = component
            source_format = "official_qai_hub_models_export"
            input_specs = KNOWN_TTS_INPUT_SPECS.get(name, {})
        else:
            name = component.name
            source_format = component.source_format
            input_specs = {
                key: {"shape": list(value[0]), "dtype": value[1]}
                for key, value in component.input_specs.items()
            }
        plans.append(
            PlannedJob(
                model=model_id,
                stage=name,
                variant="fixed-official-shape",
                target_device=device_name,
                source_format=source_format,
                target_runtime=runtime,
                precision=precision,
                input_shapes=input_specs,
                inference_jobs=0 if skip_inference else 1,
            )
        )
    return plans


def profile_tts_components(
    config: BenchConfig,
    run_dir: Path,
    device_name: str,
    hub_client: AIHubClient,
    components: Sequence[PreparedComponent],
    *,
    adapter: PiperAdapter,
    device_os: str | None = None,
    hub_device: Any | None = None,
    resume: bool,
    force_resubmit: bool,
    skip_inference: bool,
    skip_download: bool,
) -> tuple[list[StageProfile], list[dict[str, str]]]:
    manifest = JobManifest.load(run_dir / "job_manifest.json")
    model_id = str(config.section("tts").get("model_id", "pipertts_en"))
    results: list[StageProfile] = []
    failures: list[dict[str, str]] = []
    for component in components:
        request = CompileRequest(
            stage_name=component.name,
            source_model_path=component.source_path,
            source_format=component.source_format,
            target_runtime="voice_ai",
            precision="float",
            input_specs=component.input_specs,
            compile_options=component.compile_options,
            link_options=component.link_options,
            profile_options=component.profile_options,
        )
        try:
            sample_inputs = adapter.sample_inputs(component.name)
            local_outputs = adapter.reference_outputs(component.name, sample_inputs)
            execution = hub_client.execute(
                request,
                benchmark="tts",
                variant="fixed-official-shape",
                device_name=device_name,
                manifest=manifest,
                run_dir=run_dir,
                device_os=device_os,
                device=hub_device,
                configuration_checksum=config.checksum(),
                inference_inputs=sample_inputs,
                resume=resume,
                force_resubmit=force_resubmit,
                skip_inference=skip_inference,
                skip_download=skip_download,
            )
            validation = _validate_outputs(
                local_outputs,
                execution.inference_outputs,
                atol=float(config.section("validation").get("atol", 1e-3)),
                rtol=float(config.section("validation").get("rtol", 1e-2)),
            )
            if validation is not None:
                validation_path = (
                    run_dir
                    / "inference_outputs"
                    / f"{component.name}-fixed-official-shape.validation.json"
                )
                validation_path.write_text(
                    json.dumps(validation, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                if not bool(validation["within_tolerance"]):
                    failures.append(
                        {
                            "stage": component.name,
                            "category": "inference_mismatch",
                            "status": "inference_mismatch",
                            "summary": "Hosted output exceeded configured tolerance",
                        }
                    )
            result = _stage_profile(
                model_id,
                component,
                device_name,
                device_os,
                execution.record,
                execution.parsed_profile,
            )
            result.validation = validation
            if validation is not None and not bool(validation["within_tolerance"]):
                result.status = "inference_mismatch"
            results.append(result)
        except Exception as error:
            category = classify_exception(error)
            failures.append(
                {
                    "stage": component.name,
                    "category": category,
                    "status": "failed",
                    "summary": str(error)[:2000],
                }
            )
            LOGGER.error("TTS stage %s failed: %s", component.name, error)
            record = next(
                (
                    item
                    for item in reversed(manifest.records)
                    if item.stage == component.name and item.benchmark == "tts"
                ),
                None,
            )
            results.append(
                StageProfile(
                    benchmark="tts",
                    model_id=model_id,
                    stage=component.name,
                    device_name=device_name,
                    device_os=device_os,
                    runtime="voice_ai",
                    precision="float",
                    input_variant="fixed-official-shape",
                    input_specs={
                        key: {"shape": value[0], "dtype": value[1]}
                        for key, value in component.input_specs.items()
                    },
                    compile_job_id=record.compile_job_id if record else None,
                    profile_job_id=record.profile_job_id if record else None,
                    compile_job_url=record.compile_job_url if record else None,
                    profile_job_url=record.profile_job_url if record else None,
                    status=record.status if record else "failed",
                    error_summary=str(error)[:2000],
                )
            )
    return results, failures


def _stage_profile(
    model_id: str,
    component: PreparedComponent,
    device_name: str,
    device_os: str | None,
    record: Any,
    parsed: Any,
) -> StageProfile:
    distribution = parsed.inference_distribution if parsed else None
    return StageProfile(
        benchmark="tts",
        model_id=model_id,
        stage=component.name,
        device_name=device_name,
        device_os=device_os,
        runtime="voice_ai",
        precision="float",
        input_variant="fixed-official-shape",
        input_specs={
            key: {"shape": value[0], "dtype": value[1]}
            for key, value in component.input_specs.items()
        },
        compile_job_id=record.compile_job_id,
        profile_job_id=record.profile_job_id,
        inference_job_id=record.inference_job_id,
        compile_job_url=record.compile_job_url,
        profile_job_url=record.profile_job_url,
        status=record.status,
        estimated_inference_ms=parsed.estimated_inference_ms if parsed else None,
        inference_min_ms=distribution.minimum if distribution else None,
        inference_mean_ms=distribution.mean if distribution else None,
        inference_median_ms=distribution.median if distribution else None,
        inference_p90_ms=distribution.p90 if distribution else None,
        inference_p95_ms=distribution.p95 if distribution else None,
        inference_p99_ms=distribution.p99 if distribution else None,
        inference_stddev_ms=distribution.stddev if distribution else None,
        inference_samples=distribution.samples if distribution else 0,
        first_load_ms=parsed.first_load_ms if parsed else None,
        warm_load_ms=parsed.warm_load_ms if parsed else None,
        compile_ms=parsed.compile_ms if parsed else None,
        inference_peak_mib_low=parsed.inference_peak_mib[0] if parsed else None,
        inference_peak_mib_high=parsed.inference_peak_mib[1] if parsed else None,
        inference_increase_mib_low=parsed.inference_increase_mib[0] if parsed else None,
        inference_increase_mib_high=parsed.inference_increase_mib[1] if parsed else None,
        first_load_peak_mib_low=parsed.first_load_peak_mib[0] if parsed else None,
        first_load_peak_mib_high=parsed.first_load_peak_mib[1] if parsed else None,
        first_load_increase_mib_low=(
            parsed.first_load_increase_mib[0] if parsed else None
        ),
        first_load_increase_mib_high=(
            parsed.first_load_increase_mib[1] if parsed else None
        ),
        warm_load_peak_mib_low=parsed.warm_load_peak_mib[0] if parsed else None,
        warm_load_peak_mib_high=parsed.warm_load_peak_mib[1] if parsed else None,
        warm_load_increase_mib_low=(
            parsed.warm_load_increase_mib[0] if parsed else None
        ),
        warm_load_increase_mib_high=(
            parsed.warm_load_increase_mib[1] if parsed else None
        ),
        compile_peak_mib_low=parsed.compile_peak_mib[0] if parsed else None,
        compile_peak_mib_high=parsed.compile_peak_mib[1] if parsed else None,
        compile_increase_mib_low=(
            parsed.compile_increase_mib[0] if parsed else None
        ),
        compile_increase_mib_high=(
            parsed.compile_increase_mib[1] if parsed else None
        ),
        placement=parsed.placement if parsed else None,
        cpu_fallback=parsed.cpu_fallback if parsed else None,
        runtime_layers=list(parsed.runtime_layers) if parsed else None,
    )


def load_local_results(path: Path) -> list[LocalTTSResult]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [LocalTTSResult(**item) for item in raw]


def save_tts_failure_report(run_dir: Path, failures: Sequence[Mapping[str, str]]) -> None:
    if not failures:
        return
    lines = ["# PiperTTS failure report", ""]
    for failure in failures:
        lines.extend(
            [
                f"## {failure.get('stage', 'unknown')}",
                "",
                f"- Category: `{failure.get('category', 'unknown')}`",
                f"- Status: `{failure.get('status', 'failed')}`",
                "",
                "```text",
                failure.get("summary", ""),
                "```",
                "",
            ]
        )
    (run_dir / "failure_reports" / "tts_failures.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _validate_outputs(
    local_outputs: Mapping[str, Any],
    hosted_outputs: Any,
    *,
    atol: float,
    rtol: float,
) -> dict[str, object] | None:
    if not isinstance(hosted_outputs, Mapping) or not hosted_outputs:
        return None
    per_output: dict[str, object] = {}
    all_within = True
    for index, (name, reference) in enumerate(local_outputs.items()):
        values = hosted_outputs.get(name)
        if values is None and index < len(hosted_outputs):
            values = list(hosted_outputs.values())[index]
        if values is None:
            continue
        actual = values[0] if isinstance(values, list | tuple) else values
        metrics = compare_arrays(reference, actual, logits="logits" in name)
        item = asdict(metrics)
        within = metrics.max_absolute_error <= atol or metrics.max_relative_error <= rtol
        item["within_tolerance"] = within
        all_within = all_within and within
        per_output[name] = item
    if not per_output:
        return None
    return {
        "absolute_tolerance": atol,
        "relative_tolerance": rtol,
        "within_tolerance": all_within,
        "outputs": per_output,
    }
