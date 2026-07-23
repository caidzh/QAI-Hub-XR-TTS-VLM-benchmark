"""Prepare deterministic VLM inputs and architecture diagnostics."""

from __future__ import annotations

from pathlib import Path

from xrbench.vlm.architecture_inspector import ArchitectureReport, write_report
from xrbench.vlm.smolvlm_adapter import SmolVLMAdapter, VisionSample


def prepare_vlm(
    model_id: str,
    image_path: Path,
    output_dir: Path,
    *,
    adapter: SmolVLMAdapter | None = None,
) -> tuple[SmolVLMAdapter, VisionSample, ArchitectureReport]:
    adapter = adapter or SmolVLMAdapter.from_pretrained(model_id)
    sample = adapter.vision_sample(image_path)
    report = adapter.architecture_report(sample)
    write_report(report, output_dir)
    return adapter, sample, report
