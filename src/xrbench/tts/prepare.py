"""Prepare official PiperTTS components for Workbench submission."""

from __future__ import annotations

from pathlib import Path

from xrbench.tts.piper_adapter import PiperAdapter, PreparedComponent


def prepare_tts(output_dir: Path, adapter: PiperAdapter | None = None) -> list[PreparedComponent]:
    adapter = adapter or PiperAdapter.from_pretrained()
    return adapter.prepare_components(output_dir)
