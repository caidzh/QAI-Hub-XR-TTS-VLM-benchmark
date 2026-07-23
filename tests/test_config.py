from __future__ import annotations

from pathlib import Path

import pytest

from xrbench.config import checksum_mapping, load_config
from xrbench.errors import ConfigurationError


def test_load_config_deep_merges_and_overrides(tmp_path: Path) -> None:
    path = tmp_path / "track.yaml"
    path.write_text(
        "device:\n  requested_name: Custom S22\nvlm:\n  prompt_lengths: [16]\n",
        encoding="utf-8",
    )
    config = load_config(path, requested_device="CLI Device", output_dir=tmp_path / "out")
    assert config.device.requested_name == "CLI Device"
    assert config.section("vlm")["prompt_lengths"] == [16]
    assert config.section("vlm")["decode_context_lengths"] == [32, 64, 128, 256]
    assert config.section("paths")["output_dir"] == str(tmp_path / "out")


def test_config_checksum_is_order_independent() -> None:
    assert checksum_mapping({"a": 1, "b": 2}) == checksum_mapping({"b": 2, "a": 1})


def test_config_rejects_unknown_precision(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("precision:\n  modes: [fp4]\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Unsupported precision"):
        load_config(path)
