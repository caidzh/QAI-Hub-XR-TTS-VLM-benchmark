from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from xrbench.config import DeviceConfig
from xrbench.devices import resolve_device
from xrbench.errors import DeviceUnavailableError


@dataclass
class FakeDevice:
    name: str
    os: str = "13"
    attributes: list[str] = field(
        default_factory=lambda: ["chipset:snapdragon-8-gen-1", "framework:qnn"]
    )


class FakeClient:
    def __init__(self, devices: list[FakeDevice]) -> None:
        self.devices = devices

    def get_devices(self, name: str = "", os: str = "", attributes=None):
        return [device for device in self.devices if not name or device.name == name]


def test_exact_device_selection_is_deterministic() -> None:
    client = FakeClient(
        [
            FakeDevice("Samsung Galaxy S22 5G", "14"),
            FakeDevice("Samsung Galaxy S22 5G", "13"),
        ]
    )
    result = resolve_device(client, DeviceConfig())
    assert result.name == "Samsung Galaxy S22 5G"
    assert result.os == "13"
    assert result.selection_reason.startswith("exact requested")
    assert result.chipset_attributes == ("chipset:snapdragon-8-gen-1",)


def test_preferred_fallback_beats_family() -> None:
    client = FakeClient(
        [FakeDevice("Samsung Galaxy S22 Ultra"), FakeDevice("Samsung Galaxy S22")]
    )
    result = resolve_device(client, DeviceConfig(requested_name="Missing"))
    assert result.name == "Samsung Galaxy S22"
    assert "preferred S22 fallback" in result.selection_reason


def test_family_fallback_warns_and_is_explicit(caplog: pytest.LogCaptureFixture) -> None:
    client = FakeClient([FakeDevice("Samsung Galaxy S22 Ultra")])
    result = resolve_device(client, DeviceConfig(requested_name="Missing"))
    assert result.name == "Samsung Galaxy S22 Ultra"
    assert "family fallback" in caplog.text.lower()


def test_unrelated_device_is_not_silently_selected() -> None:
    client = FakeClient([FakeDevice("Samsung Galaxy S25")])
    with pytest.raises(DeviceUnavailableError, match="No permitted"):
        resolve_device(client, DeviceConfig())
