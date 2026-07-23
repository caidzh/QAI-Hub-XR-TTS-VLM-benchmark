"""Deterministic Qualcomm AI Hub hosted-device discovery."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from xrbench.config import DeviceConfig
from xrbench.errors import DeviceUnavailableError

LOGGER = logging.getLogger(__name__)
S22_5G = "Samsung Galaxy S22 5G"
S22 = "Samsung Galaxy S22"


class DeviceLike(Protocol):
    name: str
    os: str
    attributes: list[str]


class DeviceClient(Protocol):
    def get_devices(
        self, name: str = "", os: str = "", attributes: str | list[str] | None = None
    ) -> list[Any]: ...


@dataclass(frozen=True)
class ResolvedDevice:
    name: str
    os: str | None
    attributes: tuple[str, ...]
    chipset_attributes: tuple[str, ...]
    supported_frameworks: tuple[str, ...]
    selection_reason: str
    hub_device: Any

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("hub_device", None)
        return result


def _device_key(device: Any) -> tuple[str, str, tuple[str, ...]]:
    return (
        str(getattr(device, "name", "")),
        str(getattr(device, "os", "")),
        tuple(sorted(map(str, getattr(device, "attributes", []) or []))),
    )


def _build(device: Any, reason: str) -> ResolvedDevice:
    attributes = tuple(sorted(map(str, getattr(device, "attributes", []) or [])))
    chipset = tuple(
        value
        for value in attributes
        if value.startswith(("chipset:", "htp:", "hexagon:", "soc:"))
    )
    frameworks = tuple(
        value.split(":", 1)[1] if ":" in value else value
        for value in attributes
        if value.startswith(("framework:", "runtime:"))
    )
    return ResolvedDevice(
        name=str(getattr(device, "name", "")),
        os=str(getattr(device, "os", "")) or None,
        attributes=attributes,
        chipset_attributes=chipset,
        supported_frameworks=frameworks,
        selection_reason=reason,
        hub_device=device,
    )


def _exact(devices: Iterable[Any], name: str) -> list[Any]:
    return sorted((item for item in devices if getattr(item, "name", None) == name), key=_device_key)


def resolve_device(client: DeviceClient, config: DeviceConfig) -> ResolvedDevice:
    """Resolve one device according to the documented priority and fail closed."""

    devices = sorted(client.get_devices(), key=_device_key)
    if not devices:
        raise DeviceUnavailableError("Qualcomm AI Hub returned no hosted devices")

    requested = config.requested_name
    candidates = _exact(devices, requested)
    if candidates:
        return _build(candidates[0], f"exact requested device: {requested}")

    for fallback in (S22_5G, S22):
        if fallback == requested:
            continue
        candidates = _exact(devices, fallback)
        if candidates:
            return _build(candidates[0], f"preferred S22 fallback after '{requested}' was unavailable")

    family = sorted(
        (
            device
            for device in devices
            if "samsung" in str(getattr(device, "name", "")).lower()
            and "galaxy s22" in str(getattr(device, "name", "")).lower()
        ),
        key=_device_key,
    )
    if family and config.allow_s22_family_fallback:
        selected = family[0]
        LOGGER.warning("Using Samsung Galaxy S22 family fallback: %s", selected.name)
        return _build(selected, f"allowed S22-family fallback after '{requested}' was unavailable")

    if config.allow_unrelated_fallback:
        selected = devices[0]
        LOGGER.warning(
            "Using explicitly allowed unrelated device fallback: %s; results are not S22-comparable",
            selected.name,
        )
        return _build(selected, f"explicitly allowed unrelated fallback after '{requested}' was unavailable")

    available_samsung = [str(item.name) for item in devices if "samsung" in str(item.name).lower()]
    suffix = f" Available Samsung devices: {available_samsung}" if available_samsung else ""
    raise DeviceUnavailableError(
        f"No permitted Samsung Galaxy S22 device found for requested name '{requested}'.{suffix}"
    )


def discover_devices(client: DeviceClient, requested_name: str | None = None) -> list[dict[str, Any]]:
    devices = client.get_devices(name=requested_name) if requested_name else client.get_devices()
    return [_build(device, "discovery listing").to_dict() for device in sorted(devices, key=_device_key)]
