"""Defensive parser for Qualcomm AI Hub profile schema variants."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from xrbench.metrics import Distribution, describe

BYTES_PER_MIB = 1024.0 * 1024.0


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return None


def _first(containers: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> Any:
    for container in containers:
        for key in keys:
            if key in container and container[key] not in (None, "N/A", ""):
                return container[key]
    return None


def _time_ms(containers: Sequence[Mapping[str, Any]], *keys: str) -> float | None:
    value = _first(containers, keys)
    if isinstance(value, Mapping):
        if "microseconds" in value:
            return (_number(value["microseconds"]) or 0.0) / 1000.0
        if "milliseconds" in value:
            return _number(value["milliseconds"])
        value = value.get("value")
    number = _number(value)
    if number is None:
        return None
    # AI Hub public profile dictionaries use microseconds for unqualified time fields.
    return number / 1000.0


def _times_ms(containers: Sequence[Mapping[str, Any]], *keys: str) -> list[float]:
    value = _first(containers, keys)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    output: list[float] = []
    for item in value:
        if isinstance(item, Mapping):
            if "milliseconds" in item:
                number = _number(item["milliseconds"])
                if number is not None:
                    output.append(number)
                continue
            item = item.get("microseconds", item.get("value"))
        number = _number(item)
        if number is not None:
            output.append(number / 1000.0)
    return output


def _memory_range_mib(
    containers: Sequence[Mapping[str, Any]], *keys: str
) -> tuple[float | None, float | None]:
    value = _first(containers, keys)
    low: float | None = None
    high: float | None = None
    if isinstance(value, Mapping):
        low = _number(value.get("lower", value.get("low", value.get("min"))))
        high = _number(value.get("upper", value.get("high", value.get("max"))))
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes) and len(value) >= 2:
        low, high = _number(value[0]), _number(value[1])
    else:
        low = high = _number(value)
    return (
        low / BYTES_PER_MIB if low is not None else None,
        high / BYTES_PER_MIB if high is not None else None,
    )


@dataclass(frozen=True)
class ParsedProfile:
    estimated_inference_ms: float | None
    all_inference_ms: tuple[float, ...]
    inference_distribution: Distribution | None
    first_load_ms: float | None
    warm_load_ms: float | None
    compile_ms: float | None
    inference_peak_mib: tuple[float | None, float | None]
    inference_increase_mib: tuple[float | None, float | None]
    first_load_peak_mib: tuple[float | None, float | None]
    first_load_increase_mib: tuple[float | None, float | None]
    warm_load_peak_mib: tuple[float | None, float | None]
    warm_load_increase_mib: tuple[float | None, float | None]
    compile_peak_mib: tuple[float | None, float | None]
    compile_increase_mib: tuple[float | None, float | None]
    placement: dict[str, int]
    cpu_fallback: bool | None
    runtime_layers: tuple[dict[str, object], ...]


def parse_profile(profile: Mapping[str, Any]) -> ParsedProfile:
    """Parse v1 execution_summary and documented top-level/profile-summary forms."""

    execution = _mapping(profile.get("execution_summary"))
    summary = _mapping(profile.get("summary"))
    performance = _mapping(profile.get("performance"))
    containers = (execution, summary, performance, profile)

    all_inference = _times_ms(
        containers,
        "all_inference_times",
        "all_execution_times",
        "inference_times",
        "execution_times",
    )
    estimated = _time_ms(
        containers,
        "estimated_inference_time",
        "inference_time",
        "execution_time",
        "estimated_inference_time_us",
    )
    if estimated is None and all_inference:
        estimated = sum(all_inference) / len(all_inference)

    layers_raw = profile.get("execution_detail", profile.get("layer_details", []))
    layers: list[dict[str, object]] = []
    placement: Counter[str] = Counter()
    if isinstance(layers_raw, Sequence) and not isinstance(layers_raw, str | bytes):
        for layer in layers_raw:
            if not isinstance(layer, Mapping):
                continue
            normalized = {str(key): value for key, value in layer.items()}
            layers.append(normalized)
            unit = layer.get("compute_unit", layer.get("unit"))
            if unit:
                placement[str(unit).upper()] += 1

    cpu_fallback: bool | None
    if layers:
        cpu_fallback = placement.get("CPU", 0) > 0
    else:
        explicit = _first(containers, ("cpu_fallback", "has_cpu_fallback"))
        cpu_fallback = _boolean(explicit)

    return ParsedProfile(
        estimated_inference_ms=estimated,
        all_inference_ms=tuple(all_inference),
        inference_distribution=describe(all_inference),
        first_load_ms=_time_ms(containers, "first_load_time", "cold_load_time", "load_time"),
        warm_load_ms=_time_ms(containers, "warm_load_time"),
        compile_ms=_time_ms(containers, "compile_time"),
        inference_peak_mib=_memory_range_mib(
            containers,
            "inference_memory_peak_range",
            "execution_memory_peak_range",
            "estimated_inference_peak_memory",
        ),
        inference_increase_mib=_memory_range_mib(
            containers, "inference_memory_increase_range", "execution_memory_increase_range"
        ),
        first_load_peak_mib=_memory_range_mib(
            containers,
            "first_load_memory_peak_range",
            "cold_load_memory_peak_range",
            "first_load_peak_memory",
        ),
        first_load_increase_mib=_memory_range_mib(
            containers,
            "first_load_memory_increase_range",
            "cold_load_memory_increase_range",
        ),
        warm_load_peak_mib=_memory_range_mib(
            containers, "warm_load_memory_peak_range", "warm_load_peak_memory"
        ),
        warm_load_increase_mib=_memory_range_mib(
            containers, "warm_load_memory_increase_range"
        ),
        compile_peak_mib=_memory_range_mib(
            containers, "compile_memory_peak_range", "compile_peak_memory"
        ),
        compile_increase_mib=_memory_range_mib(
            containers, "compile_memory_increase_range"
        ),
        placement=dict(sorted(placement.items())),
        cpu_fallback=cpu_fallback,
        runtime_layers=tuple(layers),
    )
