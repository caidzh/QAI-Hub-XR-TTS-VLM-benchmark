"""Environment and credential diagnostics."""

from __future__ import annotations

import ctypes.util
import importlib.util
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from xrbench.config import BenchConfig
from xrbench.hub.client import remote_authorized
from xrbench.logging_utils import redact_sensitive_text
from xrbench.paths import assert_writable, cache_dir


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


@dataclass(frozen=True)
class DoctorReport:
    checks: tuple[Check, ...]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks if check.required)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "checks": [asdict(check) for check in self.checks]}


def _import_check(module: str, *, required: bool) -> Check:
    available = importlib.util.find_spec(module) is not None
    return Check(
        name=f"import:{module}",
        ok=available,
        detail="available" if available else "not installed",
        required=required,
    )


def run_doctor(
    config: BenchConfig,
    *,
    output_dir: Path,
    client: Any | None = None,
    cli_run_remote: bool = False,
) -> DoctorReport:
    checks: list[Check] = []
    version = sys.version_info
    supported = (3, 10) <= (version.major, version.minor) < (3, 12)
    checks.append(
        Check(
            "python",
            supported,
            f"{platform.python_version()} (supported: >=3.10,<3.12)",
        )
    )
    for module in ("yaml", "numpy", "qai_hub"):
        checks.append(_import_check(module, required=True))
    for module in ("qai_hub_models", "torch", "transformers", "soundfile"):
        checks.append(_import_check(module, required=False))

    cache_ok, cache_detail = assert_writable(cache_dir(config))
    checks.append(Check("cache_directory", cache_ok, cache_detail))
    output_ok, output_detail = assert_writable(output_dir)
    checks.append(Check("output_directory", output_ok, output_detail))

    portaudio = ctypes.util.find_library("portaudio")
    checks.append(
        Check(
            "libportaudio",
            portaudio is not None,
            portaudio or "not found; install libportaudio2 for PiperTTS",
            required=False,
        )
    )
    hf_configured = bool(
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    checks.append(
        Check(
            "hugging_face_authentication",
            True,
            "configured (value redacted)" if hf_configured else "not configured; optional for public model",
            required=False,
        )
    )
    checks.append(
        Check(
            "remote_submission",
            True,
            "enabled" if remote_authorized(cli_run_remote) else "disabled (safe default)",
            required=False,
        )
    )

    if importlib.util.find_spec("qai_hub") is None:
        checks.extend(
            (
                Check("qai_hub_authentication", False, "qai-hub is not installed"),
                Check("hosted_device_access", False, "not checked because qai-hub is unavailable"),
            )
        )
    else:
        try:
            if client is None:
                import qai_hub as hub

                client = hub.Client()
            devices = client.get_devices()
            checks.append(
                Check(
                    "qai_hub_authentication",
                    True,
                    "authenticated; token value was not inspected",
                )
            )
            checks.append(
                Check(
                    "hosted_device_access",
                    bool(devices),
                    f"{len(devices)} hosted device configurations visible",
                )
            )
        except Exception as error:
            checks.extend(
                (
                    Check(
                        "qai_hub_authentication",
                        False,
                        redact_sensitive_text(error)[:500],
                    ),
                    Check("hosted_device_access", False, "unavailable due to authentication/client error"),
                )
            )
    return DoctorReport(tuple(checks))


def format_doctor(report: DoctorReport) -> str:
    lines = ["XRBench doctor"]
    for check in report.checks:
        symbol = "PASS" if check.ok else ("WARN" if not check.required else "FAIL")
        lines.append(f"[{symbol}] {check.name}: {check.detail}")
    lines.append(f"Overall: {'ready' if report.ok else 'action required'}")
    return "\n".join(lines)
