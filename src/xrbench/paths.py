"""Run-directory and cache path helpers."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from xrbench.config import BenchConfig

RUN_SUBDIRS = (
    "logs",
    "profiles",
    "models",
    "inference_outputs",
    "local_reference",
    "failure_reports",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def cache_dir(config: BenchConfig) -> Path:
    configured = config.section("paths").get("cache_dir")
    raw = os.environ.get("QAIHUB_BENCH_CACHE") or configured
    path = Path(raw).expanduser() if raw else Path.home() / ".cache" / "xrbench"
    return path.resolve()


def configure_model_cache_environment(config: BenchConfig) -> None:
    """Point Qualcomm model assets at the benchmark cache unless explicitly set."""

    os.environ.setdefault(
        "QAIHM_STORE_ROOT",
        str(cache_dir(config) / "qai_hub_models"),
    )


def create_run_dir(
    config: BenchConfig,
    benchmark: str,
    explicit: str | Path | None = None,
) -> Path:
    """Create or reuse the exact run directory and all required subdirectories."""

    configured = config.section("paths").get("output_dir")
    if explicit is not None or configured:
        run_dir = Path(explicit or str(configured)).expanduser().resolve()
    else:
        root = Path(str(config.section("paths").get("output_root", "outputs")))
        if not root.is_absolute():
            root = project_root() / root
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = (root / f"{stamp}-{benchmark}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in RUN_SUBDIRS:
        (run_dir / name).mkdir(exist_ok=True)
    return run_dir


def assert_writable(path: Path) -> tuple[bool, str]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".xrbench-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as error:
        return False, str(error)
    return True, "writable"
