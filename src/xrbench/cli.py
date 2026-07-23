"""Argparse CLI for safe local and Qualcomm AI Hub benchmark workflows."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from xrbench.config import load_config
from xrbench.devices import discover_devices
from xrbench.doctor import format_doctor, run_doctor
from xrbench.errors import AuthenticationError, ConfigurationError, XRBenchError
from xrbench.hub.client import remote_authorized
from xrbench.logging_utils import configure_logging, redact_sensitive_text
from xrbench.paths import configure_model_cache_environment, project_root
from xrbench.workflows import run_tts_workflow, run_vlm_workflow

TRACK_ACTIONS = ("prepare", "local", "profile", "infer", "report", "all")
VLM_ACTIONS = ("inspect", "prepare", "export", "local", "profile", "infer", "report", "all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xrbench",
        description="Qualcomm AI Hub PiperTTS and SmolVLM component latency benchmark",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="Check environment, credentials, devices, and libraries")
    _add_general_options(doctor)
    doctor.add_argument("--run-remote", action="store_true", help="Report remote submission as enabled")

    devices = commands.add_parser("devices", help="List hosted devices without selecting unrelated hardware")
    _add_general_options(devices)
    devices.add_argument(
        "--requested-name",
        help="Pass an exact device-name filter to Qualcomm AI Hub",
    )

    tts = commands.add_parser("tts", help="Official PiperTTS-EN benchmark track")
    tts_actions = tts.add_subparsers(dest="action", required=True)
    for action in TRACK_ACTIONS:
        child = tts_actions.add_parser(action, help=_action_help("tts", action))
        _add_benchmark_options(child, default_config="configs/tts_piper.yaml")

    vlm = commands.add_parser("vlm", help="Experimental fixed-shape SmolVLM benchmark track")
    vlm_actions = vlm.add_subparsers(dest="action", required=True)
    for action in VLM_ACTIONS:
        child = vlm_actions.add_parser(action, help=_action_help("vlm", action))
        _add_benchmark_options(child, default_config="configs/vlm_smolvlm_256m.yaml")

    all_parser = commands.add_parser("all", help="Run TTS then VLM, retaining partial results")
    _add_benchmark_options(all_parser, default_config="configs/default.yaml")
    return parser


def _add_general_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=str(project_root() / "configs" / "default.yaml"),
        help="YAML configuration path",
    )
    parser.add_argument("--output-dir", help="Exact output directory to create or reuse")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")


def _add_benchmark_options(parser: argparse.ArgumentParser, *, default_config: str) -> None:
    parser.add_argument(
        "--config",
        default=str(project_root() / default_config),
        help=f"YAML configuration path (default: {default_config})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print job plan and submit nothing")
    parser.add_argument(
        "--run-remote",
        action="store_true",
        help="Authorize remote jobs (QAIHUB_RUN_REMOTE=1 is an alternative)",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse compatible persisted job IDs")
    parser.add_argument(
        "--force-resubmit",
        action="store_true",
        help="Submit new jobs even when a compatible manifest record exists",
    )
    parser.add_argument("--skip-inference", action="store_true", help="Skip hosted validation jobs")
    parser.add_argument("--skip-download", action="store_true", help="Skip artifact downloads")
    parser.add_argument("--device", help="Override device.requested_name")
    parser.add_argument("--output-dir", help="Exact output directory to create or reuse")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")


def _action_help(track: str, action: str) -> str:
    meanings = {
        "inspect": "Inspect the installed model architecture and write diagnostics",
        "prepare": "Download/load the official model and prepare deterministic inputs",
        "export": "Export independent fixed-shape PT2/ONNX stage graphs",
        "local": "Run local floating-point reference/parity validation",
        "profile": "Compile/profile stages; do not submit hosted inference",
        "infer": "Compile/profile and run deterministic hosted inference",
        "report": "Regenerate reports from an existing --output-dir",
        "all": f"Run the complete {track.upper()} workflow",
    }
    return meanings[action]


def _run_all_in_track_environments(args: argparse.Namespace) -> int:
    """Run dependency-conflicting tracks in their reproducible environments."""

    root = project_root()
    interpreters = {
        "tts": root / ".venv-tts" / "bin" / "python",
        "vlm": root / ".venv-vlm" / "bin" / "python",
    }
    missing = [str(path) for path in interpreters.values() if not path.is_file()]
    if missing:
        raise ConfigurationError(
            "The combined real run needs separate TTS and VLM environments. "
            "Create them and run the benchmark with: scripts/run_all_benchmarks.sh. "
            f"Missing: {missing}"
        )

    output_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    exit_codes: list[int] = []
    for track, interpreter in interpreters.items():
        command = [
            str(interpreter),
            "-m",
            "xrbench",
            track,
            "all",
            "--config",
            str(Path(args.config).expanduser().resolve()),
        ]
        for flag in (
            "run_remote",
            "resume",
            "force_resubmit",
            "skip_inference",
            "skip_download",
            "verbose",
        ):
            if bool(getattr(args, flag)):
                command.append(f"--{flag.replace('_', '-')}")
        if args.device:
            command.extend(["--device", str(args.device)])
        if output_root is not None:
            command.extend(["--output-dir", str(output_root / track)])
        print(f"Starting {track.upper()} track in {interpreter.parent.parent.name}")
        result = subprocess.run(command, cwd=root, check=False)
        exit_codes.append(result.returncode)
    return max(exit_codes, default=0)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(bool(getattr(args, "verbose", False)))
    try:
        config = load_config(
            args.config,
            requested_device=getattr(args, "device", None),
            output_dir=getattr(args, "output_dir", None),
        )
        configure_model_cache_environment(config)
        if args.command == "doctor":
            output = (
                Path(args.output_dir).expanduser().resolve()
                if args.output_dir
                else project_root() / str(config.section("paths").get("output_root", "outputs"))
            )
            report = run_doctor(
                config,
                output_dir=output,
                cli_run_remote=bool(args.run_remote),
            )
            print(format_doctor(report))
            return 0 if report.ok else 1
        if args.command == "devices":
            try:
                import qai_hub as hub

                items = discover_devices(hub.Client(), args.requested_name)
            except Exception as error:
                raise AuthenticationError(
                    f"Qualcomm AI Hub device discovery failed: {error}", cause=error
                ) from error
            print(json.dumps(items, indent=2, sort_keys=True))
            return 0 if items else 1

        if args.command == "all" and not bool(args.dry_run):
            return _run_all_in_track_environments(args)

        enabled = remote_authorized(
            bool(getattr(args, "run_remote", False)),
            bool(getattr(args, "dry_run", False)),
        )
        common = {
            "output_dir": args.output_dir,
            "remote_enabled": enabled,
            "dry_run": bool(args.dry_run),
            "resume": bool(args.resume),
            "force_resubmit": bool(args.force_resubmit),
            "skip_inference": bool(args.skip_inference),
            "skip_download": bool(args.skip_download),
        }
        if args.command == "tts":
            code, run_dir = run_tts_workflow(args.action, config, **common)
        elif args.command == "vlm":
            code, run_dir = run_vlm_workflow(args.action, config, **common)
        else:
            root = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
            tts_dir = root / "tts" if root else None
            vlm_dir = root / "vlm" if root else None
            tts_code, tts_run = run_tts_workflow(
                "all", config, **{**common, "output_dir": tts_dir}
            )
            vlm_code, vlm_run = run_vlm_workflow(
                "all", config, **{**common, "output_dir": vlm_dir}
            )
            print(f"TTS results: {tts_run}")
            print(f"VLM results: {vlm_run}")
            return max(tts_code, vlm_code)
        print(f"Results: {run_dir}")
        return code
    except XRBenchError as error:
        print(
            f"ERROR [{error.category}]: {redact_sensitive_text(error)}",
            file=sys.stderr,
        )
        return 2
    except KeyboardInterrupt:
        print("Interrupted; completed manifests remain resumable.", file=sys.stderr)
        return 130
    except Exception as error:
        print(
            f"ERROR [unknown]: {type(error).__name__}: {redact_sensitive_text(error)}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
