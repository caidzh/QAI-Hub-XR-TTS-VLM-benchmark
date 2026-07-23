"""Serializable remote-job planning primitives."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PlannedJob:
    model: str
    stage: str
    variant: str
    target_device: str
    source_format: str
    target_runtime: str
    precision: str
    input_shapes: dict[str, object]
    compile_jobs: int = 1
    profile_jobs: int = 1
    inference_jobs: int = 1


def summarize_plan(jobs: Iterable[PlannedJob]) -> dict[str, Any]:
    job_list = list(jobs)
    return {
        "jobs": [asdict(job) for job in job_list],
        "expected_compile_jobs": sum(job.compile_jobs for job in job_list),
        "expected_profile_jobs": sum(job.profile_jobs for job in job_list),
        "expected_inference_jobs": sum(job.inference_jobs for job in job_list),
    }


def format_plan(jobs: Iterable[PlannedJob]) -> str:
    summary = summarize_plan(jobs)
    lines = ["Qualcomm AI Hub job plan"]
    for item in summary["jobs"]:
        lines.extend(
            [
                f"- {item['model']} / {item['stage']} / {item['variant']}",
                f"  device={item['target_device']} source={item['source_format']} "
                f"runtime={item['target_runtime']} precision={item['precision']}",
                f"  input_shapes={json.dumps(item['input_shapes'], sort_keys=True)}",
            ]
        )
    lines.append(
        "Totals: "
        f"compile={summary['expected_compile_jobs']} "
        f"profile={summary['expected_profile_jobs']} "
        f"inference={summary['expected_inference_jobs']}"
    )
    return "\n".join(lines)
