"""Numerical comparison helpers for hosted inference outputs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from xrbench.metrics import ValidationMetrics


def compare_arrays(
    reference: Any,
    actual: Any,
    *,
    logits: bool = False,
    top_k: int = 5,
    epsilon: float = 1e-8,
) -> ValidationMetrics:
    ref = np.asarray(reference, dtype=np.float64)
    got = np.asarray(actual, dtype=np.float64)
    if ref.shape != got.shape:
        raise ValueError(f"Output shape mismatch: reference={ref.shape}, actual={got.shape}")
    difference = np.abs(ref - got)
    relative = difference / np.maximum(np.abs(ref), epsilon)
    ref_flat = ref.reshape(-1)
    got_flat = got.reshape(-1)
    ref_norm = float(np.linalg.norm(ref_flat))
    got_norm = float(np.linalg.norm(got_flat))
    cosine = (
        float(np.dot(ref_flat, got_flat) / (ref_norm * got_norm))
        if ref_norm > 0 and got_norm > 0
        else None
    )
    top1: bool | None = None
    overlap: float | None = None
    if logits and ref.shape[-1] > 0:
        top1 = bool(np.argmax(ref, axis=-1).reshape(-1)[-1] == np.argmax(got, axis=-1).reshape(-1)[-1])
        width = min(top_k, ref.shape[-1])
        ref_top = set(np.argpartition(ref.reshape(-1, ref.shape[-1])[-1], -width)[-width:])
        got_top = set(np.argpartition(got.reshape(-1, got.shape[-1])[-1], -width)[-width:])
        overlap = len(ref_top & got_top) / width
    return ValidationMetrics(
        max_absolute_error=float(difference.max(initial=0.0)),
        mean_absolute_error=float(difference.mean()) if difference.size else 0.0,
        max_relative_error=float(relative.max(initial=0.0)),
        cosine_similarity=cosine if cosine is None or math.isfinite(cosine) else None,
        top1_agreement=top1,
        topk_overlap=overlap,
    )


def compare_output_mappings(
    reference: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    logits_names: set[str] | None = None,
    top_k: int = 5,
) -> dict[str, ValidationMetrics]:
    logits_names = logits_names or set()
    common = reference.keys() & actual.keys()
    return {
        name: compare_arrays(
            reference[name], actual[name], logits=name in logits_names, top_k=top_k
        )
        for name in sorted(common)
    }
