"""Typed errors used by benchmark workflows."""

from __future__ import annotations


class XRBenchError(RuntimeError):
    """Base error with a stable machine-readable category."""

    category = "unknown"

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause


class ConfigurationError(XRBenchError):
    category = "configuration"


class AuthenticationError(XRBenchError):
    category = "authentication"


class DeviceUnavailableError(XRBenchError):
    category = "device_unavailable"


class DownloadError(XRBenchError):
    category = "download_failure"


class ExportError(XRBenchError):
    category = "export_failure"


class UnsupportedOperatorError(XRBenchError):
    category = "unsupported_operator"


class DynamicShapeError(XRBenchError):
    category = "dynamic_shape_failure"


class ModelTooLargeError(XRBenchError):
    category = "model_too_large"


class JobTimeoutError(XRBenchError):
    category = "timeout"


class CompileError(XRBenchError):
    category = "compile_failure"


class ProfileError(XRBenchError):
    category = "profile_failure"


class InferenceMismatchError(XRBenchError):
    category = "inference_mismatch"


def classify_exception(error: BaseException) -> str:
    """Classify an arbitrary SDK/export failure without hiding the original error."""

    if isinstance(error, XRBenchError):
        return error.category
    text = str(error).lower()
    rules = (
        (("token", "unauthor", "credential", "forbidden"), "authentication"),
        (("no device", "device unavailable", "provision"), "device_unavailable"),
        (("unsupported op", "unsupported operator", "not supported"), "unsupported_operator"),
        (("dynamic shape", "symbolic shape", "dynamic dimension"), "dynamic_shape_failure"),
        (("out of memory", "too large", "resource exhausted"), "model_too_large"),
        (("timed out", "timeout"), "timeout"),
        (("download",), "download_failure"),
        (("compile", "translation"), "compile_failure"),
        (("profile",), "profile_failure"),
        (("mismatch", "numerical"), "inference_mismatch"),
        (("export", "onnx", "torch.export"), "export_failure"),
    )
    for needles, category in rules:
        if any(needle in text for needle in needles):
            return category
    return "unknown"
