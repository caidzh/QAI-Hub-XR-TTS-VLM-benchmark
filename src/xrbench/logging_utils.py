"""Logging configuration that does not expose credential-bearing objects."""

from __future__ import annotations

import logging
import os
import re


class _CredentialFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_sensitive_text(record.getMessage())
        record.args = ()
        return True


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    for handler in logging.getLogger().handlers:
        handler.addFilter(_CredentialFilter())


def redact_secret(value: str, visible: int = 0) -> str:
    """Return a non-reversible display value for secret-like configuration."""

    if not value:
        return "<unset>"
    if visible <= 0:
        return f"<configured:{len(value)} chars>"
    return f"{value[:visible]}…<redacted>"


def redact_sensitive_text(value: object) -> str:
    """Remove configured tokens and common credential assignments from text."""

    text = str(value)
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "QAIHUB_API_TOKEN"):
        secret = os.environ.get(name)
        if secret:
            text = text.replace(secret, "<redacted>")
    text = re.sub(
        r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]+",
        "Bearer <redacted>",
        text,
    )
    text = re.sub(
        r"(?i)\b(api[_ -]?token|access[_ -]?token|authorization)"
        r"(\s*[:=]\s*)([^\s,;]+)",
        r"\1\2<redacted>",
        text,
    )
    return text
