from __future__ import annotations

import pytest

from xrbench.logging_utils import redact_sensitive_text


def test_sensitive_text_redacts_environment_and_assignments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_example_secret")
    text = redact_sensitive_text(
        "HF=hf_example_secret api_token=qualcomm-secret Authorization: Bearer abc.def"
    )
    assert "hf_example_secret" not in text
    assert "qualcomm-secret" not in text
    assert "abc.def" not in text
    assert text.count("<redacted>") >= 3
