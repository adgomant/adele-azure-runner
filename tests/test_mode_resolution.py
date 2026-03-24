"""Tests for inference mode resolution logic."""

from __future__ import annotations

import pytest

from adele_runner.config import AppConfig


@pytest.mark.parametrize(
    "provider,mode,expected",
    [
        ("azure_ai_inference", "request_response", "request_response"),
        ("azure_openai", "batch", "batch"),
        ("google_genai", "request_response", "request_response"),
        ("azure_ai_inference", "auto", "request_response"),
    ],
)
def test_resolve_inference_mode(provider: str, mode: str, expected: str):
    cfg = AppConfig.model_validate(
        {
            "inference": {"provider": provider, "mode": mode},
        }
    )
    assert cfg.resolve_inference_mode() == expected
