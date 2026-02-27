"""Tests for inference mode resolution logic."""

from __future__ import annotations

import pytest

from adele_runner.config import AppConfig


@pytest.mark.parametrize(
    "mode, batch_enabled, expected",
    [
        ("foundry_async", False, "foundry_async"),
        ("foundry_async", True, "foundry_async"),
        ("azure_openai_batch", False, "azure_openai_batch"),
        ("azure_openai_batch", True, "azure_openai_batch"),
        ("auto", False, "foundry_async"),
        ("auto", True, "azure_openai_batch"),
    ],
)
def test_resolve_inference_mode(mode: str, batch_enabled: bool, expected: str):
    cfg = AppConfig.model_validate({
        "inference": {
            "mode": mode,
            "batch": {"enabled": batch_enabled},
        },
    })
    assert cfg.resolve_inference_mode() == expected
