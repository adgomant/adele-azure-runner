"""Tests for inference mode resolution logic."""

from __future__ import annotations

import pytest

from adele_runner.config import AppConfig


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("foundry", "foundry"),
        ("batch", "batch"),
        ("auto", "foundry"),
    ],
)
def test_resolve_inference_mode(mode: str, expected: str):
    cfg = AppConfig.model_validate(
        {
            "inference": {"mode": mode},
        }
    )
    assert cfg.resolve_inference_mode() == expected
