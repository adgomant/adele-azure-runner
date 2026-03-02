"""Tests for inter-rater reliability (Cohen's kappa)."""

from __future__ import annotations

import pytest

from adele_runner.pipeline.metrics import _cohen_kappa


def test_perfect_agreement():
    """All labels agree → kappa = 1.0."""
    labels_a = [1, 1, 0, 0, 1]
    labels_b = [1, 1, 0, 0, 1]
    kappa = _cohen_kappa(labels_a, labels_b)
    assert kappa == pytest.approx(1.0)


def test_no_agreement():
    """Complete disagreement."""
    labels_a = [1, 1, 1, 0, 0]
    labels_b = [0, 0, 0, 1, 1]
    kappa = _cohen_kappa(labels_a, labels_b)
    assert kappa < 0  # Negative kappa indicates worse than chance


def test_chance_agreement():
    """Half agree by chance — kappa near 0."""
    # Both rate everything positive → p_o = 1.0, p_e = 1.0 → kappa = 0.0
    labels_a = [1, 1, 1, 1]
    labels_b = [1, 1, 1, 1]
    kappa = _cohen_kappa(labels_a, labels_b)
    # When p_e == 1.0, function returns 0.0
    assert kappa == 0.0


def test_empty_labels():
    assert _cohen_kappa([], []) == 0.0


def test_known_value():
    """Manually computed kappa for a known case.

    labels_a: [1, 1, 0, 0, 1, 0]
    labels_b: [1, 0, 0, 0, 1, 1]

    p_o = 4/6 = 0.6667
    a_pos = 3/6 = 0.5, a_neg = 0.5
    b_pos = 3/6 = 0.5, b_neg = 0.5
    p_e = 0.5*0.5 + 0.5*0.5 = 0.5
    kappa = (0.6667 - 0.5) / (1 - 0.5) = 0.3333
    """
    labels_a = [1, 1, 0, 0, 1, 0]
    labels_b = [1, 0, 0, 0, 1, 1]
    kappa = _cohen_kappa(labels_a, labels_b)
    assert kappa == pytest.approx(1 / 3, abs=0.001)


def test_moderate_agreement():
    """A case with moderate agreement."""
    labels_a = [1, 1, 1, 0, 0, 1, 0, 1, 0, 0]
    labels_b = [1, 1, 0, 0, 0, 1, 1, 1, 0, 0]
    kappa = _cohen_kappa(labels_a, labels_b)
    # 8/10 agree → p_o = 0.8, p_e = 0.5*0.5 + 0.5*0.5 = 0.5
    # kappa = 0.3 / 0.5 = 0.6
    assert kappa == pytest.approx(0.6, abs=0.01)
