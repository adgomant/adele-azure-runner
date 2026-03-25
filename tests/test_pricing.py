"""Tests for token usage aggregation and pricing estimation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adele_runner.config import AppConfig
from adele_runner.runtime.types import ChatMessage, ChatRequest
from adele_runner.schemas import InferenceOutput, JudgeOutput
from adele_runner.utils.pricing import estimate_batch_request_cost_usd, estimate_cost_usd


def _write_jsonl(path: Path, records: list) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.model_dump_json() + "\n")


def _inference(
    instance_id: str, model_id: str, prompt_tokens: int, completion_tokens: int
) -> InferenceOutput:
    return InferenceOutput(
        instance_id=instance_id,
        model_id=model_id,
        prompt="test",
        response="test",
        tokens_prompt=prompt_tokens,
        tokens_completion=completion_tokens,
    )


def _judge(instance_id: str, model_id: str, judge_name: str, score: int) -> JudgeOutput:
    return JudgeOutput(
        instance_id=instance_id,
        model_id=model_id,
        judge_name=judge_name,
        score=score,
        verdict="correct",
        reason="test",
        raw_output="{}",
        judge_prompt="",
    )


def _make_config(
    tmp_path: Path, *, pricing_enabled: bool = False, models: dict | None = None
) -> AppConfig:
    overrides = {
        "run": {"run_id": "test", "output_dir": str(tmp_path)},
        "judging": {"enabled": False},
    }
    if pricing_enabled:
        overrides["pricing"] = {
            "enabled": True,
            "models": models or {},
        }
    cfg = AppConfig.model_validate(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Token aggregation
# ---------------------------------------------------------------------------


def test_token_aggregation(tmp_path):
    from adele_runner.pipeline.metrics import summarize

    cfg = _make_config(tmp_path)
    run_dir = tmp_path / "test"
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(
        run_dir / "outputs.jsonl",
        [
            _inference("i1", "gpt-4o", 100, 50),
            _inference("i2", "gpt-4o", 200, 100),
        ],
    )
    _write_jsonl(
        run_dir / "judge_outputs.jsonl",
        [
            _judge("i1", "gpt-4o", "j1", 4),
            _judge("i2", "gpt-4o", "j1", 3),
        ],
    )

    summary = summarize(cfg)
    assert "token_usage" in summary
    assert "gpt-4o" in summary["token_usage"]
    assert summary["token_usage"]["gpt-4o"]["prompt_tokens"] == 300
    assert summary["token_usage"]["gpt-4o"]["completion_tokens"] == 150
    assert summary["token_usage"]["gpt-4o"]["total_tokens"] == 450


# ---------------------------------------------------------------------------
# Pricing estimation
# ---------------------------------------------------------------------------


def test_pricing_calculation(tmp_path):
    from adele_runner.pipeline.metrics import summarize

    cfg = _make_config(
        tmp_path,
        pricing_enabled=True,
        models={"gpt-4o": {"prompt_per_1k": 2.50, "completion_per_1k": 10.00}},
    )
    run_dir = tmp_path / "test"
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(
        run_dir / "outputs.jsonl",
        [
            _inference("i1", "gpt-4o", 1000, 500),
        ],
    )
    _write_jsonl(
        run_dir / "judge_outputs.jsonl",
        [
            _judge("i1", "gpt-4o", "j1", 5),
        ],
    )

    summary = summarize(cfg)
    assert "estimated_cost" in summary
    # 1000/1000 * 2.50 + 500/1000 * 10.00 = 2.50 + 5.00 = 7.50
    assert summary["estimated_cost"]["gpt-4o"] == pytest.approx(7.50, abs=0.01)


def test_pricing_disabled(tmp_path):
    from adele_runner.pipeline.metrics import summarize

    cfg = _make_config(tmp_path, pricing_enabled=False)
    run_dir = tmp_path / "test"
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(
        run_dir / "outputs.jsonl",
        [
            _inference("i1", "gpt-4o", 100, 50),
        ],
    )
    _write_jsonl(
        run_dir / "judge_outputs.jsonl",
        [
            _judge("i1", "gpt-4o", "j1", 4),
        ],
    )

    summary = summarize(cfg)
    assert "estimated_cost" not in summary


def test_no_pricing_for_unknown_model(tmp_path):
    from adele_runner.pipeline.metrics import summarize

    cfg = _make_config(
        tmp_path,
        pricing_enabled=True,
        models={"other-model": {"prompt_per_1k": 1.0, "completion_per_1k": 2.0}},
    )
    run_dir = tmp_path / "test"
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(
        run_dir / "outputs.jsonl",
        [
            _inference("i1", "gpt-4o", 100, 50),
        ],
    )
    _write_jsonl(
        run_dir / "judge_outputs.jsonl",
        [
            _judge("i1", "gpt-4o", "j1", 4),
        ],
    )

    summary = summarize(cfg)
    # pricing enabled but no config for gpt-4o → empty cost dict
    assert summary.get("estimated_cost", {}).get("gpt-4o") is None


# ---------------------------------------------------------------------------
# Metrics JSON export
# ---------------------------------------------------------------------------


def test_metrics_json_export(tmp_path):
    from adele_runner.pipeline.metrics import summarize

    cfg = _make_config(tmp_path)
    run_dir = tmp_path / "test"
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(
        run_dir / "outputs.jsonl",
        [
            _inference("i1", "gpt-4o", 100, 50),
        ],
    )
    _write_jsonl(
        run_dir / "judge_outputs.jsonl",
        [
            _judge("i1", "gpt-4o", "j1", 4),
        ],
    )

    summarize(cfg)
    metrics_path = cfg.metrics_path()
    assert metrics_path.exists()
    data = json.loads(metrics_path.read_text())
    assert "per_judge" in data
    assert "verification" in data


def test_runtime_cost_helper_matches_metrics_formula():
    pricing = AppConfig.model_validate(
        {"pricing": {"enabled": True, "models": {"m": {"prompt_per_1k": 2.5, "completion_per_1k": 10.0}}}}
    ).pricing.models["m"]
    cost = estimate_cost_usd(1_000, 500, pricing)
    assert cost == pytest.approx(7.5, abs=0.000001)


def test_batch_cost_estimate_is_conservative():
    pricing = AppConfig.model_validate(
        {"pricing": {"enabled": True, "models": {"m": {"prompt_per_1k": 1.0, "completion_per_1k": 2.0}}}}
    ).pricing.models["m"]
    request = ChatRequest(
        request_id="r1",
        model="m",
        messages=(ChatMessage(role="user", content="x" * 300),),
        max_tokens=200,
    )
    estimated = estimate_batch_request_cost_usd(request, pricing)
    actual = estimate_cost_usd(90, 150, pricing)
    assert estimated >= actual
