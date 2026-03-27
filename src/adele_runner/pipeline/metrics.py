"""Metrics summary: per-judge stats, inter-rater agreement, token usage, and verification score."""

from __future__ import annotations

import json
import logging
from collections import Counter
from itertools import combinations
from typing import Any

from adele_runner.config import AppConfig
from adele_runner.schemas import InferenceOutput, JudgeOutput
from adele_runner.utils.io import latest_jsonl_by_key
from adele_runner.utils.pricing import estimate_cost_usd

logger = logging.getLogger(__name__)


def _cohen_kappa(labels_a: list[int], labels_b: list[int]) -> float:
    """Compute Cohen's kappa for two lists of binary labels (0 or 1).

    kappa = (p_o - p_e) / (1 - p_e)
    where p_o is observed agreement and p_e is expected agreement by chance.
    Returns 0.0 if p_e == 1.0 (perfect agreement by chance).
    """
    n = len(labels_a)
    if n == 0:
        return 0.0

    # Observed agreement
    p_o = sum(a == b for a, b in zip(labels_a, labels_b, strict=True)) / n

    # Expected agreement by chance
    a_pos = sum(labels_a) / n
    a_neg = 1.0 - a_pos
    b_pos = sum(labels_b) / n
    b_neg = 1.0 - b_pos
    p_e = a_pos * b_pos + a_neg * b_neg

    if p_e == 1.0:
        return 0.0

    return (p_o - p_e) / (1.0 - p_e)


def compute_verification_scores(
    inference_records: list[InferenceOutput],
    judge_records: list[JudgeOutput],
    expected_judges: list[str],
) -> dict[tuple[str, str], float | None]:
    """Return avg judge score per (instance_id, model_id), or None when incomplete."""
    judge_map: dict[tuple[str, str], dict[str, JudgeOutput]] = {}
    for jr in judge_records:
        judge_map.setdefault((jr.instance_id, jr.model_id), {})[jr.judge_name] = jr

    verification_scores: dict[tuple[str, str], float | None] = {}
    for inf in inference_records:
        key = (inf.instance_id, inf.model_id)
        if inf.status != "success" or inf.response is None:
            verification_scores[key] = None
            continue
        judges = judge_map.get(key, {})
        if not expected_judges:
            verification_scores[key] = None
            continue
        scores: list[int] = []
        for judge_name in expected_judges:
            jr = judges.get(judge_name)
            if jr is None or jr.score is None:
                verification_scores[key] = None
                break
            scores.append(jr.score)
        else:
            verification_scores[key] = sum(scores) / len(scores)
    return verification_scores


def summarize(config: AppConfig) -> dict[str, Any]:
    """Compute and print a metrics summary, then export to JSON."""
    judge_path = config.judge_outputs_path()

    inference_records = list(
        latest_jsonl_by_key(config.outputs_path(), InferenceOutput, "instance_id", "model_id").values()
    )
    judge_records = list(
        latest_jsonl_by_key(judge_path, JudgeOutput, "instance_id", "model_id", "judge_name").values()
    )
    if not inference_records and not judge_records:
        logger.warning("No inference or judge outputs found at %s", config.run_dir())
        return {}

    expected_judges = [judge.name for judge in config.judging.judges] if config.judging.enabled else []
    judge_names = sorted({jr.judge_name for jr in judge_records})
    logger.info("Judges found: %s", judge_names)

    # Per-judge stats (diagnostic)
    per_judge: dict[str, dict[str, Any]] = {}
    for name in judge_names:
        records = [jr for jr in judge_records if jr.judge_name == name]
        scores = [jr.score for jr in records if jr.score is not None]
        per_judge[name] = {
            "n": len(records),
            "n_scored": len(scores),
            "n_incomplete": len(records) - len(scores),
            "mean_score": round(sum(scores) / len(scores), 3) if scores else None,
            "score_dist": dict(sorted(Counter(scores).items())),
        }

    # Verification scores (avg across all judges per instance)
    avg_scores = compute_verification_scores(inference_records, judge_records, expected_judges)
    verifiable_scores = [avg for avg in avg_scores.values() if avg is not None]
    verification_binary = [1 if avg >= 3 else 0 for avg in verifiable_scores]
    verification: dict[str, Any] = {
        "n_instances": len(avg_scores),
        "n_verifiable": len(verifiable_scores),
        "n_incomplete": len(avg_scores) - len(verifiable_scores),
        "mean_avg_score": round(sum(verifiable_scores) / len(verifiable_scores), 3) if verifiable_scores else None,
        "verification_success_rate": round(sum(verification_binary) / len(verification_binary), 3)
        if verification_binary
        else None,
        "verification_score_dist": dict(sorted(Counter(verification_binary).items())),
    }

    # ----- Inter-rater agreement (Cohen's kappa) -----
    inter_rater: dict[str, float] = {}
    if len(judge_names) >= 2:
        # Build per-judge score index: judge_name -> {(instance_id, model_id): score}
        judge_score_map: dict[str, dict[tuple[str, str], int]] = {}
        for jr in judge_records:
            if jr.score is None:
                continue
            judge_score_map.setdefault(jr.judge_name, {})[(jr.instance_id, jr.model_id)] = jr.score

        for name_a, name_b in combinations(judge_names, 2):
            scores_a = judge_score_map.get(name_a, {})
            scores_b = judge_score_map.get(name_b, {})
            overlapping = sorted(set(scores_a.keys()) & set(scores_b.keys()))
            if not overlapping:
                continue
            labels_a = [1 if scores_a[k] >= 3 else 0 for k in overlapping]
            labels_b = [1 if scores_b[k] >= 3 else 0 for k in overlapping]
            kappa = _cohen_kappa(labels_a, labels_b)
            pair_key = f"{name_a} vs {name_b}"
            inter_rater[pair_key] = round(kappa, 4)

    # ----- Token usage -----
    token_usage: dict[str, dict[str, int]] = {}
    for rec in inference_records:
        entry = token_usage.setdefault(
            rec.model_id, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        )
        entry["prompt_tokens"] += rec.tokens_prompt or 0
        entry["completion_tokens"] += rec.tokens_completion or 0
        entry["total_tokens"] += (rec.tokens_prompt or 0) + (rec.tokens_completion or 0)

    # Judge token usage (keyed by judge_name so users can price judges independently)
    for jr in judge_records:
        entry = token_usage.setdefault(
            jr.judge_name, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        )
        entry["prompt_tokens"] += jr.tokens_prompt or 0
        entry["completion_tokens"] += jr.tokens_completion or 0
        entry["total_tokens"] += (jr.tokens_prompt or 0) + (jr.tokens_completion or 0)

    # ----- Estimated cost (if pricing enabled) -----
    estimated_cost: dict[str, float] = {}
    if config.pricing.enabled:
        for model_id, usage in token_usage.items():
            model_pricing = config.pricing.models.get(model_id)
            if model_pricing is None:
                continue
            cost = estimate_cost_usd(
                usage["prompt_tokens"],
                usage["completion_tokens"],
                model_pricing,
            )
            estimated_cost[model_id] = round(cost, 6)

    # ----- Build summary dict -----
    summary: dict[str, Any] = {
        "per_judge": per_judge,
        "verification": verification,
        "inter_rater": inter_rater,
        "token_usage": token_usage,
    }
    if config.pricing.enabled:
        summary["estimated_cost"] = estimated_cost

    # ----- Print nicely -----
    print("\n=== Metrics Summary ===")
    for name, stats in per_judge.items():
        print(f"\nJudge: {name}")
        print(f"  N={stats['n']}  scored={stats['n_scored']}  incomplete={stats['n_incomplete']}")
        print(f"  mean_score={stats['mean_score']}")
        print(f"  Score distribution: {stats['score_dist']}")

    print("\n--- Verification (avg score >= 3 → sucess) ---")
    print(f"  Instances evaluated: {verification['n_instances']}")
    print(f"  Verifiable: {verification['n_verifiable']}")
    print(f"  Incomplete: {verification['n_incomplete']}")
    print(f"  Mean avg score: {verification['mean_avg_score']}")
    print(f"  Verification sucess rate: {verification['verification_success_rate']}")
    print(f"  Distribution: {verification['verification_score_dist']}")

    if inter_rater:
        print("\n--- Inter-Rater Agreement (Cohen's Kappa) ---")
        for pair, kappa in inter_rater.items():
            print(f"  {pair}: {kappa}")

    if token_usage:
        print("\n--- Token Usage ---")
        for model_id, usage in token_usage.items():
            print(
                f"  {model_id}: prompt={usage['prompt_tokens']:,}  "
                f"completion={usage['completion_tokens']:,}  "
                f"total={usage['total_tokens']:,}"
            )

    if estimated_cost:
        print("\n--- Estimated Cost ---")
        for model_id, cost in estimated_cost.items():
            print(f"  {model_id}: ${cost:.6f}")
        total_cost = sum(estimated_cost.values())
        estimated_cost["total"] = round(total_cost, 6)
        print(f"  TOTAL: ${total_cost:.6f}")

    # ----- Export to JSON -----
    metrics_path = config.metrics_path()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info("Metrics written to %s", metrics_path)

    return summary
