from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from adele_runner.config import AppConfig
from adele_runner.models import JudgeSummary, SummaryReport
from adele_runner.utils.io import ensure_run_dir, stream_jsonl


def summarize_results(config: AppConfig) -> SummaryReport:
    run_dir = ensure_run_dir(config.run.output_dir, config.run.run_id)
    judge_path = run_dir / "judge_outputs.jsonl"

    scores_by_judge: dict[str, list[int]] = defaultdict(list)
    scores_by_instance_judge: dict[tuple[str, str], int] = {}
    all_instances: set[str] = set()

    for row in stream_jsonl(judge_path):
        judge_name = str(row["judge_name"])
        instance_id = str(row["instance_id"])
        score = int(row["parsed"]["score"])
        scores_by_judge[judge_name].append(score)
        scores_by_instance_judge[(instance_id, judge_name)] = score
        all_instances.add(instance_id)

    judge_summaries = []
    threshold = config.judging.score_threshold
    for judge_name, scores in scores_by_judge.items():
        if not scores:
            continue
        avg = sum(scores) / len(scores)
        pass_rate = sum(1 for s in scores if s >= threshold) / len(scores)
        judge_summaries.append(JudgeSummary(judge_name=judge_name, average_score=avg, pass_rate=pass_rate))

    pairwise: dict[str, float] = {}
    judges = sorted(scores_by_judge.keys())
    for idx, left in enumerate(judges):
        for right in judges[idx + 1 :]:
            left_scores: list[int] = []
            right_scores: list[int] = []
            for instance in all_instances:
                l = scores_by_instance_judge.get((instance, left))
                r = scores_by_instance_judge.get((instance, right))
                if l is None or r is None:
                    continue
                left_scores.append(l)
                right_scores.append(r)
            if left_scores and right_scores:
                pairwise[f"{left}__{right}"] = float(cohen_kappa_score(left_scores, right_scores))

    return SummaryReport(total_examples=len(all_instances), judge_summaries=judge_summaries, pairwise_agreement=pairwise)
