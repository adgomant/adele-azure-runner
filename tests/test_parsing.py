from adele_runner.pipeline.judge_runner import parse_judge_output


def test_parse_judge_output_strict_json() -> None:
    out = parse_judge_output('{"score": 5, "verdict": "correct", "reason": "matches"}')
    assert out.score == 5
    assert out.verdict == "correct"


def test_parse_judge_output_repair_embedded_json() -> None:
    raw = "Result:\n```json\n{\"score\":4,\"verdict\":\"partial\",\"reason\":\"close\"}\n```"
    out = parse_judge_output(raw)
    assert out.score == 4
    assert out.verdict == "partial"
