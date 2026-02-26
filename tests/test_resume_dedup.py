from pathlib import Path

from adele_runner.utils.io import append_jsonl, load_dedup_keys


def test_resume_dedup_inference_keys(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs.jsonl"
    append_jsonl(outputs, {"instance_id": "1", "model_id": "m1", "response_text": "a"})
    append_jsonl(outputs, {"instance_id": "2", "model_id": "m1", "response_text": "b"})

    keys = load_dedup_keys(outputs, ("instance_id", "model_id"))
    assert ("1", "m1") in keys
    assert ("2", "m1") in keys
