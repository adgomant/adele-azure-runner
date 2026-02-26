import asyncio

from adele_runner.config import AppConfig
from adele_runner.models import NormalizedExample
from adele_runner.pipeline import inference_runner


def test_dry_run_inference_with_mocked_adapter(monkeypatch, tmp_path) -> None:
    cfg = AppConfig.model_validate(
        {
            "run": {"run_id": "t1", "output_dir": str(tmp_path)},
            "dataset": {"name": "adele", "hf_id": "x", "split": "train", "limit": 1, "seed": 1},
            "inference": {
                "mode": "foundry_async",
                "foundry": {"endpoint": "https://example.test", "model": "m1", "api_key_env": "NOPE"},
            },
        }
    )

    monkeypatch.setattr(
        inference_runner,
        "load_adele_examples",
        lambda _cfg: [NormalizedExample(instance_id="1", prompt="hi", ground_truth="hello")],
    )

    class FakeAdapter:
        def complete(self, prompt: str):
            return {"text": f"echo:{prompt}", "raw": {"id": "r1"}}

    monkeypatch.setattr(inference_runner, "FoundryChatAdapter", lambda _cfg: FakeAdapter())

    output_path = asyncio.run(inference_runner.run_inference(cfg))
    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8").strip()
