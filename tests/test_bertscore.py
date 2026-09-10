"""Run the actual bert-score package against a local, randomly initialized model."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("bert_score")

from able.rewards import BertScoreSimilarity, RewardScorer
from test_pipeline import _tiny_models


def test_real_offline_bertscore_is_finite_and_frozen(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    torch.set_num_threads(1)
    torch.manual_seed(10)
    _, classifier = _tiny_models(tmp_path)
    similarity = BertScoreSimilarity(str(classifier), device="cpu", batch_size=1, num_layers=1)
    values = similarity(["Hello I need support", "Thank you"], ["Hello I need support", "Please help me"])
    assert values.shape == (2,)
    assert torch.isfinite(values).all()
    assert not values.requires_grad
    assert values.device.type == "cpu"
    assert values[0] == pytest.approx(1.0, abs=1e-5)
    assert not similarity.scorer._model.training
    assert all(not parameter.requires_grad and parameter.grad is None
               for parameter in similarity.scorer._model.parameters())
    assert similarity([], []).shape == (0,)


def test_reward_config_forwards_local_bertscore_layers(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    _, classifier = _tiny_models(tmp_path)
    config = {
        "weights": [0, 0, 0, 0, 0, 1], "bertscore_model": str(classifier),
        "bertscore_num_layers": 1, "device": "cpu", "batch_size": 1,
    }
    scorer = RewardScorer(None, config)
    assert scorer.bertscore.scorer.num_layers == 1
    from types import SimpleNamespace
    example = SimpleNamespace(response="Hello I need support", context="Please help me", prompt="Please help me")
    batch = scorer.score([example], ["Hello I need support"], torch.tensor([1.0]))
    assert torch.isfinite(batch.total).all() and not batch.total.requires_grad


def test_custom_bertscore_requires_explicit_layer_count(tmp_path):
    with pytest.raises(ValueError, match="bertscore_num_layers"):
        BertScoreSimilarity(str(tmp_path), device="cpu")
    for invalid in (-1, 1.5, True):
        with pytest.raises(ValueError, match="num_layers"):
            BertScoreSimilarity(str(tmp_path), device="cpu", num_layers=invalid)
