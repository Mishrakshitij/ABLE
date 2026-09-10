"""PPO checkpoint/resume regression using local models and actual trained rewards."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from able import ppo
from able import rewards
from able.inference import load_checkpoint
from able.training import train_classifier, train_sft
from test_pipeline import _tiny_data, _tiny_models


def _assert_same_tree(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_same_tree(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert type(first) is type(second) and len(first) == len(second)
        for left, right in zip(first, second):
            _assert_same_tree(left, right)
    else:
        assert first == second


def _checkpoint_hashes(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(directory).rglob("*")) if path.is_file()}


@pytest.fixture(scope="module")
def trained_tiny_setup(tmp_path_factory):
    """Prepare genuine LoRA and classifier checkpoints once for this module."""
    torch.set_num_threads(1)
    torch.manual_seed(10)
    root = tmp_path_factory.mktemp("ppo-training-inputs")
    causal, classifier = _tiny_models(root)
    data = _tiny_data(root)
    base = {
        "data": {"path": str(data), "train_split": "train", "validation_split": "validation"},
        "model": {"name": str(causal), "device": "cpu", "dtype": "float32",
                  "lora": {"r": 2, "lora_alpha": 2, "lora_dropout": 0.0, "target_modules": ["c_attn"]}},
        "training": {"output_dir": str(root / "sft"), "seed": 10, "batch_size": 1,
                     "gradient_accumulation_steps": 2, "max_length": 96,
                     "learning_rate": 1e-3, "max_steps": 1, "epochs": 1},
    }
    assert train_sft(base)["steps"] == 1
    classifiers = {}
    for task in rewards.TASKS:
        settings = copy.deepcopy(base)
        settings["task"] = task
        settings["model"] = {"name": str(classifier), "device": "cpu"}
        settings["training"]["output_dir"] = str(root / task)
        assert train_classifier(settings)["steps"] == 1
        classifiers[task] = str(root / task)
    base["training"].update(gradient_accumulation_steps=1, max_steps=2, epochs=1,
                            max_grad_norm=1.0, save_steps=1)
    base["ppo"] = {
        "sft_checkpoint": str(root / "sft"), "clip_range": 0.2, "value_clip_range": 0.2,
        "value_coef": 0.5, "entropy_coef": 0.0, "kl_coef": 0.02, "gamma": 0.95,
        "gae_lambda": None, "ppo_epochs": 2, "rollout_batch_size": 1,
        "max_new_tokens": 4, "normalize_advantages": False,
    }
    base["rewards"] = {
        "mode": "paper", "alpha": 1.0, "beta": 0.5, "gamma": 0.5,
        "weights": [1 / 6] * 6, "classifiers": classifiers,
        "bertscore_model": "roberta-large", "device": "cpu", "max_length": 96,
    }
    return base


@pytest.fixture
def contextual_scores(monkeypatch):
    """Only the external contextual-similarity dependency is replaced."""
    calls = []

    class FixedContextualSimilarity:
        def __init__(self, **kwargs):
            pass

        def __call__(self, candidates, references):
            calls.append((list(candidates), list(references)))
            return torch.full((len(candidates),), 0.5)

    monkeypatch.setattr(rewards, "BertScoreSimilarity", FixedContextualSimilarity)
    return calls


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_lora_ppo_resume_is_identical_to_uninterrupted_updates(
    tmp_path, monkeypatch, trained_tiny_setup, contextual_scores, gradient_checkpointing,
):
    config = copy.deepcopy(trained_tiny_setup)
    config["training"]["gradient_checkpointing"] = gradient_checkpointing
    immutable_paths = [Path(config["ppo"]["sft_checkpoint"])] + [Path(value) for value in config["rewards"]["classifiers"].values()]
    original_files = {path: _checkpoint_hashes(path) for path in immutable_paths}
    frozen_references, frozen_classifiers, behavior_checks, actors = [], [], [], []

    original_loader = ppo._load_checkpoint_model

    def track_reference(path, model_config, trainable):
        model, tokenizer = original_loader(path, model_config, trainable)
        if not trainable:
            frozen_references.append((model, {name: value.detach().clone() for name, value in model.state_dict().items()}))
        return model, tokenizer

    class TrackedClassifiers(rewards.ClassifierScorer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            frozen_classifiers.append((self, {task: {name: value.detach().clone() for name, value in model.state_dict().items()}
                                              for task, model in self.models.items()}))

    original_optimizer = ppo.optimize_rollout

    def track_behavior(actor, optimizer, trajectories, tokenizer, settings):
        snapshots = [item.old_log_probs.clone() for item in trajectories]
        for item in trajectories:
            assert not item.old_log_probs.requires_grad
            assert not item.old_values.requires_grad
            assert not item.returns.requires_grad
            assert not item.advantages.requires_grad
        output = original_optimizer(actor, optimizer, trajectories, tokenizer, settings)
        for snapshot, item in zip(snapshots, trajectories):
            torch.testing.assert_close(snapshot, item.old_log_probs, rtol=0, atol=0)
        behavior_checks.append(True)
        actors.append(actor)
        return output

    monkeypatch.setattr(ppo, "_load_checkpoint_model", track_reference)
    monkeypatch.setattr(ppo, "ClassifierScorer", TrackedClassifiers)
    monkeypatch.setattr(ppo, "optimize_rollout", track_behavior)

    uninterrupted = tmp_path / "uninterrupted"
    config["training"]["output_dir"] = str(uninterrupted)
    assert ppo.train_ppo(config)["steps"] == 2

    interrupted = tmp_path / "interrupted"
    config["training"].update(output_dir=str(interrupted), max_steps=1)
    assert ppo.train_ppo(config)["steps"] == 1
    state = torch.load(interrupted / "training_state.pt", weights_only=True)
    assert state["step"] == 1 and state["epoch"] == 0 and state["next_index"] == 1
    assert all((interrupted / name).is_file() for name in (
        "adapter_model.safetensors", "adapter_config.json", "tokenizer_config.json",
        "value_head.pt", "training_state.pt", "training_config.json",
    ))

    resumed = tmp_path / "resumed"
    config["ppo"]["resume_from"] = str(interrupted)
    config["training"].update(output_dir=str(resumed), max_steps=2)
    assert ppo.train_ppo(config)["steps"] == 2
    final_model, _ = load_checkpoint(uninterrupted, "cpu")
    resumed_model, _ = load_checkpoint(resumed, "cpu")
    _assert_same_tree(final_model.state_dict(), resumed_model.state_dict())
    _assert_same_tree(torch.load(uninterrupted / "value_head.pt", weights_only=True),
                      torch.load(resumed / "value_head.pt", weights_only=True))
    _assert_same_tree(torch.load(uninterrupted / "training_state.pt", weights_only=True),
                      torch.load(resumed / "training_state.pt", weights_only=True))
    full_metrics = [json.loads(line) for line in (uninterrupted / "metrics.jsonl").read_text().splitlines()]
    resume_metrics = [json.loads(line) for line in (resumed / "metrics.jsonl").read_text().splitlines()]
    assert full_metrics[-1] == resume_metrics[-1]

    sft, _ = load_checkpoint(config["ppo"]["sft_checkpoint"], "cpu")
    assert any(not torch.equal(value, sft.state_dict()[name])
               for name, value in final_model.state_dict().items() if "lora_" in name)
    critic = torch.load(resumed / "value_head.pt", weights_only=True)
    assert critic["weight"].abs().sum() > 0
    for actor in actors:
        assert all("lora_" in name or name.startswith("value_head.")
                   for name, parameter in actor.named_parameters() if parameter.requires_grad)
    for model, snapshot in frozen_references:
        assert not any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters())
        _assert_same_tree(model.state_dict(), snapshot)
    for classifiers, snapshots in frozen_classifiers:
        for task, model in classifiers.models.items():
            assert not any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters())
            _assert_same_tree(model.state_dict(), snapshots[task])
    assert len(behavior_checks) == 4
    assert len(contextual_scores) == 8
    assert all(_checkpoint_hashes(path) == original_files[path] for path in immutable_paths)


@pytest.mark.parametrize("changed_section,changed_key,changed_value,error", [
    ("rewards", "mode", "aligned", "PPO and reward"),
    ("data", "max_examples", 1, "dataset and split"),
    ("training", "seed", 11, "training seed"),
    ("ppo", "rollout_batch_size", 2, "PPO and reward"),
])
def test_resume_rejects_changed_experiment_settings(
    tmp_path, trained_tiny_setup, contextual_scores,
    changed_section, changed_key, changed_value, error,
):
    config = copy.deepcopy(trained_tiny_setup)
    config["training"].update(output_dir=str(tmp_path / "initial"), max_steps=1)
    ppo.train_ppo(config)
    config["ppo"]["resume_from"] = str(tmp_path / "initial")
    config["training"].update(output_dir=str(tmp_path / "resumed"), max_steps=2)
    config[changed_section][changed_key] = changed_value
    with pytest.raises(ValueError, match=error):
        ppo.train_ppo(config)
