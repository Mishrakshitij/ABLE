"""Checkpointed PPO must retain the exact dropout-free behavior distribution."""

import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from able.losses import token_log_probabilities
from able.models import ActorCritic
from able.ppo import _optimization_mode, collect_rollout, collate_trajectories, optimize_rollout
from able.rewards import RewardScorer
from test_pipeline import _tiny_models


@pytest.mark.parametrize("architecture", ["phi", "gpt2"])
def test_checkpointed_policy_matches_rollout_with_attention_and_lora_dropout(tmp_path, architecture):
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, PhiConfig, PhiForCausalLM
    from peft import LoraConfig, get_peft_model

    torch.set_num_threads(1)
    torch.manual_seed(10)
    causal, _ = _tiny_models(tmp_path)
    tokenizer = AutoTokenizer.from_pretrained(causal)
    tokenizer.padding_side = "left"
    if architecture == "phi":
        base = PhiForCausalLM(PhiConfig(
            vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, partial_rotary_factor=0.5,
            attention_dropout=0.4, resid_pdrop=0.3, embd_pdrop=0.2,
            bos_token_id=1, eos_token_id=1, pad_token_id=0,
        ))
        targets = ["q_proj", "k_proj", "v_proj", "dense"]
    else:
        base = GPT2LMHeadModel(GPT2Config(
            vocab_size=len(tokenizer), n_positions=128, n_embd=32,
            n_layer=1, n_head=4, attn_pdrop=0.4, resid_pdrop=0.3, embd_pdrop=0.2,
            bos_token_id=1, eos_token_id=1, pad_token_id=0,
        ))
        targets = ["c_attn"]
    policy = get_peft_model(base, LoraConfig(
        task_type="CAUSAL_LM", r=2, lora_alpha=2, lora_dropout=0.4, target_modules=targets,
    ))
    # Nonzero B makes LoRA dropout observable before the first optimization step.
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            if "lora_B" in name:
                parameter.normal_(mean=0, std=0.05)
    reference = copy.deepcopy(policy).eval().requires_grad_(False)
    actor = ActorCritic(policy)
    actor.policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    actor.policy.enable_input_require_grads()
    config = {
        "training": {"batch_size": 2, "max_length": 96, "gradient_checkpointing": True},
        "ppo": {"max_new_tokens": 4, "ppo_epochs": 1, "normalize_advantages": False},
    }
    examples = [SimpleNamespace(prompt="User : Hello I need support Doctor :", response="Please tell me more"),
                SimpleNamespace(prompt="User : Hello Doctor :", response="Hello How can I help you")]
    trajectories, _ = collect_rollout(actor, reference, tokenizer, examples,
                                      RewardScorer(None, {"weights": [0, 0, 0, 0, 1, 0]}), config)
    batch = collate_trajectories(trajectories, tokenizer.pad_token_id, "cpu")
    checkpoint_calls = []
    for name, module in actor.policy.named_modules():
        if hasattr(module, "_gradient_checkpointing_func"):
            original = module._gradient_checkpointing_func

            def tracked(function, *args, _original=original, _name=name, **kwargs):
                checkpoint_calls.append(_name)
                return _original(function, *args, **kwargs)

            module._gradient_checkpointing_func = tracked
    with _optimization_mode(actor, True):
        assert any(module.training and getattr(module, "gradient_checkpointing", False)
                   for module in actor.policy.modules())
        assert not any(module.training for module in actor.policy.modules()
                       if isinstance(module, torch.nn.Dropout))
        assert not any(module.training for module in actor.policy.modules()
                       if "Attention" in type(module).__name__)
        logits, _ = actor(batch["input_ids"], batch["attention_mask"])
        log_probs = token_log_probabilities(logits[:, :-1], batch["input_ids"][:, 1:])
        mask = batch["response_mask"].bool()
        torch.testing.assert_close(log_probs[mask], batch["old_log_probs"][mask], atol=1e-6, rtol=1e-5)
    assert checkpoint_calls
    checkpoint_calls.clear()
    before = {name: value.detach().clone() for name, value in policy.named_parameters() if "lora_" in name}
    metrics = optimize_rollout(actor, torch.optim.AdamW(actor.parameters(), lr=1e-3), trajectories, tokenizer, config)
    assert checkpoint_calls, "Configured checkpointing must execute during the actual PPO update"
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert any(not torch.equal(value, before[name]) for name, value in policy.named_parameters() if name in before)
    assert actor.value_head.weight.abs().sum() > 0
    assert not any(module.training for module in actor.modules())


def test_checkpointing_rejects_architectures_with_unreviewed_training_behavior():
    policy = torch.nn.Linear(2, 2)
    policy.config = SimpleNamespace(hidden_size=2, model_type="unsupported")
    actor = ActorCritic(policy)
    with pytest.raises(ValueError, match="gradient checkpointing supports"):
        with _optimization_mode(actor, True):
            pass
    assert not any(module.training for module in actor.modules())
