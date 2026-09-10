"""On-policy token PPO with a learned critic and a frozen SFT reference.

This implementation samples from the unmodified categorical policy (temperature
1, no top-k/top-p filtering), then recomputes that same distribution for PPO.
Dropout stays disabled during rollout and optimization, while gradients remain
enabled for the actor and value head. Checkpoints resume at rollout boundaries.
"""

from __future__ import annotations

import json
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .losses import (
    clipped_policy_loss, clipped_value_loss, compute_advantages, masked_mean,
    sampled_kl, terminal_rewards, token_entropy, token_log_probabilities,
)
from .models import ActorCritic, load_causal_model, position_ids
from .rewards import ClassifierScorer, RewardScorer


@dataclass
class Trajectory:
    input_ids: Tensor
    response_mask: Tensor
    old_log_probs: Tensor
    old_values: Tensor
    advantages: Tensor
    returns: Tensor


def response_mask_from_generation(generated_ids: Tensor, eos_token_id: int | list[int] | None) -> Tensor:
    """Include the first EOS in the response; ignore every token after it."""
    if eos_token_id is None:
        return torch.ones_like(generated_ids, dtype=torch.bool)
    eos_ids = [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id
    eos = torch.zeros_like(generated_ids, dtype=torch.bool)
    for token_id in eos_ids:
        eos |= generated_ids.eq(token_id)
    return (eos.long().cumsum(-1) - eos.long()).eq(0)


def _model_logits(model, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
    return model(
        input_ids=input_ids, attention_mask=attention_mask,
        position_ids=position_ids(attention_mask), use_cache=False, return_dict=True,
    ).logits


@torch.no_grad()
def collect_rollout(
    actor: ActorCritic, reference_model, tokenizer, examples: list,
    scorer: RewardScorer, config: dict,
) -> tuple[list[Trajectory], dict[str, float]]:
    """Sample a batch, score it, and return detached CPU trajectories."""
    from transformers import GenerationConfig

    training, ppo = config.get("training", {}), config.get("ppo", {})
    device = next(actor.parameters()).device
    max_new_tokens = int(ppo.get("max_new_tokens", 50))
    max_length = int(training.get("max_length", 512))
    if not 0 < max_new_tokens < max_length:
        raise ValueError("PPO max_new_tokens must be positive and smaller than training.max_length")
    max_prompt_length = int(ppo.get("max_prompt_length", max_length - max_new_tokens))
    if not 0 < max_prompt_length <= max_length - max_new_tokens:
        raise ValueError("PPO max_prompt_length plus max_new_tokens exceeds max_length")
    # Keep profile metadata and recent turns, rather than truncating the whole
    # prompt from the left and losing its persona fields.
    from .batching import encode_prompt
    prompt_ids = [encode_prompt(tokenizer, example.prompt, max_prompt_length) for example in examples]
    padded = tokenizer.pad({"input_ids": prompt_ids}, padding=True, return_tensors="pt").to(device)
    actor.eval()
    reference_model.eval()
    generation = GenerationConfig(
        do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
        max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id,
        use_cache=True,
    )
    sequences = actor.policy.generate(**padded, generation_config=generation)
    prompt_length = padded["input_ids"].shape[1]
    generated = sequences[:, prompt_length:]
    generated_mask = response_mask_from_generation(generated, tokenizer.eos_token_id)
    full_attention = torch.cat((padded["attention_mask"], generated_mask.long()), dim=1)
    action_mask = torch.zeros_like(sequences[:, 1:], dtype=torch.bool)
    action_mask[:, prompt_length - 1:] = generated_mask
    logits, values = actor(sequences, full_attention)
    old_log_probs = token_log_probabilities(logits[:, :-1], sequences[:, 1:])
    old_values = values[:, :-1].float()
    del logits, values
    reference_logits = _model_logits(reference_model, sequences, full_attention)
    reference_log_probs = token_log_probabilities(reference_logits[:, :-1], sequences[:, 1:])
    del reference_logits
    generated_nll = -masked_mean(reference_log_probs, action_mask, dim=1)
    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    rewards = scorer.score(examples, texts, generated_nll)
    dense_rewards = -float(ppo.get("kl_coef", 0.0)) * sampled_kl(old_log_probs, reference_log_probs)
    token_rewards = terminal_rewards(rewards.total.to(device), action_mask, dense_rewards)
    advantages, returns = compute_advantages(
        token_rewards, old_values, action_mask,
        gamma=float(ppo.get("gamma", 0.95)), gae_lambda=ppo.get("gae_lambda"),
        normalize=False,
    )
    trajectories = []
    for row in range(sequences.shape[0]):
        valid_positions = full_attention[row].nonzero().flatten()
        start, end = int(valid_positions[0]), int(valid_positions[-1]) + 1
        span = slice(start, end - 1)
        trajectories.append(Trajectory(
            sequences[row, start:end].detach().cpu(), action_mask[row, span].cpu(),
            old_log_probs[row, span].cpu(), old_values[row, span].cpu(),
            advantages[row, span].cpu(), returns[row, span].cpu(),
        ))
    metrics = {f"reward/{name}": float(value.mean()) for name, value in rewards.components.items()}
    metrics.update({
        "reward/total": float(rewards.total.mean()),
        "rollout/reference_nll": float(generated_nll.mean()),
        "rollout/response_tokens": float(generated_mask.float().sum(-1).mean()),
        "rollout/reference_kl": float(masked_mean(sampled_kl(old_log_probs, reference_log_probs), action_mask)),
    })
    return trajectories, metrics


def collate_trajectories(trajectories: list[Trajectory], pad_token_id: int, device) -> dict[str, Tensor]:
    """Right-pad stored trajectories and their already-shifted token statistics."""
    length = max(item.input_ids.numel() for item in trajectories)
    size = len(trajectories)
    result = {
        "input_ids": torch.full((size, length), pad_token_id, dtype=torch.long, device=device),
        "attention_mask": torch.zeros(size, length, dtype=torch.long, device=device),
    }
    for name in ("response_mask", "old_log_probs", "old_values", "advantages", "returns"):
        result[name] = torch.zeros(size, length - 1, device=device)
    for row, item in enumerate(trajectories):
        n = item.input_ids.numel()
        result["input_ids"][row, :n] = item.input_ids.to(device)
        result["attention_mask"][row, :n] = 1
        for name in ("response_mask", "old_log_probs", "old_values", "advantages", "returns"):
            result[name][row, :n - 1] = getattr(item, name).to(device)
    return result


@contextmanager
def _optimization_mode(actor: ActorCritic, checkpointing: bool):
    """Enable checkpoint gates while preserving the dropout-free rollout policy.

    These decoder families implement checkpointing on their model/decoder
    blocks and implement stochastic layers in child attention/MLP modules.
    Calling ``train()`` recursively would also enable functional attention
    dropout, even if every nn.Dropout child were subsequently put in eval mode.
    """
    actor.eval()
    try:
        if checkpointing:
            supported = {"phi", "gpt2", "llama", "mistral"}
            architecture = getattr(actor.policy.config, "model_type", None)
            if architecture not in supported:
                raise ValueError("Dropout-free PPO gradient checkpointing supports Phi, GPT-2, Llama, and Mistral")
            if not actor.policy.is_gradient_checkpointing:
                actor.policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                actor.policy.enable_input_require_grads()
            for module in actor.policy.modules():
                if getattr(module, "gradient_checkpointing", False):
                    # Set only this module's flag; never recurse into children.
                    module.training = True
        yield
    finally:
        actor.eval()


def optimize_rollout(actor: ActorCritic, optimizer, trajectories: list[Trajectory], tokenizer, config: dict) -> dict[str, float]:
    with _optimization_mode(actor, bool(config.get("training", {}).get("gradient_checkpointing", False))):
        return _optimize_rollout(actor, optimizer, trajectories, tokenizer, config)


def _optimize_rollout(actor: ActorCritic, optimizer, trajectories: list[Trajectory], tokenizer, config: dict) -> dict[str, float]:
    training, ppo = config.get("training", {}), config.get("ppo", {})
    batch_size = int(training.get("batch_size", 8))
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    epochs = int(ppo.get("ppo_epochs", 4))
    if min(batch_size, accumulation, epochs) <= 0:
        raise ValueError("PPO batch_size, gradient_accumulation_steps, and ppo_epochs must be positive")
    device = next(actor.parameters()).device
    if ppo.get("normalize_advantages", True):
        # Whiten across the complete rollout, not each variable-length minibatch.
        selected = torch.cat([item.advantages[item.response_mask.bool()] for item in trajectories])
        mean, variance = selected.mean(), selected.var(unbiased=False)
        for item in trajectories:
            item.advantages = (item.advantages - mean) * torch.rsqrt(variance + 1e-8) * item.response_mask
    history = []
    order = list(range(len(trajectories)))
    for _ in range(epochs):
        random.shuffle(order)
        batches = [order[start:start + batch_size] for start in range(0, len(order), batch_size)]
        for group_start in range(0, len(batches), accumulation):
            group = batches[group_start:group_start + accumulation]
            group_token_count = sum(int(trajectories[index].response_mask.sum()) for indices in group for index in indices)
            optimizer.zero_grad(set_to_none=True)
            for indices in group:
                batch = collate_trajectories([trajectories[index] for index in indices], tokenizer.pad_token_id, device)
                logits, values = actor(batch["input_ids"], batch["attention_mask"])
                log_probs = token_log_probabilities(logits[:, :-1], batch["input_ids"][:, 1:])
                mask = batch["response_mask"]
                policy = clipped_policy_loss(log_probs, batch["old_log_probs"], batch["advantages"], mask, float(ppo.get("clip_range", 0.2)))
                value_loss = clipped_value_loss(values[:, :-1], batch["old_values"], batch["returns"], mask, float(ppo.get("value_clip_range", 0.2)))
                entropy = masked_mean(token_entropy(logits[:, :-1]), mask)
                loss = policy.loss + float(ppo.get("value_coef", 0.5)) * value_loss - float(ppo.get("entropy_coef", 0.0)) * entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite PPO loss")
                (loss * (mask.sum() / group_token_count)).backward()
                history.append({
                    "train/loss": float(loss.detach()), "train/policy_loss": float(policy.loss.detach()),
                    "train/value_loss": float(value_loss.detach()), "train/entropy": float(entropy.detach()),
                    "train/clip_fraction": float(policy.clip_fraction.detach()),
                    "train/approximate_kl": float(policy.approximate_kl.detach()),
                })
            torch.nn.utils.clip_grad_norm_(actor.parameters(), float(training.get("max_grad_norm", 1.0)))
            optimizer.step()
    return {key: sum(row[key] for row in history) / len(history) for key in history[0]}


def _load_checkpoint_model(path: str, model_config: dict, trainable: bool):
    source = Path(path)
    if not source.is_dir():
        raise FileNotFoundError(f"Expected an existing SFT/PPO checkpoint directory: {source}")
    adapter = source / "adapter_config.json"
    base = model_config.get("name", "microsoft/phi-2")
    if adapter.exists():
        metadata = json.loads(adapter.read_text())
        base = metadata.get("base_model_name_or_path") or base
    return load_causal_model(
        base if adapter.exists() else str(source),
        device=model_config.get("device", "auto"), dtype=model_config.get("dtype"),
        adapter_path=str(source) if adapter.exists() else None, trainable=trainable,
    )


def save_checkpoint(actor, tokenizer, optimizer, directory: str | Path, config: dict, step: int, epoch: int, next_index: int) -> None:
    directory = Path(directory)
    actor.save_pretrained(directory, tokenizer)
    state = {
        "optimizer": optimizer.state_dict(), "step": step, "epoch": epoch,
        "next_index": next_index, "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    torch.save(state, directory / "training_state.pt")
    (directory / "training_config.json").write_text(json.dumps(config, indent=2) + "\n")


def train_ppo(config: dict) -> dict:
    """Run PPO after SFT and classifier training, returning the final checkpoint.

    ``training.max_steps`` counts rollout updates; ``ppo.ppo_epochs`` controls
    passes through each rollout. ``training.epochs`` bounds dataset passes.
    ``ppo.rollout_batch_size`` corresponds to the rollout/update interval.
    """
    from .data import iter_examples

    training, ppo, data_config = config["training"], config["ppo"], config["data"]
    model_config, reward_config = config["model"], config["rewards"]
    seed = int(training.get("seed", 10))
    random.seed(seed)
    torch.manual_seed(seed)
    output = Path(training["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    sft_checkpoint = ppo["sft_checkpoint"]
    if output.resolve() == Path(sft_checkpoint).resolve():
        raise ValueError("PPO output_dir must differ from the frozen SFT checkpoint")
    resume = ppo.get("resume_from")
    policy, tokenizer = _load_checkpoint_model(resume or sft_checkpoint, model_config, trainable=True)
    actor = ActorCritic(policy)
    reference, _ = _load_checkpoint_model(sft_checkpoint, model_config, trainable=False)
    if training.get("gradient_checkpointing", False):
        actor.policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        actor.policy.enable_input_require_grads()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in actor.parameters() if parameter.requires_grad],
        lr=float(training.get("learning_rate", 1e-5)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    examples = list(iter_examples(data_config["path"], data_config.get("train_split", "train"), max_examples=data_config.get("max_examples")))
    if not examples:
        raise ValueError("The PPO training split contains no response examples")
    if any(getattr(example, f"{task}_label") is None for example in examples for task in ("persona", "gender_age", "politeness", "empathy")):
        raise ValueError("PPO reward training requires complete response labels")
    batch_size = int(training.get("batch_size", 8))
    rollout_size = int(ppo.get("rollout_batch_size", 640))
    max_steps = int(training.get("max_steps", 32000))
    epochs = int(training.get("epochs", 20))
    if min(batch_size, rollout_size, max_steps, epochs) <= 0:
        raise ValueError("PPO batch sizes, max_steps, and epochs must be positive")
    classifiers = ClassifierScorer(
        reward_config["classifiers"], device=reward_config.get("device", "auto"),
        max_length=int(reward_config.get("max_length", 512)),
        batch_size=int(reward_config.get("batch_size", batch_size)),
    ) if any(reward_config.get("weights", [1 / 6] * 6)[:4]) else None
    scorer = RewardScorer(classifiers, reward_config)
    step, start_epoch, next_index = 0, 0, 0
    if resume:
        previous_config = json.loads((Path(resume) / "training_config.json").read_text())
        if previous_config["data"] != data_config:
            raise ValueError("Resume must preserve the dataset and split configuration")
        prior_ppo = {key: value for key, value in previous_config["ppo"].items() if key != "resume_from"}
        current_ppo = {key: value for key, value in ppo.items() if key != "resume_from"}
        if prior_ppo != current_ppo or previous_config["rewards"] != reward_config:
            raise ValueError("Resume must preserve PPO and reward settings, including rollout size and the frozen SFT checkpoint")
        mutable_training = {"output_dir", "epochs", "max_steps", "save_steps"}
        prior_training = {key: value for key, value in previous_config["training"].items() if key not in mutable_training}
        current_training = {key: value for key, value in training.items() if key not in mutable_training}
        if prior_training != current_training:
            raise ValueError("Resume must preserve training seed, optimizer, and minibatch settings")
        actor.load_value_head(resume)
        state = torch.load(Path(resume) / "training_state.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        step, start_epoch, next_index = state["step"], state["epoch"], state["next_index"]
        if not 0 <= next_index <= len(examples):
            raise ValueError("Resume position is outside the training dataset")
    last_metrics = {}
    final_epoch, final_index = start_epoch, next_index
    with (output / "metrics.jsonl").open("a", encoding="utf-8") as log:
        for epoch in range(start_epoch, epochs):
            order = list(range(len(examples)))
            random.Random(seed + epoch).shuffle(order)
            start = next_index if epoch == start_epoch else 0
            for offset in range(start, len(examples), rollout_size):
                if step >= max_steps:
                    break
                chosen = [examples[index] for index in order[offset:offset + rollout_size]]
                trajectories, metrics, sizes = [], [], []
                for chunk in range(0, len(chosen), batch_size):
                    current = chosen[chunk:chunk + batch_size]
                    records, scores = collect_rollout(actor, reference, tokenizer, current, scorer, config)
                    trajectories.extend(records)
                    metrics.append(scores)
                    sizes.append(len(current))
                last_metrics = {key: sum(row[key] * size for row, size in zip(metrics, sizes)) / sum(sizes) for key in metrics[0]}
                last_metrics.update(optimize_rollout(actor, optimizer, trajectories, tokenizer, config))
                step += 1
                final_epoch, final_index = epoch, offset + len(chosen)
                last_metrics.update({"step": step, "epoch": epoch + 1})
                log.write(json.dumps(last_metrics) + "\n")
                log.flush()
                print(json.dumps(last_metrics), flush=True)
                save_steps = int(training.get("save_steps", 100))
                if save_steps > 0 and step % save_steps == 0:
                    save_checkpoint(actor, tokenizer, optimizer, output / f"checkpoint-{step}", config, step, final_epoch, final_index)
            if step >= max_steps:
                break
        save_checkpoint(actor, tokenizer, optimizer, output, config, step, final_epoch, final_index)
    return {"output_dir": str(output), "steps": step, "metrics": last_metrics}
