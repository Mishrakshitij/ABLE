"""Token-masked supervised and proximal-policy optimization objectives.

All token tensors use [batch, time]. Masks select *response* tokens, including
their first EOS, and never select prompt or padding positions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


def masked_mean(values: Tensor, mask: Tensor, dim=None) -> Tensor:
    """Average selected entries; empty selections contribute a differentiable zero."""
    mask = mask.to(device=values.device, dtype=values.dtype)
    selected = torch.where(mask.bool(), values, torch.zeros_like(values))
    return selected.sum(dim=dim) / mask.sum(dim=dim).clamp_min(1)


def response_cross_entropy(logits: Tensor, labels: Tensor, ignore_index: int = -100) -> Tensor:
    """Causal cross entropy, shifted once, with prompt/padding labels ignored."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("Expected logits [batch, time, vocabulary] and labels [batch, time]")
    targets = labels[:, 1:]
    losses = F.cross_entropy(
        logits[:, :-1].float().transpose(1, 2), targets,
        ignore_index=ignore_index, reduction="none",
    )
    return masked_mean(losses, targets.ne(ignore_index))


def token_log_probabilities(logits: Tensor, token_ids: Tensor) -> Tensor:
    """Gather action log probabilities; logits and IDs are already aligned."""
    if logits.shape[:-1] != token_ids.shape:
        raise ValueError("Token IDs must match every logits dimension except vocabulary")
    return F.log_softmax(logits.float(), dim=-1).gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)


def token_entropy(logits: Tensor) -> Tensor:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return -(log_probs.exp() * log_probs).sum(-1)


def sampled_kl(log_probs: Tensor, reference_log_probs: Tensor) -> Tensor:
    """Signed Monte Carlo KL contribution for samples from the current policy."""
    return log_probs - reference_log_probs


def whiten(values: Tensor, mask: Tensor, epsilon: float = 1e-8) -> Tensor:
    mean = masked_mean(values, mask)
    variance = masked_mean((values - mean).square(), mask)
    return ((values - mean) * torch.rsqrt(variance + epsilon)) * mask


@dataclass
class PolicyLoss:
    loss: Tensor
    clip_fraction: Tensor
    approximate_kl: Tensor


def clipped_policy_loss(
    log_probs: Tensor, old_log_probs: Tensor, advantages: Tensor,
    mask: Tensor, clip_ratio: float = 0.2,
) -> PolicyLoss:
    """Negative clipped surrogate (paper Eq. 11); behavior policy is detached."""
    if not 0 < clip_ratio < 1:
        raise ValueError("clip_ratio must lie strictly between zero and one")
    log_ratio = log_probs - old_log_probs.detach()
    ratio = log_ratio.exp()
    advantage = advantages.detach()
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantage
    loss = -masked_mean(torch.minimum(unclipped, clipped), mask)
    return PolicyLoss(
        loss,
        masked_mean((ratio.sub(1).abs() > clip_ratio).float(), mask),
        masked_mean((ratio - 1) - log_ratio, mask),
    )


def clipped_value_loss(
    values: Tensor, old_values: Tensor, returns: Tensor, mask: Tensor,
    clip_range: float = 0.2,
) -> Tensor:
    """Clipped critic regression, including the conventional factor of one half."""
    if clip_range <= 0:
        raise ValueError("clip_range must be positive")
    old_values, returns = old_values.detach(), returns.detach()
    clipped = old_values + (values - old_values).clamp(-clip_range, clip_range)
    return 0.5 * masked_mean(torch.maximum((values - returns).square(), (clipped - returns).square()), mask)


def terminal_rewards(scores: Tensor, mask: Tensor, token_rewards: Tensor | None = None) -> Tensor:
    """Put each sequence score on its last selected token, preserving dense rewards."""
    if scores.shape != mask.shape[:1]:
        raise ValueError("One terminal score is required per sequence")
    if not mask.bool().any(-1).all():
        raise ValueError("Every trajectory must contain a response token")
    result = torch.zeros_like(mask, dtype=scores.dtype) if token_rewards is None else token_rewards.clone()
    positions = torch.arange(mask.shape[-1], device=mask.device).expand_as(mask)
    last = positions.masked_fill(~mask.bool(), -1).max(-1).values
    result[torch.arange(mask.shape[0], device=mask.device), last] += scores
    return result * mask


def discounted_returns(rewards: Tensor, mask: Tensor, gamma: float = 0.95) -> Tensor:
    """Monte Carlo returns for terminal episodes; padding cannot bootstrap returns."""
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    result = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[:, 0])
    for step in reversed(range(rewards.shape[1])):
        running = (rewards[:, step] + gamma * running) * mask[:, step]
        result[:, step] = running
    return result


def compute_advantages(
    rewards: Tensor, values: Tensor, mask: Tensor, gamma: float = 0.95,
    gae_lambda: float | None = None, normalize: bool = False,
) -> tuple[Tensor, Tensor]:
    """Return (advantages, critic targets).

    With ``gae_lambda=None``, use discounted returns minus the old critic,
    matching Eq. 10's return baseline. GAE is an explicitly optional extension.
    All trajectories terminate at EOS or the configured generation limit.
    """
    values = values.detach()
    if gae_lambda is None:
        returns = discounted_returns(rewards, mask, gamma)
        advantages = (returns - values) * mask
    else:
        if not 0 <= gae_lambda <= 1 or not 0 <= gamma <= 1:
            raise ValueError("gamma and gae_lambda must be in [0, 1]")
        advantages = torch.zeros_like(rewards)
        running = torch.zeros_like(rewards[:, 0])
        for step in reversed(range(rewards.shape[1])):
            next_value = values[:, step + 1] * mask[:, step + 1] if step + 1 < values.shape[1] else 0.0
            delta = rewards[:, step] + gamma * next_value - values[:, step]
            running = (delta + gamma * gae_lambda * running) * mask[:, step]
            advantages[:, step] = running
        returns = (advantages + values) * mask
    if normalize:
        advantages = whiten(advantages, mask)
    return advantages.detach(), returns.detach()
