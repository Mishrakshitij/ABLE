import unittest
from types import SimpleNamespace

import torch
from torch import nn

from able.losses import (
    clipped_policy_loss, clipped_value_loss, compute_advantages,
    discounted_returns, masked_mean, response_cross_entropy, terminal_rewards,
)
from able.models import ActorCritic, position_ids
from able.ppo import Trajectory, collect_rollout, collate_trajectories, optimize_rollout, response_mask_from_generation


class LossTests(unittest.TestCase):
    def test_response_ce_ignores_prompt_and_padding(self):
        logits = torch.zeros(1, 5, 3, requires_grad=True)
        labels = torch.tensor([[-100, -100, 1, 2, -100]])
        loss = response_cross_entropy(logits, labels)
        self.assertAlmostEqual(loss.item(), float(torch.log(torch.tensor(3.0))), places=6)
        loss.backward()
        self.assertEqual(logits.grad[0, 0].abs().sum().item(), 0)
        self.assertGreater(logits.grad[0, 1].abs().sum().item(), 0)
        self.assertEqual(logits.grad[0, 3:].abs().sum().item(), 0)

    def test_empty_mask_has_zero_finite_loss(self):
        values = torch.tensor([1.0, float("nan")])
        self.assertEqual(masked_mean(values, torch.tensor([1, 0])).item(), 1)
        logits = torch.zeros(1, 3, 4, requires_grad=True)
        loss = response_cross_entropy(logits, torch.full((1, 3), -100))
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_ppo_clip_for_positive_and_negative_advantages(self):
        ratios = torch.tensor([[1.5, 0.5, 100.0]])
        advantages = torch.tensor([[2.0, -2.0, 100.0]])
        result = clipped_policy_loss(ratios.log(), torch.zeros_like(ratios), advantages, torch.tensor([[1, 1, 0]]), .2)
        # min(3,2.4)=2.4; min(-1,-1.6)=-1.6 => negative mean = -0.4.
        self.assertAlmostEqual(result.loss.item(), -.4, places=6)
        self.assertEqual(result.clip_fraction.item(), 1)

    def test_old_policy_and_advantage_are_detached(self):
        current = torch.tensor([[.1]], requires_grad=True)
        old = torch.tensor([[0.0]], requires_grad=True)
        advantage = torch.tensor([[1.0]], requires_grad=True)
        clipped_policy_loss(current, old, advantage, torch.ones(1, 1)).loss.backward()
        self.assertLess(current.grad.item(), 0)
        self.assertIsNone(old.grad)
        self.assertIsNone(advantage.grad)

    def test_value_clipping_uses_worse_regression(self):
        loss = clipped_value_loss(torch.tensor([[1.0]]), torch.tensor([[0.0]]), torch.tensor([[1.0]]), torch.ones(1, 1), .2)
        self.assertAlmostEqual(loss.item(), .32, places=6)

    def test_terminal_returns_stop_at_padding(self):
        mask = torch.tensor([[0, 1, 1, 0], [1, 1, 1, 1]])
        rewards = terminal_rewards(torch.tensor([2.0, 4.0]), mask)
        self.assertTrue(torch.equal(rewards, torch.tensor([[0., 0., 2., 0.], [0., 0., 0., 4.]])))
        expected = torch.tensor([[0., 1., 2., 0.], [.5, 1., 2., 4.]])
        self.assertTrue(torch.equal(discounted_returns(rewards, mask, .5), expected))
        values = torch.ones_like(rewards)
        advantages, returns = compute_advantages(rewards, values, mask, gamma=.5)
        self.assertTrue(torch.equal(returns, expected))
        self.assertTrue(torch.equal(advantages, (expected - values) * mask))
        gae, gae_returns = compute_advantages(rewards, values, mask, gamma=.5, gae_lambda=1)
        self.assertTrue(torch.equal(gae, advantages))
        self.assertTrue(torch.equal(gae_returns, returns))

    def test_first_eos_is_included_even_when_eos_is_padding(self):
        generated = torch.tensor([[3, 0, 0, 0], [4, 5, 6, 7], [0, 0, 0, 0]])
        expected = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.bool)
        self.assertTrue(torch.equal(response_mask_from_generation(generated, 0), expected))

    def test_positions_are_padding_invariant(self):
        self.assertTrue(torch.equal(position_ids(torch.tensor([[0, 0, 1, 1, 1]])), torch.tensor([[0, 0, 0, 1, 2]])))

    def test_real_ppo_update_trains_actor_and_critic(self):
        class TinyPolicy(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_size=8)
                self.embedding = nn.Embedding(10, 8)
                self.output = nn.Linear(8, 10)

            def forward(self, input_ids, **kwargs):
                hidden = self.embedding(input_ids)
                return SimpleNamespace(logits=self.output(hidden), hidden_states=[hidden])

        torch.manual_seed(1)
        actor = ActorCritic(TinyPolicy())
        before = actor.policy.output.weight.detach().clone()
        records = [Trajectory(
            input_ids=torch.tensor([1, 2, 3]), response_mask=torch.tensor([0, 1]),
            old_log_probs=torch.zeros(2), old_values=torch.zeros(2),
            advantages=torch.tensor([0., 1.]), returns=torch.tensor([0., 1.]),
        )]
        batch = collate_trajectories(records, 0, "cpu")
        from able.losses import token_log_probabilities
        with torch.no_grad():
            logits, _ = actor(batch["input_ids"], batch["attention_mask"])
            records[0].old_log_probs = token_log_probabilities(logits[:, :-1], batch["input_ids"][:, 1:])[0]
        optimizer = torch.optim.AdamW(actor.parameters(), lr=.01)
        metrics = optimize_rollout(actor, optimizer, records, SimpleNamespace(pad_token_id=0), {
            "training": {"batch_size": 1}, "ppo": {"ppo_epochs": 1, "normalize_advantages": False},
        })
        self.assertFalse(torch.equal(before, actor.policy.output.weight))
        self.assertGreater(actor.value_head.weight.abs().sum().item(), 0)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))

    def test_tiny_gpt2_rollout_matches_recomputed_policy_and_updates(self):
        """Run real generation and PPO without downloading any pretrained model."""
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from able.rewards import RewardBatch
        from able.losses import token_log_probabilities

        tokenizer_backend = Tokenizer(WordLevel({"<eos>": 0, "<unk>": 1, "Patient": 2, "Doctor": 3, "hello": 4, "support": 5}, unk_token="<unk>"))
        tokenizer_backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_backend, eos_token="<eos>", pad_token="<eos>", unk_token="<unk>")
        tokenizer.padding_side = "left"
        config = GPT2Config(vocab_size=6, n_positions=32, n_embd=16, n_layer=1, n_head=2, bos_token_id=0, eos_token_id=0, pad_token_id=0)
        torch.manual_seed(3)
        actor = ActorCritic(GPT2LMHeadModel(config))
        reference = GPT2LMHeadModel(config)
        reference.load_state_dict(actor.policy.state_dict())
        reference.eval().requires_grad_(False)

        class FixedReward:
            def score(self, examples, texts, reference_nll):
                self_nll = reference_nll.detach().cpu()
                return RewardBatch(torch.ones(len(examples)), {"naturalness": -self_nll})

        examples = [SimpleNamespace(prompt="Patient hello Doctor", response="support"), SimpleNamespace(prompt="hello Doctor", response="support")]
        settings = {"training": {"batch_size": 2, "max_length": 24}, "ppo": {"max_new_tokens": 4, "ppo_epochs": 1, "normalize_advantages": False}}
        trajectories, metrics = collect_rollout(actor, reference, tokenizer, examples, FixedReward(), settings)
        self.assertEqual(len(trajectories), 2)
        self.assertAlmostEqual(metrics["rollout/reference_kl"], 0, places=5)
        batch = collate_trajectories(trajectories, tokenizer.pad_token_id, "cpu")
        with torch.no_grad():
            logits, _ = actor(batch["input_ids"], batch["attention_mask"])
            probabilities = token_log_probabilities(logits[:, :-1], batch["input_ids"][:, 1:])
        mask = batch["response_mask"].bool()
        torch.testing.assert_close(probabilities[mask], batch["old_log_probs"][mask], atol=1e-6, rtol=1e-5)
        before = actor.policy.transformer.wte.weight.detach().clone()
        optimize_rollout(actor, torch.optim.AdamW(actor.parameters(), lr=.01), trajectories, tokenizer, settings)
        self.assertFalse(torch.equal(before, actor.policy.transformer.wte.weight))
        self.assertGreater(actor.value_head.weight.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
