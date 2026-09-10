import unittest
from types import SimpleNamespace

import torch

from able.rewards import (
    COMPONENTS, TASKS, RewardScorer, classifier_reward, coherence_reward,
    combine_rewards, naturalness_reward,
)


class RewardTests(unittest.TestCase):
    def test_printed_classifier_equation_and_aligned_sign(self):
        reference, generated = torch.tensor([.8]), torch.tensor([.6])
        paper = classifier_reward(reference, generated, alpha=1.5)
        aligned = classifier_reward(reference, generated, alpha=1.5, mode="aligned")
        self.assertAlmostEqual(paper.item(), -.1, places=6)
        self.assertTrue(torch.equal(aligned, -paper))
        better = classifier_reward(reference, torch.tensor([.9]), mode="aligned")
        worse = classifier_reward(reference, torch.tensor([.2]), mode="aligned")
        self.assertGreater(better.item(), worse.item())

    def test_naturalness_modes_are_explicit(self):
        nll = torch.tensor([.1, 2.0])
        self.assertTrue(torch.equal(naturalness_reward(nll), torch.tanh(nll)))
        self.assertGreater(naturalness_reward(nll, "aligned")[0], naturalness_reward(nll, "aligned")[1])
        with self.assertRaises(ValueError):
            naturalness_reward(nll, "unknown")

    def test_weighted_coherence_and_reward_ablations(self):
        self.assertAlmostEqual(coherence_reward(torch.tensor([.8]), torch.tensor([.4]), .25, .75).item(), .5)
        components = {name: torch.tensor([float(i + 1)]) for i, name in enumerate(COMPONENTS)}
        self.assertAlmostEqual(combine_rewards(components, [1 / 6] * 6).item(), 3.5)
        self.assertEqual(combine_rewards(components, [0] * 6).item(), 0)
        self.assertEqual(combine_rewards(components, [0, 0, 0, 0, .5, .5]).item(), 5.5)
        with self.assertRaises(ValueError):
            combine_rewards(components, [1] * 6)
        with self.assertRaises(ValueError):
            coherence_reward(torch.zeros(1), torch.zeros(1), .5, .6)

    def test_reward_batch_uses_gold_class_and_response_only(self):
        calls = []

        class MockClassifiers:
            def probabilities(self, texts):
                calls.append(list(texts))
                values = [.1, .2, .7] if texts == ["reference"] else [.1, .6, .3]
                return {name: torch.tensor([values]) for name in TASKS}

            def target_ids(self, task, labels):
                return torch.tensor(labels)

        def similarity(candidates, references):
            self.assertEqual(candidates, ["candidate"])
            return torch.tensor([.8 if references == ["reference"] else .4])

        example = SimpleNamespace(response="reference", prompt="private metadata", context="dialogue context", **{f"{task}_label": 1 for task in TASKS})
        scorer = RewardScorer(MockClassifiers(), {"mode": "paper"}, bertscore=similarity)
        result = scorer.score([example], ["candidate"], torch.tensor([.5]))
        self.assertEqual(calls, [["reference"], ["candidate"]])
        self.assertAlmostEqual(result.components["persona"].item(), -.4, places=6)
        self.assertAlmostEqual(result.components["coherence"].item(), .6, places=6)
        self.assertEqual(set(result.components), set(COMPONENTS))
        self.assertFalse(result.total.requires_grad)

    def test_nonfinite_reward_hyperparameters_are_rejected(self):
        zero = torch.zeros(1)
        for invalid in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                classifier_reward(zero, zero, alpha=invalid)
            with self.assertRaises(ValueError):
                coherence_reward(zero, zero, beta=invalid)
            with self.assertRaises(ValueError):
                combine_rewards({name: zero for name in COMPONENTS}, [invalid, 0, 0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
