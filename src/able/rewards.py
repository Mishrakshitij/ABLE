"""Six ABLE rewards, backed by trained classifiers and contextual BERTScore.

``paper`` follows Eqs. 3–9 literally. ``aligned`` reverses the signs of
R1–R5, since the printed signs otherwise reward a lower correct-class
probability and a higher language-model loss under reward maximization.
These modes are separate scientific choices, never silently interchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from .models import resolve_device

TASKS = ("persona", "gender_age", "politeness", "empathy")
COMPONENTS = TASKS + ("naturalness", "coherence")


@dataclass
class RewardBatch:
    total: Tensor
    components: dict[str, Tensor]


def classifier_reward(reference_probability: Tensor, generated_probability: Tensor, alpha: float = 1.0, mode: str = "paper") -> Tensor:
    if mode not in {"paper", "aligned"}:
        raise ValueError("Reward mode must be 'paper' or 'aligned'")
    if not math.isfinite(alpha) or not 1 <= alpha <= 2:
        raise ValueError("The paper's alpha penalty must be in [1, 2]")
    result = reference_probability - alpha * generated_probability
    return result if mode == "paper" else -result


def naturalness_reward(nll: Tensor, mode: str = "paper") -> Tensor:
    if mode not in {"paper", "aligned"}:
        raise ValueError("Reward mode must be 'paper' or 'aligned'")
    result = torch.tanh(nll)
    return result if mode == "paper" else -result


def coherence_reward(reference_f1: Tensor, context_f1: Tensor, beta: float = 0.5, gamma: float = 0.5) -> Tensor:
    if not all(math.isfinite(value) for value in (beta, gamma)) or min(beta, gamma) < 0 or abs(beta + gamma - 1) > 1e-6:
        raise ValueError("Coherence beta and gamma must be nonnegative and sum to one")
    return beta * reference_f1 + gamma * context_f1


def combine_rewards(components: dict[str, Tensor], weights: Sequence[float]) -> Tensor:
    if len(weights) != 6 or any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("Exactly six nonnegative reward weights are required")
    # All-zero weights express the ABLE-R ablation from Section 5.2.
    if abs(sum(weights) - 1) > 1e-6 and sum(weights) != 0:
        raise ValueError("Reward weights must sum to one, or all be zero for ABLE-R")
    values = torch.stack([components[name] for name in COMPONENTS], dim=-1)
    return (values * values.new_tensor(weights)).sum(-1)


class ClassifierScorer:
    """Load the four trained response classifiers, freeze them, and batch inference."""

    def __init__(self, checkpoints: dict[str, str], device: str = "auto", max_length: int = 512, batch_size: int = 8):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if set(checkpoints) != set(TASKS):
            raise ValueError(f"Classifier checkpoints must have keys {TASKS}")
        self.device = resolve_device(device)
        self.max_length, self.batch_size = max_length, batch_size
        if max_length <= 0 or batch_size <= 0:
            raise ValueError("Classifier max_length and batch_size must be positive")
        self.models, self.tokenizers = {}, {}
        expected_sizes = {"persona": 19, "gender_age": 6, "politeness": 3, "empathy": 3}
        for task, checkpoint in checkpoints.items():
            metadata_file = Path(checkpoint) / "able_classifier.json"
            if metadata_file.is_file():
                metadata = json.loads(metadata_file.read_text())
                if metadata.get("task") != task or metadata.get("num_labels") != expected_sizes[task]:
                    raise ValueError(f"Checkpoint {checkpoint} has incompatible classifier task metadata for {task}")
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint, trust_remote_code=False)
            if model.config.num_labels != expected_sizes[task]:
                raise ValueError(f"{task} requires {expected_sizes[task]} classes; checkpoint has {model.config.num_labels}")
            model.to(self.device).eval().requires_grad_(False)
            self.models[task] = model
            self.tokenizers[task] = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=False)

    @torch.no_grad()
    def probabilities(self, texts: Sequence[str]) -> dict[str, Tensor]:
        if not texts:
            raise ValueError("Classifier scoring requires at least one text")
        outputs = {}
        for task in TASKS:
            chunks = []
            for start in range(0, len(texts), self.batch_size):
                tokens = self.tokenizers[task](
                    list(texts[start:start + self.batch_size]), padding=True,
                    truncation=True, max_length=self.max_length, return_tensors="pt",
                ).to(self.device)
                logits = self.models[task](**tokens).logits
                chunks.append(logits.float().softmax(-1).cpu())
            outputs[task] = torch.cat(chunks)
        return outputs

    def target_ids(self, task: str, labels: Sequence[int | str]) -> Tensor:
        mapping = self.models[task].config.label2id
        ids = []
        for label in labels:
            if isinstance(label, int):
                value = label
            elif label in mapping:
                value = mapping[label]
            else:
                raise ValueError(f"Unknown {task} target {label!r}; expected one of {list(mapping)}")
            if not 0 <= value < self.models[task].config.num_labels:
                raise ValueError(f"Out-of-range {task} class ID {value}")
            ids.append(value)
        return torch.tensor(ids, dtype=torch.long)


class BertScoreSimilarity:
    """Contextual F1; model/tokenizer downloads follow the bert-score package."""

    def __init__(
        self, model_type: str = "roberta-large", device: str = "auto",
        batch_size: int = 8, num_layers: int | None = None,
    ):
        from bert_score import BERTScorer
        from bert_score.utils import model2layers

        if num_layers is not None and (isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers < 0):
            raise ValueError("BERTScore num_layers must be a nonnegative integer or None")
        if num_layers is None and model_type not in model2layers:
            raise ValueError("Set bertscore_num_layers for a local or custom BERTScore model")
        if batch_size <= 0:
            raise ValueError("BERTScore batch_size must be positive")
        self.batch_size = batch_size
        self.scorer = BERTScorer(
            model_type=model_type, device=str(resolve_device(device)),
            batch_size=batch_size, lang="en", rescale_with_baseline=False,
            num_layers=num_layers,
        )
        self.scorer._model.eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, candidates: Sequence[str], references: Sequence[str]) -> Tensor:
        if len(candidates) != len(references):
            raise ValueError("Candidate and reference lengths differ")
        if not candidates:
            return torch.empty(0, dtype=torch.float32)
        _, _, f1 = self.scorer.score(list(candidates), list(references), batch_size=self.batch_size)
        return f1.detach().float().cpu()


class RewardScorer:
    """Evaluate the paper's six scalar rewards for a batch of generated responses.

    ``reference_nll`` means mean sampled-response token NLL under the frozen
    SFT model. This makes the otherwise unspecified Loss(y, y-hat) in Eq. 7
    operational and prevents the policy from changing its own fluency judge.
    Classifiers receive response text alone, never gold metadata in their input.
    """

    def __init__(self, classifiers: ClassifierScorer, config: dict | None = None, bertscore=None):
        config = config or {}
        self.classifiers = classifiers
        self.mode = config.get("mode", "paper")
        self.alpha = float(config.get("alpha", 1.0))
        self.beta = float(config.get("beta", 0.5))
        self.gamma = float(config.get("gamma", 0.5))
        self.weights = config.get("weights", [1 / 6] * 6)
        # Validate once, before allocating the BERTScore model.
        dummy = torch.tensor([0.0])
        classifier_reward(dummy, dummy, self.alpha, self.mode)
        coherence_reward(dummy, dummy, self.beta, self.gamma)
        combine_rewards({name: dummy for name in COMPONENTS}, self.weights)
        self.bertscore = bertscore
        if self.bertscore is None and self.weights[5] > 0:
            self.bertscore = BertScoreSimilarity(
                model_type=config.get("bertscore_model", "roberta-large"),
                device=config.get("device", "auto"), batch_size=config.get("batch_size", 8),
                num_layers=config.get("bertscore_num_layers"),
            )

    @torch.no_grad()
    def score(self, examples: Sequence, generated_texts: Sequence[str], reference_nll: Tensor) -> RewardBatch:
        if not examples or len(examples) != len(generated_texts) or reference_nll.shape != (len(examples),):
            raise ValueError("Examples, generations, and per-response NLL must have equal nonzero lengths")
        references = [example.response for example in examples]
        zeros = torch.zeros(len(examples), dtype=torch.float32)
        components = {name: zeros.clone() for name in COMPONENTS}
        if any(self.weights[:4]):
            reference_probs = self.classifiers.probabilities(references)
            generated_probs = self.classifiers.probabilities(generated_texts)
            for task in TASKS:
                targets = self.classifiers.target_ids(task, [getattr(example, f"{task}_label") for example in examples])
                ref = reference_probs[task].gather(1, targets[:, None]).squeeze(1)
                gen = generated_probs[task].gather(1, targets[:, None]).squeeze(1)
                components[task] = classifier_reward(ref, gen, self.alpha, self.mode).cpu()
        components["naturalness"] = naturalness_reward(reference_nll.detach().float().cpu(), self.mode)
        if self.weights[5] > 0:
            contexts = [getattr(example, "context", None) or example.prompt for example in examples]
            reference_f1 = self.bertscore(generated_texts, references)
            context_f1 = self.bertscore(generated_texts, contexts)
            components["coherence"] = coherence_reward(reference_f1, context_f1, self.beta, self.gamma)
        total = combine_rewards(components, self.weights)
        if not torch.isfinite(total).all():
            raise ValueError("Non-finite reward: check classifier, BERTScore, and reference model outputs")
        return RewardBatch(total.detach(), {name: value.detach() for name, value in components.items()})
