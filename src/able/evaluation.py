"""Classification diagnostics and the automatic metrics in Eqs. 13–19."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from .batching import batches


def classification_metrics(confusion: list[list[int]]) -> dict:
    size = len(confusion)
    if not size or any(len(row) != size for row in confusion):
        raise ValueError("Confusion matrix must be nonempty and square")
    if any(v < 0 for row in confusion for v in row):
        raise ValueError("Confusion counts must be nonnegative")
    count = sum(map(sum, confusion))
    if not count:
        raise ValueError("Evaluation requires at least one example")
    f1 = []
    for i in range(size):
        tp = confusion[i][i]
        denominator = sum(confusion[i]) + sum(row[i] for row in confusion)
        f1.append(2 * tp / denominator if denominator else 0.0)
    return {"examples": count, "accuracy": sum(confusion[i][i] for i in range(size)) / count,
            "macro_f1": sum(f1) / size, "class_f1": f1, "confusion_matrix": confusion}


def evaluate_predictions(examples, predictions_path, classifiers=None, similarity=None, batch_size=8):
    """Report available metrics only; never substitute proxy classifier scores.

    Paper accuracies compare classifier argmax(reference) to argmax(generation).
    Gold-label accuracy is reported separately. Nrep averages BERTScore against
    the two previous generated Doctor turns within the same conversation.
    """
    predictions = {}
    with Path(predictions_path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            row = json.loads(line)
            identifier = row["example_id"]
            if identifier in predictions:
                raise ValueError(f"Duplicate prediction ID {identifier} at line {number}")
            if not isinstance(row.get("response"), str):
                raise ValueError("Every prediction must have a string response")
            for key in ("generated_nll", "scored_tokens", "token_count"):
                if key in row and (not isinstance(row[key], (int, float)) or not math.isfinite(row[key]) or row[key] < 0):
                    raise ValueError(f"Invalid prediction metric {key}")
            predictions[identifier] = row
    if not predictions:
        raise ValueError("Predictions file is empty")
    tasks = ("persona", "gender_age", "politeness", "empathy")
    agreement = dict.fromkeys(tasks, 0)
    correct = dict.fromkeys(tasks, 0)
    labelled = dict.fromkeys(tasks, 0)
    seen, counts, nlls, weighted_nlls, scored_tokens = set(), [], [], [], []
    histories = defaultdict(list)
    repetition = []
    for group in batches(examples, batch_size):
        selected = [e for e in group if e.example_id in predictions]
        if not selected:
            continue
        for e in selected:
            if e.example_id in seen:
                raise ValueError(f"Duplicate dataset example {e.example_id}")
            seen.add(e.example_id)
        rows = [predictions[e.example_id] for e in selected]
        texts = [row["response"] for row in rows]
        if classifiers is not None:
            reference = classifiers.probabilities([e.response for e in selected])
            generated = classifiers.probabilities(texts)
            for task in tasks:
                ref = reference[task].argmax(-1).tolist()
                pred = generated[task].argmax(-1).tolist()
                agreement[task] += sum(a == b for a, b in zip(ref, pred))
                for e, value in zip(selected, pred):
                    target = getattr(e, task + "_label")
                    if target is not None:
                        correct[task] += int(target == value)
                        labelled[task] += 1
        for e, row in zip(selected, rows):
            if "token_count" in row:
                counts.append(row["token_count"])
            if "generated_nll" in row and row.get("scored_tokens", 0) > 0:
                nlls.append(row["generated_nll"])
                scored_tokens.append(row["scored_tokens"])
                weighted_nlls.append(row["generated_nll"] * row["scored_tokens"])
            previous = histories[e.conversation_id]
            if previous and e.turn_id <= previous[-1][0]:
                raise ValueError("Examples must be in conversation turn order")
            if similarity is not None and len(previous) >= 2:
                similarities = similarity([row["response"]] * 2, [x[1] for x in previous[-2:]])
                repetition.append(float(sum(similarities) / 2))
            previous.append((e.turn_id, row["response"]))
    unknown = set(predictions) - seen
    if unknown:
        raise ValueError(f"Predictions contain {len(unknown)} IDs outside the requested dataset split")
    count = len(seen)
    result = {"examples": count}
    if len(counts) == count:
        result["response_length_tokens"] = sum(counts) / count
    if len(nlls) == count:
        result["mean_response_perplexity"] = sum(math.exp(min(700, n)) / count for n in nlls)
        result["token_perplexity"] = math.exp(min(700, sum(weighted_nlls) / sum(scored_tokens)))
    if classifiers is not None:
        result["classifier_agreement"] = {task: agreement[task] / count for task in tasks}
        result["gold_label_accuracy"] = {task: correct[task] / labelled[task] if labelled[task] else None for task in tasks}
    if similarity is not None:
        result["nrep_bertscore"] = sum(repetition) / len(repetition) if repetition else None
        result["nrep_eligible_turns"] = len(repetition)
    return result
