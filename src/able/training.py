"""Supervised causal-language-model and reward-classifier training."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .batching import batches, collate_responses, shuffled
from .config import seed_everything, write_json
from .data import iter_examples

TASK_SIZES = {"persona": 19, "gender_age": 6, "politeness": 3, "empathy": 3}


def _examples(config, split, limit=None):
    return iter_examples(config["data"]["path"], split, max_examples=limit)


def _settings(config):
    training = config["training"]
    for name in ("batch_size", "epochs", "gradient_accumulation_steps", "max_length"):
        if int(training.get(name, {"batch_size": 8, "epochs": 1,
                                  "gradient_accumulation_steps": 1, "max_length": 512}[name])) < 1:
            raise ValueError(f"training.{name} must be positive")
    if training.get("max_steps") is not None and int(training["max_steps"]) < 1:
        raise ValueError("training.max_steps must be positive")
    seed_everything(int(training.get("seed", 10)))
    return training


def _save(model, tokenizer, optimizer, config, output, step, metrics):
    import torch

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    torch.save({"optimizer": optimizer.state_dict(), "step": step}, output / "optimizer.pt")
    write_json(output / "run_config.json", config)
    write_json(output / "metrics.json", metrics)


def train_sft(config: dict) -> dict:
    """Train Phi-2/LoRA using only Doctor response tokens as CE targets."""
    import torch
    from .models import load_causal_model

    settings = _settings(config)
    spec = config["model"]
    model, tokenizer = load_causal_model(
        spec["name"], device=spec.get("device", "auto"), lora=spec.get("lora"),
        adapter_path=spec.get("adapter_path"), dtype=spec.get("dtype"), trainable=True)
    device = next(model.parameters()).device
    model.config.use_cache = False
    if settings.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=float(settings.get("learning_rate", 2e-4)),
                                 weight_decay=float(settings.get("weight_decay", 0.01)))
    batch_size = int(settings.get("batch_size", 8))
    accumulation = int(settings.get("gradient_accumulation_steps", 1))
    max_length = int(settings.get("max_length", 512))
    max_steps = settings.get("max_steps")
    step = 0
    seen = 0
    total_loss = 0.0
    total_tokens = 0
    model.train()
    for epoch in range(int(settings.get("epochs", 8))):
        stream = shuffled(_examples(config, config["data"].get("train_split", "train"),
                                    config["data"].get("max_examples")),
                          int(settings.get("seed", 10)) + epoch)
        # Accumulate sums, then divide by the true number of response tokens.
        for group in batches(batches(stream, batch_size), accumulation):
            optimizer.zero_grad(set_to_none=True)
            prepared = [collate_responses(b, tokenizer, max_length, device) for b in group]
            counts = [int((b["labels"][:, 1:] != -100).sum()) for b in prepared]
            group_tokens = sum(counts)
            for batch, count in zip(prepared, counts):
                loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite SFT loss")
                (loss * count / group_tokens).backward()
                total_loss += float(loss.detach()) * count
                total_tokens += count
                seen += len(batch["input_ids"])
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings.get("max_grad_norm", 1.0)))
            optimizer.step()
            step += 1
            if step % max(1, int(settings.get("logging_steps", 10))) == 0:
                print(json.dumps({"step": step, "epoch": epoch + 1,
                                  "train_response_nll": total_loss / total_tokens}), flush=True)
            if max_steps is not None and step >= int(max_steps):
                break
        if max_steps is not None and step >= int(max_steps):
            break
    if not step:
        raise ValueError("Training split contains no Doctor examples")
    metrics = {"steps": step, "examples_seen": seen, "train_response_nll": total_loss / total_tokens}
    metrics.update(evaluate_sft(model, tokenizer, _examples(
        config, config["data"].get("validation_split", "validation"),
        config["data"].get("max_validation_examples")), batch_size, max_length))
    _save(model, tokenizer, optimizer, config, settings["output_dir"], step, metrics)
    return metrics


def evaluate_sft(model, tokenizer, examples, batch_size=8, max_length=512):
    import torch

    was_training = model.training
    model.eval()
    nll, count = 0.0, 0
    try:
        with torch.no_grad():
            for items in batches(examples, batch_size):
                batch = collate_responses(items, tokenizer, max_length, next(model.parameters()).device)
                tokens = int((batch["labels"][:, 1:] != -100).sum())
                nll += float(model(**batch).loss) * tokens
                count += tokens
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError("Validation split contains no scored response tokens")
    return {"validation_response_nll": nll / count,
            "validation_token_perplexity": math.exp(min(700, nll / count)),
            "validation_tokens": count}


def train_classifier(config: dict) -> dict:
    """Train one independent RoBERTa classifier on Doctor response text."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    settings = _settings(config)
    task = config.get("task")
    if task not in TASK_SIZES:
        raise ValueError(f"task must be one of {list(TASK_SIZES)}")
    spec = config["model"]
    device = spec.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(spec["name"])
    id2label = {i: str(i) for i in range(TASK_SIZES[task])}
    model = AutoModelForSequenceClassification.from_pretrained(
        spec["name"], num_labels=TASK_SIZES[task], id2label=id2label,
        label2id={v: k for k, v in id2label.items()}, ignore_mismatched_sizes=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings.get("learning_rate", 2e-5)))
    batch_size = int(settings.get("batch_size", 8))
    accumulation = int(settings.get("gradient_accumulation_steps", 1))
    max_length = int(settings.get("max_length", 512))
    max_steps = settings.get("max_steps")
    label_key = task + "_label"

    def encode(items):
        labels = [getattr(e, label_key) for e in items]
        if any(label is None for label in labels):
            raise ValueError(f"Missing {label_key}; classifier training requires complete labels")
        batch = tokenizer([e.response for e in items], padding=True, truncation=True,
                          max_length=max_length, return_tensors="pt").to(device)
        batch["labels"] = torch.tensor(labels, dtype=torch.long, device=device)
        return batch

    step, seen, loss_sum = 0, 0, 0.0
    model.train()
    for epoch in range(int(settings.get("epochs", 8))):
        stream = shuffled(_examples(config, config["data"].get("train_split", "train"),
                                    config["data"].get("max_examples")),
                          int(settings.get("seed", 10)) + epoch)
        for group in batches(batches(stream, batch_size), accumulation):
            optimizer.zero_grad(set_to_none=True)
            group_size = sum(len(items) for items in group)
            for items in group:
                loss = model(**encode(items)).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite classifier loss")
                (loss * len(items) / group_size).backward()
                loss_sum += float(loss.detach()) * len(items)
                seen += len(items)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings.get("max_grad_norm", 1.0)))
            optimizer.step()
            step += 1
            if step % max(1, int(settings.get("logging_steps", 10))) == 0:
                print(json.dumps({"task": task, "step": step, "epoch": epoch + 1,
                                  "train_loss": loss_sum / seen}), flush=True)
            if max_steps is not None and step >= int(max_steps):
                break
        if max_steps is not None and step >= int(max_steps):
            break
    if not step:
        raise ValueError("Training split contains no Doctor examples")
    matrix = [[0] * TASK_SIZES[task] for _ in range(TASK_SIZES[task])]
    model.eval()
    with torch.no_grad():
        for items in batches(_examples(config, config["data"].get("validation_split", "validation"),
                                       config["data"].get("max_validation_examples")), batch_size):
            batch = encode(items)
            predicted = model(**batch).logits.argmax(-1).tolist()
            for gold, pred in zip(batch["labels"].tolist(), predicted):
                matrix[gold][pred] += 1
    from .evaluation import classification_metrics
    metrics = {"task": task, "steps": step, "examples_seen": seen, "train_loss": loss_sum / seen,
               **classification_metrics(matrix)}
    _save(model, tokenizer, optimizer, config, settings["output_dir"], step, metrics)
    write_json(Path(settings["output_dir"]) / "able_classifier.json",
               {"task": task, "num_labels": TASK_SIZES[task], "label_order": list(id2label.values())})
    return metrics
