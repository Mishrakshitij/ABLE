"""Checkpoint loading and reproducible response generation."""
from __future__ import annotations

import json
from pathlib import Path

from .batching import batches, encode_prompt
from .config import seed_everything


def load_checkpoint(checkpoint, device="auto"):
    from .models import load_causal_model

    adapter = Path(checkpoint) / "adapter_config.json"
    if adapter.exists():
        base = json.loads(adapter.read_text())["base_model_name_or_path"]
        return load_causal_model(base, adapter_path=str(checkpoint), device=device, trainable=False)
    return load_causal_model(str(checkpoint), device=device, trainable=False)


def generate_batch(model, tokenizer, prompts, max_new_tokens=50, max_length=512,
                   temperature=0.0, top_p=1.0, return_tokens=False):
    import torch

    if max_new_tokens < 1 or max_length <= max_new_tokens:
        raise ValueError("Require 0 < max_new_tokens < max_length")
    if temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("Require temperature >= 0 and 0 < top_p <= 1")
    budget = max_length - max_new_tokens
    sequences = [encode_prompt(tokenizer, prompt, budget) for prompt in prompts]
    width = max(map(len, sequences))
    device = next(model.parameters()).device
    pad = tokenizer.pad_token_id
    input_ids = torch.tensor([[pad] * (width - len(s)) + s for s in sequences], device=device)
    attention_mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in sequences], device=device)
    options = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0,
               "pad_token_id": pad, "eos_token_id": tokenizer.eos_token_id}
    if temperature > 0:
        options.update(temperature=temperature, top_p=top_p)
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, **options)
    generated = output[:, width:]
    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    if not return_tokens:
        return texts
    # First EOS is an action; subsequent fill tokens are padding, even when
    # EOS and PAD share an ID. A length-capped response has no invented EOS.
    eos = generated.eq(tokenizer.eos_token_id)
    generated_mask = (eos.long().cumsum(-1) - eos.long()).eq(0)
    full_attention = torch.cat([attention_mask, generated_mask.long()], dim=-1)
    response_mask = torch.zeros_like(output[:, 1:], dtype=torch.bool)
    response_mask[:, width - 1:] = generated_mask
    visible = generated_mask.clone()
    for special in tokenizer.all_special_ids:
        visible &= generated.ne(special)
    return texts, output, full_attention, response_mask, visible.sum(-1)


def generate_dataset(checkpoint, examples, output, *, device="auto", seed=10,
                     batch_size=8, max_new_tokens=50, max_length=512,
                     temperature=0.0, top_p=1.0):
    """Write keyed predictions and generated-response likelihoods as JSONL."""
    import torch
    from .losses import masked_mean, token_log_probabilities
    from .models import position_ids

    seed_everything(seed)
    model, tokenizer = load_checkpoint(checkpoint, device)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for items in batches(examples, batch_size):
            texts, ids, attention, mask, lengths = generate_batch(
                model, tokenizer, [e.prompt for e in items], max_new_tokens, max_length,
                temperature, top_p, return_tokens=True)
            with torch.no_grad():
                logits = model(input_ids=ids, attention_mask=attention,
                               position_ids=position_ids(attention)).logits
                log_probs = token_log_probabilities(logits[:, :-1], ids[:, 1:])
                nll = -masked_mean(log_probs, mask, dim=1)
            for e, text, score, tokens, length in zip(items, texts, nll.tolist(), mask.sum(-1).tolist(), lengths.tolist()):
                record = {"example_id": e.example_id, "conversation_id": e.conversation_id,
                          "turn_id": e.turn_id, "response": text,
                          "token_count": length,
                          "scored_tokens": tokens, "generated_nll": score}
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                count += 1
    if not count:
        raise ValueError("No generation examples")
    return {"examples": count, "output": str(output)}
