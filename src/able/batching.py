"""Response-only tokenization and bounded-memory iteration."""
from __future__ import annotations

import random
from itertools import islice


def batches(iterable, size: int):
    if size < 1:
        raise ValueError("batch size must be positive")
    iterator = iter(iterable)
    while batch := list(islice(iterator, size)):
        yield batch


def shuffled(iterable, seed: int, buffer_size: int = 2048):
    """Deterministic buffered shuffle without retaining the entire corpus."""
    if buffer_size < 1:
        raise ValueError("shuffle buffer must be positive")
    rng = random.Random(seed)
    buffer = []
    for item in iterable:
        if len(buffer) == buffer_size:
            index = rng.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = item
        else:
            buffer.append(item)
    rng.shuffle(buffer)
    yield from buffer


def encode_prompt(tokenizer, prompt: str, max_prompt_length: int) -> list[int]:
    """Preserve profile prefix and most recent dialogue within a token budget."""
    if max_prompt_length < 1:
        raise ValueError("max_prompt_length must be positive")
    context = tokenizer.encode(prompt, add_special_tokens=False)
    if not context:
        context = [tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id]
    if len(context) > max_prompt_length:
        prefix = min(96, max_prompt_length // 3)
        context = context[:prefix] + context[-(max_prompt_length - prefix):]
    return context


def encode_response(tokenizer, prompt: str, response: str, max_length: int) -> dict:
    """Reserve a context token, mask context, and always retain response EOS.

    Profile tokens at the beginning and recent dialogue at the end share the
    context budget when a transcript is longer than the configured window.
    """
    if max_length < 4:
        raise ValueError("max_length must be at least 4")
    if tokenizer.eos_token_id is None:
        raise ValueError("The causal tokenizer must define an EOS token")
    target = tokenizer.encode(response, add_special_tokens=False)
    target = target[:max_length - 2] + [tokenizer.eos_token_id]
    budget = max_length - len(target)
    context = encode_prompt(tokenizer, prompt, budget)
    ids = context + target
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(context) + target}


def collate_responses(examples, tokenizer, max_length: int, device=None):
    import torch

    encoded = [encode_response(tokenizer, e.prompt, e.response, max_length) for e in examples]
    if not encoded:
        raise ValueError("Cannot collate an empty batch")
    width = max(len(e["input_ids"]) for e in encoded)
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    result = {}
    for name, fill in (("input_ids", pad), ("attention_mask", 0), ("labels", -100)):
        result[name] = torch.tensor([e[name] + [fill] * (width - len(e[name]))
                                     for e in encoded], dtype=torch.long, device=device)
    return result
