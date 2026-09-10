"""Hugging Face model loading and a token-state value head for ABLE."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn


def resolve_device(device: str = "auto") -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)


def load_causal_model(
    model_name: str, device: str = "auto", lora: dict | None = None,
    adapter_path: str | None = None, trainable: bool = True,
    dtype: str | None = None,
):
    """Load a full checkpoint or a base model with an optional LoRA adapter.

    Models load on a single explicit device. No remote Python model code is
    executed. Local checkpoint directories work without network access.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    target = resolve_device(device)
    if dtype in (None, "auto"):
        torch_dtype = torch.float32 if target.type == "cpu" else torch.bfloat16
    else:
        choices = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        if dtype not in choices:
            raise ValueError(f"Unsupported dtype {dtype!r}")
        torch_dtype = choices[dtype]
    tokenizer_source = adapter_path if adapter_path and (Path(adapter_path) / "tokenizer_config.json").exists() else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("A causal tokenizer must define an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, trust_remote_code=False)
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable)
    elif lora and lora.get("enabled", True):
        from peft import LoraConfig, get_peft_model
        options = {key: value for key, value in lora.items() if key != "enabled"}
        options.setdefault("task_type", "CAUSAL_LM")
        model = get_peft_model(model, LoraConfig(**options))
    model.to(target)
    model.config.pad_token_id = tokenizer.pad_token_id
    if not trainable:
        model.requires_grad_(False)
        model.eval()
    return model, tokenizer


def position_ids(attention_mask: Tensor) -> Tensor:
    """Give left-padded and unpadded versions of a prompt identical positions."""
    return (attention_mask.long().cumsum(-1) - 1).clamp_min(0)


class ActorCritic(nn.Module):
    """Causal response policy and a scalar value estimate before each action."""

    def __init__(self, causal_model: nn.Module):
        super().__init__()
        self.policy = causal_model
        config = causal_model.config
        hidden_size = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)
        if hidden_size is None:
            raise ValueError("Causal model config must expose hidden_size or n_embd")
        self.value_head = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)
        self.value_head.to(next(causal_model.parameters()).device)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor]:
        outputs = self.policy(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids(attention_mask), output_hidden_states=True,
            use_cache=False, return_dict=True,
        )
        values = self.value_head(outputs.hidden_states[-1].to(self.value_head.weight.dtype)).squeeze(-1)
        return outputs.logits, values

    def save_pretrained(self, directory: str | Path, tokenizer=None) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.policy.save_pretrained(directory)
        if tokenizer is not None:
            tokenizer.save_pretrained(directory)
        torch.save(self.value_head.state_dict(), directory / "value_head.pt")

    def load_value_head(self, directory: str | Path) -> None:
        state = torch.load(Path(directory) / "value_head.pt", map_location=self.value_head.weight.device, weights_only=True)
        self.value_head.load_state_dict(state)
