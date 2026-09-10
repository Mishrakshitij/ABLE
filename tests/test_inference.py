"""Generated likelihoods must score the exact prompt and sampled token path."""

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from able.inference import generate_batch, generate_dataset


class NonRoundTripTokenizer:
    """Decoded strings deliberately cannot be encoded back to their token IDs."""

    eos_token_id = 0
    bos_token_id = 0

    def __init__(self, pad_id):
        self.pad_token_id = pad_id
        self.all_special_ids = list({self.eos_token_id, pad_id})
        self.encoded_texts = []

    def encode(self, text, add_special_tokens=False):
        self.encoded_texts.append(text)
        if text.startswith("decoded:"):
            raise AssertionError("Likelihood scoring must not re-encode decoded generations")
        return [int(token) for token in text.split()]

    def batch_decode(self, sequences, skip_special_tokens=True):
        return ["decoded:" + "/".join(
            str(token) for token in sequence.tolist() if token not in self.all_special_ids
        ) for sequence in sequences]


class FixedGenerationModel(torch.nn.Module):
    """One short EOS response and one response that reaches the token limit."""

    def __init__(self, pad_id):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.pad_id = pad_id
        self.forward_inputs = None

    def generate(self, input_ids, attention_mask, **options):
        assert options["max_new_tokens"] == 4
        self.generation_prompt = input_ids.detach().clone()
        self.generation_attention = attention_mask.detach().clone()
        suffix = input_ids.new_tensor([[7, 0, self.pad_id, self.pad_id], [8, 9, 10, 11]])
        self.generated_ids = torch.cat([input_ids, suffix], dim=1)
        return self.generated_ids

    def forward(self, input_ids, attention_mask, position_ids):
        self.forward_inputs = {"ids": input_ids.clone(), "attention": attention_mask.clone(),
                               "positions": position_ids.clone()}
        vocabulary = torch.arange(32, device=input_ids.device)
        centers = (input_ids + position_ids) % 32
        logits = -(vocabulary[None, None, :] - centers[:, :, None]).abs().float() * 0.17
        self.scoring_logits = logits
        return SimpleNamespace(logits=logits)


PROMPTS = ["1 2 3", " ".join(str(token) for token in range(3, 25))]


@pytest.mark.parametrize("pad_id", [0, 2])
def test_generation_masks_first_eos_and_excludes_padding(pad_id):
    model, tokenizer = FixedGenerationModel(pad_id), NonRoundTripTokenizer(pad_id)
    _, ids, attention, response_mask, lengths = generate_batch(
        model, tokenizer, PROMPTS, max_new_tokens=4, max_length=12, return_tokens=True,
    )
    assert ids.shape == (2, 12)
    assert model.generation_prompt[0].tolist() == [pad_id] * 5 + [1, 2, 3]
    assert model.generation_prompt[1].tolist() == [3, 4, 19, 20, 21, 22, 23, 24]
    assert attention[0].tolist() == [0] * 5 + [1] * 5 + [0, 0]
    assert response_mask[0].nonzero().flatten().tolist() == [7, 8]
    assert response_mask[1].nonzero().flatten().tolist() == [7, 8, 9, 10]
    assert lengths.tolist() == [1, 4]


@pytest.mark.parametrize("pad_id", [0, 2])
def test_likelihood_uses_exact_generation_context_and_tokens(tmp_path, monkeypatch, pad_id):
    model, tokenizer = FixedGenerationModel(pad_id), NonRoundTripTokenizer(pad_id)
    monkeypatch.setattr("able.inference.load_checkpoint", lambda *_: (model, tokenizer))
    examples = [SimpleNamespace(example_id=f"{i}:2", conversation_id=str(i), turn_id=2,
                                prompt=prompt, response="unused reference")
                for i, prompt in enumerate(PROMPTS)]
    destination = tmp_path / "predictions.jsonl"
    generate_dataset("unused", examples, destination, device="cpu", max_new_tokens=4,
                     max_length=12, batch_size=2)
    rows = [json.loads(line) for line in destination.read_text().splitlines()]
    assert tokenizer.encoded_texts == PROMPTS
    assert torch.equal(model.forward_inputs["ids"], model.generated_ids)
    assert torch.equal(model.forward_inputs["ids"][:, :8], model.generation_prompt)
    assert torch.equal(model.forward_inputs["attention"][:, :8], model.generation_attention)
    assert model.forward_inputs["positions"][0, 5:10].tolist() == [0, 1, 2, 3, 4]
    assert [row["scored_tokens"] for row in rows] == [2, 4]
    assert [row["token_count"] for row in rows] == [1, 4]
    assert [row["response"] for row in rows] == ["decoded:7", "decoded:8/9/10/11"]
    # Independent log-softmax calculation on exactly the sampled positions.
    # The EOS counts once for the first response and is never added to the second.
    for row_index, targets in enumerate(([7, 0], [8, 9, 10, 11])):
        values = model.scoring_logits[row_index, 7:7 + len(targets)]
        expected = (torch.logsumexp(values, dim=-1)
                    - values[torch.arange(len(targets)), torch.tensor(targets)]).mean().item()
        assert rows[row_index]["generated_nll"] == pytest.approx(expected, abs=1e-6)
