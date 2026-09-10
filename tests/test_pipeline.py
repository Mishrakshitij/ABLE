"""Exercise the real training/generation path with offline, randomly initialized models."""
import csv
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from able.data import COLUMNS, iter_examples
from able.evaluation import evaluate_predictions
from able.inference import generate_dataset, load_checkpoint
from able.rewards import ClassifierScorer
from able.training import train_classifier, train_sft


def _tiny_models(root):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast,
                              RobertaConfig, RobertaForSequenceClassification)

    words = ["[PAD]", "[EOS]", "[UNK]", "Hello", "Thank", "you", "How", "can", "I",
             "help", "Please", "tell", "me", "more", "User", "Doctor", ":", "Gender",
             "Male", "Age", "Younger", "Persona", "High", "Low", "Openness", "(", ")",
             "O", "C", "E", "A", "N", "Conscientiousness", "Extraversion", "Agreeableness",
             "Neuroticism", ",", ".", "?", "need", "support"]
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                       eos_token="[EOS]", pad_token="[PAD]", bos_token="[EOS]",
                                       model_max_length=128)
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    causal = root / "causal"
    causal.mkdir()
    GPT2LMHeadModel(GPT2Config(vocab_size=len(words), n_positions=128, n_ctx=128,
                              n_embd=16, n_layer=1, n_head=2, bos_token_id=1,
                              eos_token_id=1, pad_token_id=0)).save_pretrained(causal)
    tokenizer.save_pretrained(causal)
    classifier = root / "classifier"
    classifier.mkdir()
    RobertaForSequenceClassification(RobertaConfig(
        vocab_size=len(words), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=130, num_labels=3,
        bos_token_id=1, eos_token_id=1, pad_token_id=0)).save_pretrained(classifier)
    tokenizer.save_pretrained(classifier)
    return causal, classifier


def _tiny_data(root):
    data = root / "datasets"
    data.mkdir()
    persona = "High Openness (O),High Conscientiousness (C),High Extraversion (E),High Agreeableness (A),Low Neuroticism (N)"
    for index, split in enumerate(("train", "validation", "test")):
        with (data / f"{split}-00000.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            for turn, (speaker, utterance) in enumerate([
                ("Patient", "Hello I need support"), ("Doctor", "Hello How can I help you ?"),
                ("Patient", "Thank you"), ("Doctor", "Please tell me more .")], 1):
                writer.writerow({"Convo_id": index + 1, "Turn_id": turn,
                                 "Speaker": speaker, "Utterance": utterance, "Gender": "Male",
                                 "Age": "Younger", "Persona": persona, "Issue": "support",
                                 "Physical disability": "Mobility Impairments", "row_id": index * 4 + turn - 1,
                                 "turn_index": turn, "split": split, "persona_label": 0,
                                 "gender_age_label": 0, "politeness_label": 2, "empathy_label": 1})
    return data


def test_training_checkpoint_generation_and_metrics(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(10)
    causal, classifier = _tiny_models(tmp_path)
    data = _tiny_data(tmp_path)
    config = {"data": {"path": str(data)}, "model": {"name": str(causal), "device": "cpu",
              "lora": {"r": 2, "lora_alpha": 2, "lora_dropout": 0.0, "target_modules": ["c_attn"]}},
              "training": {"output_dir": str(tmp_path / "sft"), "seed": 10, "batch_size": 1,
              "gradient_accumulation_steps": 2, "max_length": 96, "max_steps": 1, "epochs": 1}}
    metrics = train_sft(config)
    assert metrics["steps"] == 1 and metrics["examples_seen"] == 2
    assert metrics["validation_token_perplexity"] > 0
    reloaded, tokenizer = load_checkpoint(tmp_path / "sft", "cpu")
    assert not any(p.requires_grad for p in reloaded.parameters())
    checkpoints = {}
    for task in ("persona", "gender_age", "politeness", "empathy"):
        config["task"] = task
        config["model"] = {"name": str(classifier), "device": "cpu"}
        config["training"]["output_dir"] = str(tmp_path / task)
        metrics = train_classifier(config)
        assert metrics["steps"] == 1 and metrics["examples"] == 2
        checkpoints[task] = str(tmp_path / task)
    scorers = ClassifierScorer(checkpoints, device="cpu", max_length=96)
    probabilities = scorers.probabilities(["Hello", "Thank you"])
    assert probabilities["persona"].shape == (2, 19)
    assert all(torch.allclose(value.sum(-1), torch.ones(2), atol=1e-6) for value in probabilities.values())
    predictions = tmp_path / "predictions.jsonl"
    generate_dataset(tmp_path / "sft", iter_examples(data, "test"), predictions,
                     device="cpu", max_length=96, max_new_tokens=4, batch_size=2)
    result = evaluate_predictions(iter_examples(data, "test"), predictions, classifiers=scorers)
    assert result["examples"] == 2
    assert set(result["classifier_agreement"]) == set(checkpoints)
    assert result["token_perplexity"] > 0
    assert len(predictions.read_text().splitlines()) == 2
