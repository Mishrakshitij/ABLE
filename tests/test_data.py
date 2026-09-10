import csv
import hashlib
import io
import json
import zipfile

import pytest

from able.data import COLUMNS, SOURCE_COLUMNS, iter_examples, label_mappings, persona_class, verify_dataset

PERSONA = "High Openness (O),High Conscientiousness (C),High Extraversion (E),High Agreeableness (A),Low Neuroticism (N)"


def write_split(directory, split, conversation_id, texts, *, base=0, source_turns=None):
    path = directory / f"{split}-00000.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for index, (speaker, text) in enumerate(texts, 1):
            writer.writerow(dict(zip(COLUMNS, [
                str(conversation_id), str(source_turns[index - 1] if source_turns else index),
                speaker, text, "Male", "Younger", PERSONA, "Mobility Aids", "Amputations",
                str(base + index - 1), str(index), split, "0", "0", "2", "2",
            ])))


def test_examples_keep_target_and_future_out_of_prompt(tmp_path):
    write_split(tmp_path, "train", 1, [
        ("Patient", "First question"), ("Doctor", "First answer"),
        ("Patient", "Second question"), ("Doctor", "Second answer"),
    ], source_turns=[1, 2, 2, 3])
    examples = list(iter_examples(tmp_path, "train"))
    assert [example.example_id for example in examples] == ["1:2", "1:4"]
    assert "First question" in examples[0].prompt
    assert "First answer" not in examples[0].prompt
    assert "Second question" not in examples[0].prompt
    assert "First answer" in examples[1].prompt
    assert "Second answer" not in examples[1].prompt
    assert "Gender:" not in examples[1].context
    assert "First answer" in examples[1].context
    assert len(list(iter_examples(tmp_path, "train", max_examples=1))) == 1
    assert list(iter_examples(tmp_path, "train", max_examples=0)) == []


def test_verifier_detects_duplicate_content_leakage(tmp_path):
    write_split(tmp_path, "train", 1, [("Patient", "Help me"), ("Doctor", "I can help")])
    write_split(tmp_path, "validation", 2, [("Patient", " Help  me "), ("Doctor", "I CAN HELP")], base=2)
    write_split(tmp_path, "test", 3, [("Patient", "Other question"), ("Doctor", "Other answer")], base=4)
    report = verify_dataset(tmp_path, require_manifest=False)
    assert not report["valid"]
    assert any("content leaks" in error for error in report["errors"])


def test_verifier_rejects_profile_class_mismatch(tmp_path):
    for index, split in enumerate(("train", "validation", "test")):
        write_split(tmp_path, split, index, [("Patient", split), ("Doctor", "Reply")], base=index * 2)
    path = tmp_path / "train-00000.csv"
    path.write_text(path.read_text().replace(",train,0,0,2,2", ",train,1,0,2,2"))
    report = verify_dataset(tmp_path, require_manifest=False)
    assert not report["valid"]
    assert any("Persona class mismatch" in error for error in report["errors"])


def test_persona_order_is_independent_of_string_order():
    assert persona_class(PERSONA) == 0
    assert persona_class(
        "High Neuroticism (N), Low Conscientiousness (C), Low Extraversion (E), High Agreeableness (A), Low Openness(O)"
    ) == 18
    with pytest.raises(ValueError, match="five OCEAN"):
        persona_class("High Openness (O)")


def test_source_verification_detects_changed_utterance(tmp_path):
    source_rows = []
    for index, split in enumerate(("train", "validation", "test")):
        write_split(tmp_path, split, index, [("Patient", split), ("Doctor", "Reply")], base=index * 2)
        with (tmp_path / f"{split}-00000.csv").open() as handle:
            source_rows.extend(csv.DictReader(handle))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=SOURCE_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(source_rows)
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("PERPDSCD.csv", buffer.getvalue())
    assert verify_dataset(tmp_path, source_archive=archive, require_manifest=False)["valid"]
    path = tmp_path / "train-00000.csv"
    path.write_text(path.read_text().replace("Reply", "Changed reply"))
    report = verify_dataset(tmp_path, source_archive=archive, require_manifest=False)
    assert not report["valid"]
    assert any("source fields changed" in error for error in report["errors"])


def write_minimal_release(directory, *, wrong_split_mapping=False):
    splits = {}
    for index, split in enumerate(("train", "validation", "test")):
        write_split(directory, split, index, [("Patient", split), ("Doctor", "Reply")], base=index * 2)
        splits[split] = [str(index)]
    if wrong_split_mapping:
        splits["validation"] = splits["train"]
    (directory / "splits.json").write_text(json.dumps(splits))
    (directory / "labels.json").write_text(json.dumps(label_mappings()))
    files = [{"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
             for path in sorted(directory.iterdir()) if path.name != "manifest.json"]
    (directory / "manifest.json").write_text(json.dumps({"files": files}))


def test_release_verifier_requires_manifest(tmp_path):
    write_minimal_release(tmp_path)
    (tmp_path / "manifest.json").unlink()
    report = verify_dataset(tmp_path)
    assert not report["valid"]
    assert any("Missing required dataset manifest" in error for error in report["errors"])


def test_release_verifier_rejects_unlisted_shard(tmp_path):
    write_minimal_release(tmp_path)
    assert verify_dataset(tmp_path)["valid"]
    # An extra empty shard does not change any counts or listed file hashes.
    with (tmp_path / "train-00001.csv").open("w", newline="") as handle:
        csv.writer(handle).writerow(COLUMNS)
    report = verify_dataset(tmp_path)
    assert not report["valid"]
    assert any("cover all loaded shards" in error for error in report["errors"])


def test_release_verifier_checks_split_membership_beyond_hashes(tmp_path):
    write_minimal_release(tmp_path, wrong_split_mapping=True)
    report = verify_dataset(tmp_path)
    assert not report["valid"]
    assert any("splits.json conversation membership mismatch" in error for error in report["errors"])


def test_release_verifier_checks_label_definitions(tmp_path):
    write_minimal_release(tmp_path)
    labels_path = tmp_path / "labels.json"
    labels = json.loads(labels_path.read_text())
    labels["politeness_label"]["0"] = "polite"
    labels_path.write_text(json.dumps(labels))
    report = verify_dataset(tmp_path)
    assert not report["valid"]
    assert any("canonical class mappings" in error for error in report["errors"])
