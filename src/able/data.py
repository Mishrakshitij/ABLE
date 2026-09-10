"""Streaming PERPDSCD readers, response examples, and release verification.

The CSV preserves source text and identifiers. ``turn_index`` is the canonical
position within a conversation; it remains unique even when ``Turn_id`` is not.
Only Doctor turns become response targets, and their prompts contain earlier
turns only. This module uses the Python standard library.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import itertools
import io
import json
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

SPLITS = ("train", "validation", "test")
SOURCE_COLUMNS = (
    "Convo_id", "Turn_id", "Speaker", "Utterance", "Gender", "Age",
    "Persona", "Issue", "Physical disability",
)
LABEL_COLUMNS = ("persona_label", "gender_age_label", "politeness_label", "empathy_label")
COLUMNS = SOURCE_COLUMNS + ("row_id", "turn_index", "split") + LABEL_COLUMNS
CLASS_COUNTS = {"persona_label": 19, "gender_age_label": 6,
                "politeness_label": 3, "empathy_label": 3}
GENDER_AGE_CLASSES = (
    ("Male", "Younger"), ("Male", "Middle Aged"), ("Male", "Older"),
    ("Female", "Younger"), ("Female", "Middle Aged"), ("Female", "Older"),
)
# O, C, E, A, N intensities in Appendix A.1.2 order, using 0 for low and 1 for high.
PERSONA_TRAITS = (
    (1, 1, 1, 1, 0), (0, 1, 1, 1, 0), (1, 0, 1, 1, 0), (0, 0, 1, 1, 0),
    (1, 1, 0, 1, 0), (0, 1, 0, 1, 0), (1, 0, 0, 1, 0), (0, 0, 0, 1, 0),
    (1, 1, 1, 0, 0), (0, 1, 1, 0, 0), (1, 0, 1, 0, 0), (0, 0, 1, 0, 0),
    (0, 0, 0, 0, 0), (1, 1, 0, 0, 0), (0, 1, 0, 0, 0), (1, 1, 1, 1, 1),
    (0, 0, 1, 1, 1), (0, 1, 0, 1, 1), (0, 0, 0, 1, 1),
)


@dataclass(frozen=True)
class Example:
    """One response target; optional labels are never silently invented."""

    example_id: str
    conversation_id: str
    turn_id: int
    prompt: str
    response: str
    persona_label: int
    gender_age_label: int
    politeness_label: int | None
    empathy_label: int | None
    context: str = ""


def persona_class(persona: str) -> int:
    """Map an OCEAN profile to its zero-based Appendix A.1.2 class."""
    matches = re.findall(
        r"\b(High|Low)\s+(Openness|Conscientiousness|Extraversion|Agreeableness|Neuroticism)\b",
        persona, flags=re.IGNORECASE,
    )
    values = {trait.casefold(): int(level.casefold() == "high") for level, trait in matches}
    if len(matches) != 5 or len(values) != 5:
        raise ValueError(f"Expected each of the five OCEAN traits once: {persona!r}")
    traits = tuple(values[name] for name in (
        "openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"))
    try:
        return PERSONA_TRAITS.index(traits)
    except ValueError as exc:
        raise ValueError(f"Profile is outside the 19 PERPDSCD personas: {persona!r}") from exc


def gender_age_class(gender: str, age: str) -> int:
    """Class order: male younger/middle/older, then female younger/middle/older."""
    return GENDER_AGE_CLASSES.index((gender, age))


def label_mappings() -> dict:
    """Canonical public class definitions, shared by readers and verification."""
    return {
        "persona_label": {
            str(index): dict(zip(("O", "C", "E", "A", "N"),
                                 ("high" if level else "low" for level in traits)))
            for index, traits in enumerate(PERSONA_TRAITS)
        },
        "gender_age_label": {
            str(index): {"gender": gender, "age": age}
            for index, (gender, age) in enumerate(GENDER_AGE_CLASSES)
        },
        "politeness_label": {"0": "impolite", "1": "neutral", "2": "polite"},
        "empathy_label": {"0": "non_empathetic", "1": "neutral", "2": "empathetic"},
    }


def format_prompt(
    history: Iterable[Mapping[str, str] | tuple[str, str]], *,
    gender: str, age: str, persona: str,
) -> str:
    """Format a profile and prior turns for the next Doctor response.

    History accepts ``(speaker, text)`` pairs or chat ``role``/``content``
    mappings. Patient/user and Doctor/assistant names are supported. The caller
    must supply only observed context, never the target response.
    """
    lines = [f"Gender: {gender}", f"Age: {age}", f"Persona: {persona}", ""]
    roles = {"patient": "User", "user": "User", "doctor": "Doctor", "assistant": "Doctor"}
    for turn in history:
        if isinstance(turn, Mapping):
            speaker, text = turn["role"], turn["content"]
        else:
            speaker, text = turn
        if speaker.casefold() not in roles:
            raise ValueError(f"Unsupported conversation role: {speaker!r}")
        lines.append(f"{roles[speaker.casefold()]}: {text}")
    lines.append("Doctor:")
    return "\n".join(lines)


def _shards(dataset_dir: str | Path, split: str | None = None) -> list[Path]:
    root = Path(dataset_dir)
    if split is not None and split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
    result = []
    for part in (SPLITS if split is None else (split,)):
        paths = set(root.rglob(f"{part}-*.csv")) | set(root.rglob(f"{part}-*.csv.gz"))
        if not paths:
            raise FileNotFoundError(f"No {part} CSV shards under {root}")
        result.extend(sorted(paths))
    return result


def iter_rows(dataset_dir: str | Path, split: str | None = None) -> Iterator[dict[str, str]]:
    """Read CSV shards without loading the corpus into memory."""
    for path in _shards(dataset_dir, split):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(COLUMNS):
                raise ValueError(f"Unexpected CSV schema in {path}: {reader.fieldnames}")
            for line_number, row in enumerate(reader, 2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Malformed CSV row in {path}:{line_number}")
                if split is not None and row["split"] != split:
                    raise ValueError(f"Incorrect split in {path}:{line_number}")
                yield row


def _label(row: Mapping[str, str], field: str, *, optional: bool = False) -> int | None:
    value = row[field]
    if value == "" and optional:
        return None
    try:
        label = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field} for row {row.get('row_id')}: {value!r}") from exc
    if not 0 <= label < CLASS_COUNTS[field]:
        raise ValueError(f"Out-of-range {field}: {label}")
    return label


def iter_examples(
    dataset_dir: str | Path, split: str, max_examples: int | None = None,
) -> Iterator[Example]:
    """Yield Doctor examples with causal context and canonical unique IDs."""
    if max_examples is not None and max_examples < 0:
        raise ValueError("max_examples must be nonnegative")
    if max_examples == 0:
        return
    seen = set()
    count = 0
    for conversation_id, conversation in itertools.groupby(
        iter_rows(dataset_dir, split), key=lambda row: row["Convo_id"],
    ):
        if conversation_id in seen:
            raise ValueError(f"Conversation {conversation_id} is not contiguous")
        seen.add(conversation_id)
        history: list[tuple[str, str]] = []
        for position, row in enumerate(conversation, 1):
            if int(row["turn_index"]) != position:
                raise ValueError(f"Incorrect turn_index in conversation {conversation_id}")
            if row["Speaker"] not in ("Patient", "Doctor"):
                raise ValueError(f"Unknown speaker {row['Speaker']!r}")
            if row["Speaker"] == "Doctor":
                if not history:
                    raise ValueError(f"Doctor response without context in {conversation_id}")
                yield Example(
                    example_id=f"{conversation_id}:{position}",
                    conversation_id=conversation_id, turn_id=position,
                    prompt=format_prompt(history, gender=row["Gender"], age=row["Age"], persona=row["Persona"]),
                    response=row["Utterance"],
                    persona_label=_label(row, "persona_label"),
                    gender_age_label=_label(row, "gender_age_label"),
                    politeness_label=_label(row, "politeness_label", optional=True),
                    empathy_label=_label(row, "empathy_label", optional=True),
                    context="\n".join(f"{'User' if speaker == 'Patient' else 'Doctor'}: {text}" for speaker, text in history),
                )
                count += 1
                if max_examples is not None and count >= max_examples:
                    return
            history.append((row["Speaker"], row["Utterance"]))


def conversation_fingerprint(rows: Sequence[Mapping[str, str]]) -> str:
    """Hash complete normalized dialogue content, independent of IDs/profile."""
    content = [(row["Speaker"], re.sub(r"\s+", " ", row["Utterance"]).strip().casefold()) for row in rows]
    return hashlib.sha256(json.dumps(content, ensure_ascii=False).encode("utf-8")).hexdigest()


def verify_dataset(
    dataset_dir: str | Path, source_archive: str | Path | None = None, *,
    require_manifest: bool = True,
) -> dict:
    """Check schema, labels, integrity, and conversation leakage across splits.

    ``valid`` is false on structural or manifest errors. Missing optional style
    labels are counted explicitly; a manifest can require complete coverage with
    ``require_style_labels: true``. Repeated source Turn_id values are accepted
    because canonical turn_index and row_id provide unique addresses.
    Release metadata is required by default; ``require_manifest=False`` allows
    structural checks of small fixtures before release metadata is assembled.
    """
    root = Path(dataset_dir)
    errors: list[str] = []
    conversation_splits: dict[str, str] = {}
    fingerprints: dict[str, str] = {}
    split_conversations: dict[str, list[str]] = {split: [] for split in SPLITS}
    row_ids: set[int] = set()
    stats = {}
    duplicate_conversations = 0
    total_missing = 0
    source_hashes: list[bytes] | None = None
    try:
        if source_archive is not None:
            with zipfile.ZipFile(source_archive) as archive:
                members = [name for name in archive.namelist() if Path(name).name == "PERPDSCD.csv"]
                if len(members) != 1:
                    raise ValueError("Source archive must contain exactly one PERPDSCD.csv")
                with archive.open(members[0]) as raw:
                    source_reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
                    if source_reader.fieldnames != list(SOURCE_COLUMNS):
                        raise ValueError("Unexpected source archive CSV schema")
                    source_hashes = [hashlib.sha256(json.dumps(
                        [row[field] for field in SOURCE_COLUMNS], ensure_ascii=False,
                    ).encode("utf-8")).digest() for row in source_reader]
        for split in SPLITS:
            counts = Counter()
            distributions = {field: Counter() for field in LABEL_COLUMNS}
            lengths = []
            for conversation_id, group in itertools.groupby(
                iter_rows(root, split), key=lambda row: row["Convo_id"],
            ):
                rows = list(group)
                if conversation_id in conversation_splits:
                    errors.append(f"Conversation {conversation_id} occurs more than once/across splits")
                conversation_splits[conversation_id] = split
                split_conversations[split].append(conversation_id)
                fingerprint = conversation_fingerprint(rows)
                if fingerprint in fingerprints:
                    duplicate_conversations += 1
                    if fingerprints[fingerprint] != split:
                        errors.append(f"Duplicate conversation content leaks into {split}: {conversation_id}")
                fingerprints[fingerprint] = split
                counts["conversations"] += 1
                lengths.append(len(rows))
                expected_profile = tuple(rows[0][key] for key in SOURCE_COLUMNS[4:])
                for position, row in enumerate(rows, 1):
                    counts["utterances"] += 1
                    counts["doctor_responses"] += row["Speaker"] == "Doctor"
                    counts["patient_utterances"] += row["Speaker"] == "Patient"
                    if any(not row[field].strip() for field in SOURCE_COLUMNS):
                        errors.append(f"Empty source field at row {row['row_id']}")
                    if row["Speaker"] not in ("Patient", "Doctor"):
                        errors.append(f"Invalid speaker at row {row['row_id']}")
                    if position == 1 and row["Speaker"] != "Patient":
                        errors.append(f"Conversation {conversation_id} does not start with Patient")
                    if int(row["turn_index"]) != position:
                        errors.append(f"Invalid turn_index at row {row['row_id']}")
                    row_id = int(row["row_id"])
                    if row_id < 0 or row_id in row_ids:
                        errors.append(f"Invalid/duplicate row_id {row_id}")
                    row_ids.add(row_id)
                    if source_hashes is not None:
                        digest = hashlib.sha256(json.dumps(
                            [row[field] for field in SOURCE_COLUMNS], ensure_ascii=False,
                        ).encode("utf-8")).digest()
                        if not 0 <= row_id < len(source_hashes) or digest != source_hashes[row_id]:
                            errors.append(f"Original source fields changed at row {row_id}")
                    if tuple(row[key] for key in SOURCE_COLUMNS[4:]) != expected_profile:
                        errors.append(f"Inconsistent profile in {conversation_id}")
                    for field in LABEL_COLUMNS:
                        label = _label(row, field, optional=field in LABEL_COLUMNS[2:])
                        distributions[field][str(label) if label is not None else "missing"] += 1
                        if label is None:
                            total_missing += 1
                    if _label(row, "persona_label") != persona_class(row["Persona"]):
                        errors.append(f"Persona class mismatch at row {row_id}")
                    if _label(row, "gender_age_label") != gender_age_class(row["Gender"], row["Age"]):
                        errors.append(f"Gender-age class mismatch at row {row_id}")
            stats[split] = {**dict(counts), "min_turns": min(lengths, default=0),
                            "max_turns": max(lengths, default=0),
                            "mean_turns": sum(lengths) / len(lengths) if lengths else 0,
                            "labels": {field: dict(sorted(counter.items())) for field, counter in distributions.items()}}
        if source_hashes is not None and row_ids != set(range(len(source_hashes))):
            errors.append("Distributed rows do not cover the source archive exactly")
        if row_ids != set(range(len(row_ids))):
            errors.append("row_id values must cover consecutive integers from zero")
        manifests = list(root.rglob("manifest.json"))
        if require_manifest and not manifests:
            errors.append("Missing required dataset manifest.json")
        if len(manifests) > 1:
            errors.append("Multiple dataset manifests found")
        metadata_root = manifests[0].parent if manifests else root
        for filename, expected in (("labels.json", label_mappings()), ("splits.json", split_conversations)):
            metadata_path = metadata_root / filename
            if not metadata_path.is_file():
                if require_manifest or manifests:
                    errors.append(f"Missing required dataset {filename}")
                continue
            actual = json.loads(metadata_path.read_text(encoding="utf-8"))
            if filename == "labels.json":
                if actual != expected:
                    errors.append("labels.json does not match canonical class mappings")
            elif not isinstance(actual, dict) or set(actual) != set(SPLITS):
                errors.append("splits.json must contain exactly train, validation, and test")
            else:
                for split in SPLITS:
                    values = actual[split]
                    if (not isinstance(values, list)
                            or any(not isinstance(value, str) for value in values)
                            or len(values) != len(set(values))
                            or set(values) != set(expected[split])):
                        errors.append(f"splits.json conversation membership mismatch: {split}")
        if manifests:
            manifest_path = manifests[0]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("require_style_labels") and total_missing:
                errors.append(f"Missing required style labels: {total_missing}")
            listed_files: set[Path] = set()
            listed_csvs: set[Path] = set()
            for entry in manifest.get("files", []):
                path = (manifest_path.parent / entry["path"]).resolve()
                if not path.is_relative_to(manifest_path.parent.resolve()):
                    errors.append(f"Manifest path escapes dataset: {entry['path']}")
                    continue
                if path in listed_files:
                    errors.append(f"Duplicate manifest file entry: {entry['path']}")
                listed_files.add(path)
                if str(path).endswith((".csv", ".csv.gz")):
                    listed_csvs.add(path)
                if not path.is_file():
                    errors.append(f"Missing manifest file {entry['path']}")
                elif hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                    errors.append(f"Checksum mismatch: {entry['path']}")
            if listed_csvs != {path.resolve() for path in _shards(root)}:
                errors.append("Manifest CSV entries must cover all loaded shards exactly")
            for filename in ("labels.json", "splits.json"):
                if (metadata_root / filename).resolve() not in listed_files:
                    errors.append(f"Manifest must include a checksum for {filename}")
            if "splits" in manifest and manifest["splits"] != stats:
                errors.append("Manifest split statistics do not match CSV contents")
    except (ValueError, KeyError, OSError, TypeError, zipfile.BadZipFile) as exc:
        errors.append(str(exc))
    return {"valid": not errors, "errors": errors, "splits": stats,
            "conversations": len(conversation_splits), "utterances": len(row_ids),
            "duplicate_conversations": duplicate_conversations,
            "missing_style_labels": total_missing}
