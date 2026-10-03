"""Benchmark configuration: load/validate benchmark.yaml, resolve the fixed set to
ontology leaf classes, and generate reproducible random sets."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from vcm_common.ontology import Ontology, get_ontology


@dataclass(frozen=True)
class Prompt:
    kind: str                            # "command" | "negative"
    say: str
    expected_class: Optional[str] = None  # leaf class key (commands only)


@dataclass
class BenchmarkConfig:
    wake_timeout_s: float
    result_timeout_s: float
    negatives: list[str]
    fixed_set: list[Prompt]
    random_set: dict
    targets: dict
    phrases: dict[str, list[str]]


def normalize_slot(s: str) -> str:
    """Case/format-insensitive slot key. '6 AM' -> '6 00 am', 'Drink water' -> 'drink water'."""
    s = s.strip().lower()
    s = re.sub(r"(?<![\d:])(\d{1,2})\s*(am|pm)\b", r"\1:00 \2", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def resolve_slot(ontology: Ontology, intent: str, slot: str) -> str:
    """Resolve a prompt's slot text to the ontology's exact slot value, else raise."""
    idx = {normalize_slot(v): v for v in ontology.slot_values.get(intent, ())}
    key = normalize_slot(slot)
    if key not in idx:
        raise ValueError(f"slot '{slot}' does not resolve to an allowed {intent} value")
    return idx[key]


def resolve_fixed_set(ontology: Ontology, raw: list[dict], negatives: list[str]) -> list[Prompt]:
    """Expand the fixed_set YAML ({negative: N} + slot shorthands) into Prompt objects.
    Raises on any prompt that does not resolve to exactly one leaf class."""
    prompts: list[Prompt] = []
    for item in raw:
        if "negative" in item:
            i = int(item["negative"])
            if not 0 <= i < len(negatives):
                raise ValueError(f"negative index {i} out of range (pool has {len(negatives)})")
            prompts.append(Prompt(kind="negative", say=negatives[i]))
            continue
        expect = item["expect"]
        slot = item.get("slot")
        if slot is not None:
            slot = resolve_slot(ontology, expect, slot)
            leaf = f"{expect}|{slot}"
        else:
            leaf = expect
        if leaf not in ontology.leaf_labels():
            raise ValueError(f"prompt {item['say']!r} -> {leaf!r} is not a leaf class")
        prompts.append(Prompt(kind="command", say=item["say"], expected_class=leaf))
    return prompts


def command_leaves(ontology: Ontology) -> list[str]:
    return [k for k in ontology.leaf_labels() if k not in ontology.local_only]


def generate_random_set(ontology: Ontology, phrases: dict[str, list[str]],
                        n_commands: int, n_negatives: int, negatives: list[str],
                        seed: int) -> list[Prompt]:
    """Draw n_commands distinct command leaves + n_negatives negatives, shuffled with `seed`."""
    rng = random.Random(seed)
    leaves = command_leaves(ontology)
    if n_commands > len(leaves):
        raise ValueError(f"n_commands {n_commands} exceeds {len(leaves)} command leaves")
    prompts: list[Prompt] = []
    for leaf in rng.sample(leaves, n_commands):
        say = rng.choice(phrases.get(leaf) or [leaf])
        prompts.append(Prompt(kind="command", say=say, expected_class=leaf))
    for neg in rng.sample(negatives, n_negatives):
        prompts.append(Prompt(kind="negative", say=neg))
    rng.shuffle(prompts)
    return prompts


def _load_phrases(path: Path) -> dict[str, list[str]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {k: list(v) for k, v in (data.get("phrases") or {}).items()}


def load_benchmark_config(path: Path, ontology: Optional[Ontology] = None) -> BenchmarkConfig:
    ont = ontology or get_ontology()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    negatives = [str(x) for x in data["negatives"]]
    fixed = resolve_fixed_set(ont, data["fixed_set"], negatives)
    phrases_path = Path(path).with_name("command_phrases.yaml")
    phrases = _load_phrases(phrases_path) if phrases_path.exists() else {}
    return BenchmarkConfig(
        wake_timeout_s=float(data["wake_timeout_s"]),
        result_timeout_s=float(data["result_timeout_s"]),
        negatives=negatives,
        fixed_set=fixed,
        random_set=dict(data["random_set"]),
        targets=dict(data["targets"]),
        phrases=phrases,
    )
