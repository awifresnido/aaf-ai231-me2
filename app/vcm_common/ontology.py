"""Closed command ontology shared by the edge service and the laptop app.

The file ``vcm_common/ontology.json`` is a vendored copy of
``tiny-vcm/configs/ontology.json``. ``tests/test_ontology.py`` fails if the two
drift apart while the tiny-vcm checkout is present, so re-copy it whenever the
training ontology changes (``python scripts/sync_ontology.py``).

Leaf-label order MUST match ``tiny-vcm/src/vcm_data_loader.create_label_mapping``:
sorted training labels, with each slotted intent expanded in slot-value order.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

LEAF_SEP = "|"
DEFAULT_ONTOLOGY_PATH = Path(__file__).with_name("ontology.json")


@dataclass(frozen=True)
class Ontology:
    """Closed-vocabulary command set.

    Attributes
    ----------
    fixed : tuple of str
        Intents without slots.
    slotted : tuple of str
        Intents that carry exactly one closed slot value.
    local_only : tuple of str
        Non-command classes (``UNKNOWN``, ``SILENCE``): never executed.
    slot_values : dict
        Allowed slot values per slotted intent, in canonical order.
    schema_version : str
    """

    fixed: tuple[str, ...]
    slotted: tuple[str, ...]
    local_only: tuple[str, ...]
    slot_values: dict[str, tuple[str, ...]]
    schema_version: str

    @classmethod
    def load(cls, path: Path = DEFAULT_ONTOLOGY_PATH) -> "Ontology":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        labels = data["labels"]
        return cls(
            fixed=tuple(labels["fixed"]),
            slotted=tuple(labels["slotted"]),
            local_only=tuple(labels["local_only"]),
            slot_values={k: tuple(v) for k, v in data["slot_values"].items()},
            schema_version=data["schema_version"],
        )

    # ------------------------------------------------------------------
    @property
    def intents(self) -> tuple[str, ...]:
        """Executable intents (fixed + slotted)."""
        return self.fixed + self.slotted

    @property
    def all_labels(self) -> tuple[str, ...]:
        return self.fixed + self.slotted + self.local_only

    def leaf_labels(self) -> list[str]:
        """Class keys in the order tiny-vcm's leaf label_mode uses."""
        leaves: list[str] = []
        for label in sorted(self.all_labels):
            if label in self.slot_values:
                leaves.extend(f"{label}{LEAF_SEP}{v}" for v in self.slot_values[label])
            else:
                leaves.append(label)
        return leaves

    def intent_labels(self) -> list[str]:
        """Class keys in the order tiny-vcm's intent label_mode uses."""
        return sorted(self.all_labels)

    @staticmethod
    def decode(class_key: str) -> tuple[str, Optional[str]]:
        """``'BRIGHTNESS|60 percent'`` -> ``('BRIGHTNESS', '60 percent')``."""
        if LEAF_SEP in class_key:
            intent, slot = class_key.split(LEAF_SEP, 1)
            return intent, slot
        return class_key, None

    def is_command(self, intent: str) -> bool:
        return intent in self.intents

    def validate(self, intent: str, slot: Optional[str]) -> Optional[str]:
        """Return an error string if (intent, slot) is not executable, else None."""
        if intent not in self.intents:
            return f"'{intent}' is not an executable intent"
        if intent in self.slotted:
            if slot is None:
                return f"{intent} needs a slot value but the model gave none"
            if slot not in self.slot_values[intent]:
                return f"'{slot}' is not an allowed {intent} value"
        elif slot is not None:
            return f"{intent} takes no slot, got '{slot}'"
        return None


@lru_cache(maxsize=1)
def get_ontology() -> Ontology:
    return Ontology.load()
