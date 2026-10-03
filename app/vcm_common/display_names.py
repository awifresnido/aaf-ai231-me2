"""Friendly display names for models, datasets and recording sessions.

``config/display_names.yaml`` is the single source for every friendly name.
Internal IDs stay unchanged everywhere in code, manifests, CSV exports and logs;
this module only maps an ID to the name the UI shows, and vice-versa for lookup.

A dataset may carry an optional ``display_id``: the public alias shown wherever a
dataset id is displayed (so an internal id such as ``mark_optionb`` never reaches
the UI). ``as_dict()`` keys the datasets map by that alias, so downstream consumers
only ever see the public ids.

Loaded once at laptop startup and validated: every registered model ID must have
an entry (fail early with a clear error if one is missing).
"""
from __future__ import annotations

from pathlib import Path

import yaml


class DisplayNames:
    def __init__(self, data: dict):
        self._models: dict[str, dict] = data.get("models", {}) or {}
        self._datasets: dict[str, dict] = data.get("datasets", {}) or {}
        self._sessions: dict[str, dict] = data.get("sessions", {}) or {}

    @classmethod
    def load(cls, path: Path) -> "DisplayNames":
        if not path.exists():
            raise FileNotFoundError(f"display names file not found: {path}")
        return cls(yaml.safe_load(path.read_text(encoding="utf-8")) or {})

    # -- model ---------------------------------------------------------------
    def model_name(self, model_id: str) -> str:
        return self._models.get(model_id, {}).get("name", model_id)

    def model(self, model_id: str) -> dict:
        return self._models.get(model_id, {})

    def model_role(self, model_id: str) -> str:
        """Optional role tag for the UI ("" when absent). Only "benchmark" today:
        a reference model shown for comparison, ordered last in the picker."""
        return self._models.get(model_id, {}).get("role", "") or ""

    def model_ids(self) -> list[str]:
        return list(self._models)

    # -- dataset / session ---------------------------------------------------
    def dataset_name(self, dataset_id: str) -> str:
        return self._datasets.get(dataset_id, {}).get("name", dataset_id)

    def dataset_display_id(self, dataset_id: str) -> str:
        """Public alias for a dataset id (``display_id`` if set, else the id)."""
        return self._datasets.get(dataset_id, {}).get("display_id") or dataset_id

    def session_name(self, session_id: str) -> str:
        return self._sessions.get(session_id, {}).get("name", session_id)

    def dataset(self, dataset_id: str) -> dict:
        return self._datasets.get(dataset_id, {})

    # -- validation ----------------------------------------------------------
    def missing_model_names(self, registered_ids: list[str]) -> list[str]:
        """IDs present in the registry but absent from the names map."""
        return [i for i in registered_ids if i not in self._models]

    def order_models(self, registered_ids: list[str]) -> list[str]:
        """Registered IDs in display order: normal models first, then role=benchmark."""
        return sorted(registered_ids, key=lambda i: (self.model_role(i) == "benchmark",))

    def as_dict(self) -> dict:
        return {
            "models": self._models,
            # Key the datasets map by the public display_id so the internal id
            # (e.g. mark_optionb) never appears in any API payload.
            "datasets": {self.dataset_display_id(k): v for k, v in self._datasets.items()},
            "sessions": self._sessions,
        }
