"""Model registry: scans ``models/<task>/<id>/manifest.yaml`` and loads engines
on demand. New checkpoints are deployed by copying a folder in and pressing
'Rescan' in the UI; no service restart and no code change."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from vcm_common.manifest import ModelManifest

from .engines.base import EngineUnavailable, VCMEngine, build_vcm_engine
from .engines.scripted import SCRIPTED_MODEL_ID, ScriptedEngine, scripted_manifest

log = logging.getLogger(__name__)


@dataclass
class Entry:
    manifest: ModelManifest
    status: str = "not_loaded"  # not_loaded | ready | unavailable | invalid
    detail: Optional[str] = None
    engine: Optional[VCMEngine] = None
    size_bytes: Optional[int] = None
    sha256_12: Optional[str] = None
    params: Optional[int] = None

    def summary(self) -> dict:
        m = self.manifest
        return {
            "id": m.id,
            "display_name": m.display_name,
            "description": m.description,
            "based_on": m.based_on,
            "engine": m.engine,
            "members": list(m.members),
            "label_scheme": m.label_scheme,
            "n_classes": len(m.labels),
            "quantization": m.quantization,
            "architecture": m.architecture,
            "lineage": m.lineage.model_dump(),
            "offline_metrics": m.offline_metrics,
            "scores": m.scores.model_dump(),
            "min_confidence": m.reject.min_confidence,
            "window_s": m.features.max_duration_s,
            "status": self.status,
            "detail": self.detail,
            "size_bytes": self.size_bytes,
            "sha256_12": self.sha256_12,
            "params": self.params,
        }


@dataclass
class ModelRegistry:
    models_dir: Path
    engine_settings: dict = field(default_factory=dict)
    entries: dict[str, Entry] = field(default_factory=dict)

    def scan(self) -> None:
        """(Re)read every manifest. Loaded engines of unchanged models are kept."""
        found: dict[str, Entry] = {SCRIPTED_MODEL_ID: self._scripted_entry()}
        vcm_dir = self.models_dir / "vcm"
        for folder in sorted(p for p in vcm_dir.glob("*") if (p / "manifest.yaml").exists()):
            try:
                m = ModelManifest.load(folder)
            except Exception as exc:  # malformed YAML/manifest must not kill the service
                found[folder.name] = Entry(
                    manifest=ModelManifest(id=folder.name, display_name=folder.name,
                                           task="vcm", engine="invalid"),
                    status="invalid", detail=f"manifest error: {exc}",
                )
                continue
            prev = self.entries.get(m.id)
            entry = Entry(manifest=m)
            w = m.weights_path()
            if w is not None and w.exists():
                entry.size_bytes = w.stat().st_size
                entry.sha256_12 = _sha256_12(w)
                if prev and prev.status == "ready" and prev.sha256_12 == entry.sha256_12:
                    entry = prev
            elif w is not None:
                entry.status, entry.detail = "unavailable", f"weights missing: {w.name}"
            found[m.id] = entry
        self.entries = found

    def _scripted_entry(self) -> Entry:
        prev = self.entries.get(SCRIPTED_MODEL_ID)
        if prev:
            return prev
        engine = ScriptedEngine(scripted_manifest())
        return Entry(manifest=engine.manifest, status="ready", engine=engine)

    def scripted(self) -> ScriptedEngine:
        engine = self.entries[SCRIPTED_MODEL_ID].engine
        assert isinstance(engine, ScriptedEngine)
        return engine

    def get(self, model_id: str) -> VCMEngine:
        """Return a loaded engine, loading it on first use."""
        entry = self.entries.get(model_id)
        if entry is None:
            raise EngineUnavailable(f"no model with id '{model_id}'")
        if entry.status == "ready" and entry.engine is not None:
            return entry.engine
        if entry.status == "invalid":
            raise EngineUnavailable(entry.detail or "invalid manifest")
        if entry.manifest.engine == "agreement":
            return self._load_agreement(entry)
        try:
            engine = build_vcm_engine(entry.manifest, self.engine_settings)
            engine.load()
        except EngineUnavailable as exc:
            entry.status, entry.detail = "unavailable", str(exc)
            raise
        entry.engine, entry.status, entry.detail = engine, "ready", None
        entry.params = engine.param_count()
        log.info("loaded model %s (%s params)", model_id, entry.params)
        return engine

    def _load_agreement(self, entry: Entry) -> VCMEngine:
        """Load every member; the agreement entry is ready only if all load.

        Verifies every member shares the agreement manifest's label list (same
        order) and fails loudly otherwise.
        """
        m = entry.manifest
        members: list[VCMEngine] = []
        total = 0
        for mid in m.members:
            if mid not in self.entries:
                entry.status, entry.detail = "unavailable", f"member {mid} not registered"
                raise EngineUnavailable(entry.detail)
            mem_manifest = self.entries[mid].manifest
            if mem_manifest.labels != m.labels:
                entry.status, entry.detail = "unavailable", \
                    f"labels mismatch: {mid} vs {m.id}"
                raise EngineUnavailable(entry.detail)
            member = self.get(mid)
            members.append(member)
            pc = member.param_count()
            if pc is not None:
                total += pc
        from .engines.agreement import AgreementEngine

        engine = AgreementEngine(m)
        entry.engine, entry.status, entry.detail = engine, "ready", None
        entry.params = total
        log.info("loaded agreement %s (%d params over %d members)", m.id, total, len(members))
        return engine

    def selectable_ids(self) -> list[str]:
        return [k for k, e in self.entries.items() if k != SCRIPTED_MODEL_ID]

    def summaries(self) -> list[dict]:
        return [e.summary() for k, e in self.entries.items() if k != SCRIPTED_MODEL_ID]


def _sha256_12(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]
