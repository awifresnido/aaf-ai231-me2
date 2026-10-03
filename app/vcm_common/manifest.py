"""Model package manifest: every model the app can load is a folder with a
``manifest.yaml`` next to (or pointing at) its weights. The app never hard-codes a model."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class FeatureSpec(BaseModel):
    sample_rate: int = 16_000
    n_mels: int = 40
    n_fft: int = 512
    hop_length: int = 160
    f_min: int = 50
    f_max: int = 7600
    top_db: float = 80.0
    trim_db: Optional[float] = 40.0
    max_duration_s: float = 3.0


class RejectSpec(BaseModel):
    min_confidence: float = 0.70
    non_command_labels: list[str] = Field(default_factory=lambda: ["UNKNOWN", "SILENCE"])


class Lineage(BaseModel):
    training_phase: Optional[str] = None
    training_data: Optional[str] = None
    git_commit: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    trained_on: Optional[str] = None
    notes: Optional[str] = None


class ScoreEntry(BaseModel):
    """One training/validation/test score block, copied from tiny-vcm reports."""

    dataset_id: Optional[str] = None
    split: Optional[str] = None
    metrics: dict[str, float] = Field(default_factory=dict)
    threshold: Optional[float] = None
    source: Optional[str] = None
    note: Optional[str] = None
    status: Literal["reported", "not_evaluated", "not_recorded", "not_applicable"] = "reported"

    @model_validator(mode="after")
    def _reported_requires_metrics(self) -> "ScoreEntry":
        if self.status == "reported" and not self.metrics:
            raise ValueError("status 'reported' requires non-empty metrics")
        return self


def _empty_score() -> ScoreEntry:
    return ScoreEntry(status="not_evaluated", metrics={})


class Scores(BaseModel):
    training: ScoreEntry = Field(default_factory=_empty_score)
    validation: ScoreEntry = Field(default_factory=_empty_score)
    test: ScoreEntry = Field(default_factory=_empty_score)


class ModelManifest(BaseModel):
    """Self-description of one model package."""

    id: str
    display_name: str
    description: Optional[str] = None
    based_on: Optional[str] = None
    task: Literal["vcm", "wakeword"]
    engine: str  # torch_tinyvcm | onnx_tinyvcm | scripted | agreement
    weights: Optional[str] = None  # relative to the manifest folder, or absolute
    architecture: dict[str, Any] = Field(default_factory=dict)
    label_scheme: Literal["leaf", "intent", "wake"] = "leaf"
    labels: list[str] = Field(default_factory=list)
    features: FeatureSpec = Field(default_factory=FeatureSpec)
    reject: RejectSpec = Field(default_factory=RejectSpec)
    quantization: str = "fp32"
    lineage: Lineage = Field(default_factory=Lineage)
    offline_metrics: dict[str, Any] = Field(default_factory=dict)
    engine_options: dict[str, Any] = Field(default_factory=dict)
    members: list[str] = Field(default_factory=list)   # agreement engine only
    scores: Scores = Field(default_factory=Scores)

    # filled in by the registry, not by the YAML
    folder: Optional[str] = None

    @classmethod
    def load(cls, folder: Path) -> "ModelManifest":
        data = yaml.safe_load((folder / "manifest.yaml").read_text(encoding="utf-8"))
        m = cls(**data)
        m.folder = str(folder)
        return m

    def weights_path(self) -> Optional[Path]:
        if not self.weights or not self.folder:
            return None
        return (Path(self.folder) / self.weights).resolve()
