"""Configuration loading.

A run is described by one YAML file (see config/). Paths inside the config are
resolved relative to the repository root so no machine-specific absolute paths
are needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DatasetConfig:
    adapter: str  # kaldi_dir | hf_eurospeech | hf_voxpopuli | hf_mls_en | common_voice_local | fixture
    name: str
    language: str = "en"
    split: str = "test"
    revision: str | None = None
    path: str | None = None  # local directory for kaldi_dir / common_voice_local
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectionConfig:
    # "pilot" uses pilot_max_items / pilot_target_hours; "full" uses target_hours.
    target_hours: float | None = None  # Required only for a full baseline evaluation.
    pilot_target_hours: float = 0.02
    pilot_max_items: int = 5
    seed: int = 42
    min_duration_s: float = 0.5
    max_duration_s: float = 30.0
    # Upper bound on items scanned when streaming (protects against huge downloads).
    max_scan_items: int | None = 400
    allow_insufficient: bool = False


@dataclass
class ModelConfig:
    name: str = "ibm-granite/granite-4.0-1b-speech"
    revision: str | None = None
    dtype: str = "auto"  # auto | float32 | bfloat16 | float16
    device: str = "auto"  # auto | cpu | cuda
    # Retained as run metadata. Granite Speech's documented ASR prompt does not
    # take a forced-language argument.
    language: str | None = None
    max_new_tokens: int = 256
    batch_size: int = 1


@dataclass
class DeliverableConfig:
    """Fixed-size settings for the bilingual Step 2 exam artifact."""

    candidate_limit_per_language: int = 1_200
    rows_per_language: int = 1_000
    total_rows: int = 2_000


@dataclass
class RunConfig:
    run_name: str
    mode: str  # pilot | full
    dataset: DatasetConfig
    selection: SelectionConfig
    model: ModelConfig
    deliverable: DeliverableConfig = field(default_factory=DeliverableConfig)
    output_dir: str = "results"
    audio_cache_dir: str = "data/audio_cache"
    manifest_dir: str = "data/manifests"
    context_mode: str = "none"  # baseline is always "none"; see ner_asr.context

    @property
    def output_path(self) -> Path:
        return resolve(self.output_dir)

    @property
    def manifest_path(self) -> Path:
        return resolve(self.manifest_dir) / f"{self.run_name}_manifest.csv"

    @property
    def predictions_path(self) -> Path:
        return self.output_path / "predictions" / f"{self.run_name}_predictions.jsonl"

    @property
    def metrics_dir(self) -> Path:
        return self.output_path / "metrics"

    def target_seconds(self) -> float | None:
        hours = self.selection.pilot_target_hours if self.mode == "pilot" else self.selection.target_hours
        return None if hours is None else hours * 3600.0

    def max_items(self) -> int | None:
        return self.selection.pilot_max_items if self.mode == "pilot" else None


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


def load_config(path: str | Path, **overrides: Any) -> RunConfig:
    with open(resolve(path), encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    for key, value in overrides.items():
        raw[key] = value
    cfg = RunConfig(
        run_name=raw["run_name"],
        mode=raw.get("mode", "pilot"),
        dataset=DatasetConfig(**raw["dataset"]),
        selection=SelectionConfig(**raw.get("selection", {})),
        model=ModelConfig(**raw.get("model", {})),
        deliverable=DeliverableConfig(**raw.get("deliverable", {})),
        output_dir=raw.get("output_dir", "results"),
        audio_cache_dir=raw.get("audio_cache_dir", "data/audio_cache"),
        manifest_dir=raw.get("manifest_dir", "data/manifests"),
        context_mode=raw.get("context_mode", "none"),
    )
    if cfg.mode not in {"pilot", "full"}:
        raise ValueError(f"mode must be 'pilot' or 'full', got {cfg.mode!r}")
    if cfg.mode == "full" and cfg.selection.target_hours is None:
        raise ValueError(
            "Full mode needs selection.target_hours; set an explicit evaluation-duration target."
        )
    if cfg.deliverable.candidate_limit_per_language < cfg.deliverable.rows_per_language:
        raise ValueError(
            "deliverable.candidate_limit_per_language must be at least "
            "deliverable.rows_per_language."
        )
    if cfg.deliverable.rows_per_language <= 0:
        raise ValueError("deliverable.rows_per_language must be positive.")
    if cfg.deliverable.total_rows != 2 * cfg.deliverable.rows_per_language:
        raise ValueError(
            "deliverable.total_rows must equal twice deliverable.rows_per_language "
            "for the EN/DE artifact."
        )
    return cfg
