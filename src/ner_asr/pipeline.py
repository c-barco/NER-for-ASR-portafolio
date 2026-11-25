"""Dataset selection, transcription, scoring and Parquet export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import iter_records
from .config import RunConfig, load_config, resolve
from .deliverable import (
    MINIMUM_ROWS,
    export_step2_eurospeech_bilingual,
)
from .evaluate import build_results, export, summarize
from .runner import environment_info, run_transcription
from .selection import InsufficientAudioError, cache_audio, read_manifest, select_by_duration, write_manifest

def step_select(cfg: RunConfig, *, overwrite: bool = False):
    path = cfg.manifest_path
    if path.exists() and not overwrite:
        print(f"Manifest already exists, reusing it (fixed selection): {path}")
        return read_manifest(path), None
    sel = cfg.selection
    records = iter_records(cfg.dataset, sel)
    selected, report = select_by_duration(
        records,
        target_seconds=cfg.target_seconds(),
        seed=sel.seed,
        min_duration_s=sel.min_duration_s,
        max_duration_s=sel.max_duration_s,
        max_items=cfg.max_items(),
        max_scan_items=sel.max_scan_items if cfg.mode == "pilot" else None,
        # In pilot mode the duration is only a cap; in full mode a shortfall is an error
        # unless explicitly allowed (and then recorded in the selection report).
        allow_insufficient=sel.allow_insufficient or cfg.mode == "pilot",
    )
    if (
        cfg.mode == "pilot"
        and cfg.dataset.adapter == "hf_eurospeech"
        and len(selected) != sel.pilot_max_items
    ):
        raise InsufficientAudioError(
            f"EuroSpeech {cfg.dataset.language} pilot selected {len(selected)} recordings; "
            f"expected exactly {sel.pilot_max_items}. Increase selection.max_scan_items or inspect the source split."
        )
    cache_audio(selected, resolve(cfg.audio_cache_dir))
    write_manifest(selected, path, report)
    print(json.dumps(report.to_dict(), indent=2))
    return selected, report


def run_info(cfg: RunConfig) -> dict:
    return {
        "run_name": cfg.run_name,
        "mode": cfg.mode,
        "condition": "baseline_no_context",
        "model_name": cfg.model.name,
        "model_revision": cfg.model.revision,
        "language": cfg.model.language,
        "max_new_tokens": cfg.model.max_new_tokens,
        "batch_size": cfg.model.batch_size,
        "dataset": cfg.dataset.name,
        "dataset_revision": cfg.dataset.revision,
        "split": cfg.dataset.split,
    }


def step_transcribe(cfg: RunConfig, transcriber=None) -> dict:
    if cfg.context_mode != "none":
        raise ValueError("The Step 2 baseline must run with context_mode: none.")
    records = read_manifest(cfg.manifest_path)
    if transcriber is None:
        from .transcribe import Granite4ASRTranscriber

        transcriber = Granite4ASRTranscriber(cfg.model)
    # Only audio paths are passed on; prompt is fixed to None (no entity hints).
    fn = lambda paths: transcriber.transcribe_batch(paths, prompt=None)  # noqa: E731
    stats = run_transcription(
        records,
        fn,
        cfg.predictions_path,
        run_info=run_info(cfg),
        batch_size=cfg.model.batch_size,
    )
    print(stats)
    return stats


def step_evaluate(cfg: RunConfig) -> dict:
    records = read_manifest(cfg.manifest_path)
    rows = build_results(records, cfg.predictions_path)
    summary = summarize(rows, run_info={**run_info(cfg), "environment": environment_info()})
    export(rows, summary, cfg.metrics_dir, cfg.run_name)
    print("Mean WER:", summary["per_recording_wer_mean"])
    print("Mean CER:", summary["per_recording_cer_mean"])
    return summary


def _validate_bilingual_deliverable_configs(en_cfg: RunConfig, de_cfg: RunConfig) -> None:
    """Keep the two independently selected held-out streams comparable."""
    for expected_language, cfg in (("en", en_cfg), ("de", de_cfg)):
        if cfg.dataset.adapter != "hf_eurospeech":
            raise ValueError("The Step 2 parquet export is defined for dataset.adapter: hf_eurospeech.")
        if cfg.dataset.language != expected_language:
            raise ValueError(
                f"Expected the {expected_language!r} EuroSpeech config, got language={cfg.dataset.language!r}."
            )
        if cfg.dataset.split != "test":
            raise ValueError(
                "The Step 2 deliverable must use each language's held-out EuroSpeech test split."
            )
        if cfg.deliverable.rows_per_language != MINIMUM_ROWS:
            raise ValueError(
                f"The fixed bilingual deliverable needs {MINIMUM_ROWS} rows per language, got "
                f"{cfg.deliverable.rows_per_language} for {expected_language}."
            )
    if en_cfg.selection.seed != de_cfg.selection.seed:
        raise ValueError("English and German deliverable configs must use the same fixed selection seed.")
    if en_cfg.deliverable != de_cfg.deliverable:
        raise ValueError("English and German deliverable configs must use identical deliverable quotas.")


def step_export_deliverable(
    en_cfg: RunConfig,
    de_cfg: RunConfig,
    *,
    overwrite: bool = False,
):
    """Create the required balanced, bilingual one-file Step 2 parquet artifact."""
    _validate_bilingual_deliverable_configs(en_cfg, de_cfg)
    output = resolve("data/deliverable") / "step2_eurospeech_en_de.parquet"
    artifact, report = export_step2_eurospeech_bilingual(
        {
            "en": iter_records(en_cfg.dataset, en_cfg.selection),
            "de": iter_records(de_cfg.dataset, de_cfg.selection),
        },
        output_path=output,
        seed=en_cfg.selection.seed,
        candidate_limit_per_language=en_cfg.deliverable.candidate_limit_per_language,
        rows_per_language=en_cfg.deliverable.rows_per_language,
        minimum_rows_per_language=MINIMUM_ROWS,
        overwrite=overwrite,
    )
    return artifact, report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["select", "transcribe", "evaluate", "export-deliverable", "all"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--config-de", help="German EuroSpeech config; required for export-deliverable")
    ap.add_argument("--mode", choices=["pilot", "full"])
    ap.add_argument("--overwrite-manifest", action="store_true")
    ap.add_argument("--overwrite-deliverable", action="store_true")
    args = ap.parse_args(argv)
    overrides = {"mode": args.mode} if args.mode else {}
    cfg = load_config(args.config, **overrides)
    if args.mode == "full" and not cfg.run_name.endswith("_full"):
        cfg.run_name = f"{cfg.run_name}_full"
    if args.step in ("select", "all"):
        step_select(cfg, overwrite=args.overwrite_manifest)
    if args.step in ("transcribe", "all"):
        step_transcribe(cfg)
    if args.step in ("evaluate", "all"):
        step_evaluate(cfg)
    if args.step == "export-deliverable":
        if not args.config_de:
            ap.error("export-deliverable requires --config-de config/eurospeech_de.yaml")
        de_cfg = load_config(args.config_de, **overrides)
        step_export_deliverable(
            cfg,
            de_cfg,
            overwrite=args.overwrite_deliverable,
        )


if __name__ == "__main__":
    main()
