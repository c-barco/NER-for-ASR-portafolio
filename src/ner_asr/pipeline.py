"""High-level steps used by both the notebook and the command line.

    python -m ner_asr.pipeline select    --config config/baseline_pilot.yaml
    python -m ner_asr.pipeline transcribe --config config/baseline_pilot.yaml
    python -m ner_asr.pipeline evaluate  --config config/baseline_pilot.yaml
    python -m ner_asr.pipeline all       --config config/baseline_pilot.yaml [--mode full]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import iter_records
from .config import RunConfig, load_config, resolve
from .evaluate import build_results, export, summarize
from .runner import environment_info, run_transcription
from .selection import cache_audio, read_manifest, select_by_duration, write_manifest


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
        from .transcribe import QwenASRTranscriber

        transcriber = QwenASRTranscriber(cfg.model)
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
    paths = export(rows, summary, cfg.metrics_dir, cfg.run_name)
    print(json.dumps({k: v for k, v in summary.items() if k != "run_info"}, indent=2, default=str))
    print({k: str(v) for k, v in paths.items()})
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["select", "transcribe", "evaluate", "all"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["pilot", "full"])
    ap.add_argument("--overwrite-manifest", action="store_true")
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


if __name__ == "__main__":
    main()
