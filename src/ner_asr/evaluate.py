"""Baseline WER and CER exports."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .metrics import corpus_scores, describe, score_utterance
from .normalize import NORMALIZATION_VERSION, lecturer_normalize, normalize_text
from .records import AudioRecord
from .runner import load_predictions

RESULT_COLUMNS = [
    "sample_id",
    "audio_path",
    "duration_seconds",
    "speaker_id",
    "source_split",
    "reference_text",
    "prediction",
    "reference_norm",
    "prediction_norm",
    "wer",
    "cer",
    "status",
    "error_message",
    "runtime_s",
    "n_generated_tokens",
    "flag_hit_token_limit",
    "flag_empty_prediction",
    "flag_short_vs_reference",
    "flag_wer_above_1",
    # lab_06-compatible aliases (no-context condition)
    "wav_path",
    "label",
    "pred_without_context",
    "wer_without_context",
    "cer_without_context",
]


def build_results(records: list[AudioRecord], predictions_path: Path) -> list[dict]:
    preds = load_predictions(predictions_path)
    rows = []
    for rec in records:
        p = preds.get(rec.sample_id)
        status = p["status"] if p else "missing"
        prediction = p.get("prediction") if p else None
        ref_norm = normalize_text(rec.reference_text)
        row = {
            "sample_id": rec.sample_id,
            "audio_path": rec.audio_ref,
            "duration_seconds": rec.duration_seconds,
            "speaker_id": rec.speaker_id,
            "source_split": rec.source_split,
            "reference_text": rec.reference_text,
            "prediction": prediction,
            "reference_norm": ref_norm,
            "prediction_norm": None,
            "wer": math.nan,
            "cer": math.nan,
            "status": status,
            "error_message": (p or {}).get("error_message") if p else "no prediction recorded",
            "runtime_s": (p or {}).get("runtime_s"),
            "n_generated_tokens": (p or {}).get("n_generated_tokens"),
            "flag_hit_token_limit": bool((p or {}).get("hit_token_limit")),
            "flag_empty_prediction": False,
            "flag_short_vs_reference": False,
            "flag_wer_above_1": False,
        }
        if status == "ok":
            pred_norm = normalize_text(prediction)
            s = score_utterance(ref_norm, pred_norm)
            ref_w = len(ref_norm.split())
            row.update(
                prediction_norm=pred_norm,
                wer=s.wer,
                cer=s.cer,
                _score=s,
                flag_empty_prediction=(pred_norm == ""),
                # Possible truncation: far fewer words than the reference on a non-trivial clip.
                flag_short_vs_reference=(ref_w >= 6 and len(pred_norm.split()) < 0.5 * ref_w),
                flag_wer_above_1=(not math.isnan(s.wer) and s.wer > 1.0),
            )
        row.update(
            wav_path=row["audio_path"],
            label=row["reference_norm"],
            pred_without_context=row["prediction_norm"],
            wer_without_context=row["wer"],
            cer_without_context=row["cer"],
        )
        rows.append(row)
    return rows


def summarize(rows: list[dict], *, run_info: dict | None = None) -> dict:
    ok = [r for r in rows if r["status"] == "ok"]
    failed = [r for r in rows if r["status"] != "ok"]
    scored = [r for r in ok if not math.isnan(r["wer"])]
    scores = [r["_score"] for r in scored]
    # Conservative variant: failed recordings count as empty predictions (all deletions).
    from .metrics import score_utterance as _su

    with_failures = scores + [_su(r["reference_norm"], "") for r in failed if r["reference_norm"]]
    lecturer_style = []
    for r in ok:
        if r["reference_text"]:
            import jiwer

            lecturer_style.append(jiwer.wer(r["reference_text"], r["prediction"] or ""))
    total_s = sum(r["duration_seconds"] or 0 for r in rows)
    ok_s = sum(r["duration_seconds"] or 0 for r in ok)
    runtimes = [r["runtime_s"] for r in ok if r["runtime_s"] is not None]
    summary = {
        "normalization": NORMALIZATION_VERSION,
        "n_selected": len(rows),
        "n_success": len(ok),
        "n_failed": len(failed),
        "n_scored": len(scored),
        "n_empty_reference": len(ok) - len(scored),
        "selected_audio_hours": total_s / 3600,
        "evaluated_audio_hours": ok_s / 3600,
        **{k: v for k, v in corpus_scores(scores).items()},
        "corpus_wer_failures_as_deletions": corpus_scores(with_failures)["corpus_wer"],
        **describe([r["wer"] for r in scored], "per_recording_wer"),
        **describe([r["cer"] for r in scored], "per_recording_cer"),
        # Equivalent in content to ``results[["wer", "cer"]].describe()``
        # without making pandas a runtime requirement for the pipeline.
        "per_recording_wer_cer_describe": describe_wer_cer(rows),
        "perfect_transcription_rate": (sum(1 for r in scored if r["wer"] == 0) / len(scored)) if scored else math.nan,
        "lecturer_style_mean_wer_unnormalized": (sum(lecturer_style) / len(lecturer_style)) if lecturer_style else math.nan,
        "total_inference_runtime_s": sum(runtimes),
        "real_time_factor": (sum(runtimes) / ok_s) if ok_s else math.nan,
        "flags": {
            "hit_token_limit": sum(r["flag_hit_token_limit"] for r in rows),
            "empty_prediction": sum(r["flag_empty_prediction"] for r in rows),
            "short_vs_reference": sum(r["flag_short_vs_reference"] for r in rows),
            "wer_above_1": sum(r["flag_wer_above_1"] for r in rows),
        },
        "failure_reasons": _count([r["error_message"] for r in failed]),
    }
    if run_info:
        summary["run_info"] = run_info
    return summary


def _count(values: list) -> dict:
    out: dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return out


def _scored_values(rows: list[dict], column: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(column)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    return values


def _percentile(values: list[float], fraction: float) -> float:
    """Linear-interpolated percentile, matching the usual DataFrame default."""
    if not values:
        return math.nan
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _describe_values(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    count = len(ordered)
    if not ordered:
        return {
            "count": 0,
            "mean": math.nan,
            "std": math.nan,
            "min": math.nan,
            "25%": math.nan,
            "50%": math.nan,
            "75%": math.nan,
            "max": math.nan,
        }
    mean = sum(ordered) / count
    sample_std = math.sqrt(sum((value - mean) ** 2 for value in ordered) / (count - 1)) if count > 1 else math.nan
    return {
        "count": count,
        "mean": mean,
        "std": sample_std,
        "min": ordered[0],
        "25%": _percentile(ordered, 0.25),
        "50%": _percentile(ordered, 0.50),
        "75%": _percentile(ordered, 0.75),
        "max": ordered[-1],
    }


def describe_wer_cer(rows: list[dict]) -> dict[str, dict[str, float | int]]:
    """Return the same fields as ``results[["wer", "cer"]].describe()``."""
    by_metric = {metric: _describe_values(_scored_values(rows, metric)) for metric in ("wer", "cer")}
    return {
        statistic: {metric: by_metric[metric][statistic] for metric in ("wer", "cer")}
        for statistic in ("count", "mean", "std", "min", "25%", "50%", "75%", "max")
    }


def _wer_descending_key(row: dict) -> tuple[int, float]:
    value = row.get("wer")
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        # Keep failed/unscored rows in the output, after scored recordings.
        return (1, 0.0)
    return (0, -float(value))


def _table_value(value):
    return "" if value is None or (isinstance(value, float) and math.isnan(value)) else value


def _write_result_table(path: Path, rows: list[dict], *, delimiter: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction="ignore", delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _table_value(row.get(key))
                    for key in RESULT_COLUMNS
                }
            )


def export(rows: list[dict], summary: dict, out_dir: Path, run_name: str) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{run_name}_per_recording.csv"
    _write_result_table(csv_path, rows, delimiter=",")
    tsv_path = out_dir / f"{run_name}_results_without_context.tsv"
    _write_result_table(tsv_path, sorted(rows, key=_wer_descending_key), delimiter="\t")
    json_path = out_dir / f"{run_name}_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_nan_to_none(summary), f, indent=2)
    return {
        "per_recording_csv": csv_path,
        "per_recording_tsv": tsv_path,
        "summary_json": json_path,
    }


def _nan_to_none(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj


def lecturer_vs_project_normalization(text: str) -> tuple[str, str]:
    """Helper for the report: show how both normalisations treat one string."""
    return lecturer_normalize(text), normalize_text(text)
