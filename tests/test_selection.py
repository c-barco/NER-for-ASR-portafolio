import pytest

from ner_asr.records import AudioRecord
from ner_asr.selection import (
    InsufficientAudioError,
    check_disjoint,
    read_manifest,
    select_by_duration,
    write_manifest,
)


def recs(n=20, dur=10.0, split="test"):
    return [
        AudioRecord(f"id{i:03d}", f"/x/{i}.wav", f"text {i}", dur, f"s{i % 4}", split, "fx")
        for i in range(n)
    ]


def test_reaches_target_and_is_deterministic():
    a, rep = select_by_duration(recs(), target_seconds=55, seed=1)
    b, _ = select_by_duration(list(reversed(recs())), target_seconds=55, seed=1)
    assert [r.sample_id for r in a] == [r.sample_id for r in b]  # input order irrelevant
    assert rep.selected_seconds >= 55 and len(a) == 6 and rep.target_reached
    c, _ = select_by_duration(recs(), target_seconds=55, seed=2)
    assert [r.sample_id for r in a] != [r.sample_id for r in c]


def test_insufficient_audio_fails_loudly():
    with pytest.raises(InsufficientAudioError):
        select_by_duration(recs(n=3), target_seconds=3600, seed=0)
    sel, rep = select_by_duration(recs(n=3), target_seconds=3600, seed=0, allow_insufficient=True)
    assert len(sel) == 3 and not rep.target_reached


def test_filters_duplicates_and_bad_items():
    items = recs(n=5)
    items.append(AudioRecord("id000", "/dup.wav", "dup", 10.0, None, "test"))
    items.append(AudioRecord("short", "/s.wav", "t", 0.1, None, "test"))
    items.append(AudioRecord("long", "/l.wav", "t", 99.0, None, "test"))
    items.append(AudioRecord("empty", "/e.wav", "  ", 5.0, None, "test"))
    sel, rep = select_by_duration(items, target_seconds=None, seed=0, min_duration_s=0.5, max_duration_s=30)
    assert rep.duplicate_ids == 1 and rep.excluded_too_short == 1
    assert rep.excluded_too_long == 1 and rep.excluded_empty_text == 1
    assert len(sel) == 5


def test_refuses_to_mix_splits():
    with pytest.raises(ValueError, match="several splits"):
        select_by_duration(recs(3, split="test") + recs(1, split="train"), target_seconds=None, seed=0)


def test_pilot_max_items_and_scan_limit():
    sel, rep = select_by_duration(recs(), target_seconds=None, seed=0, max_items=2, max_scan_items=8)
    assert len(sel) == 2 and rep.scan_limited and rep.candidates_scanned == 8


def test_manifest_roundtrip(tmp_path):
    sel, rep = select_by_duration(recs(4), target_seconds=None, seed=0)
    sel[0].speaker_id = None  # optional metadata may be missing
    p = tmp_path / "m.csv"
    write_manifest(sel, p, rep)
    back = read_manifest(p)
    assert [r.sample_id for r in back] == [r.sample_id for r in sel]
    assert back[0].speaker_id is None and back[1].duration_seconds == 10.0
    assert p.with_suffix(".selection.json").exists()


def test_disjoint_check():
    out = check_disjoint(recs(4), recs(6), by_speaker=True)
    assert out["overlapping_ids"] and out["overlapping_speakers"]
