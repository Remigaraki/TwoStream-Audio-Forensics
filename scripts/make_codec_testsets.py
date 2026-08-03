"""Generate codec-compressed copies of the test-set audio for robustness evaluation.

Uses src/pipeline/augment.py's transcode()/CODEC_CONDITIONS — the same codec
path A1's training actually drew from (train.py imports from this module) —
not data/torture_pipeline.py, which is disconnected from training entirely.

Resumable: skips a file if its output already exists. Safe to rerun after a
killed Kaggle session; it just picks up where it left off.

Usage — one condition (recommended: generate, predict, delete, repeat):
    python scripts/make_codec_testsets.py \\
        --manifest /kaggle/working/manifest.csv \\
        --output_base /kaggle/tmp/codec_test \\
        --conditions opus_16

Usage — split one condition across accounts:
    python scripts/make_codec_testsets.py \\
        --manifest /kaggle/working/manifest.csv \\
        --output_base /kaggle/tmp/codec_test \\
        --conditions opus_16 --shard 0/4
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.augment import transcode, CODEC_CONDITIONS


def _condition_label(codec: str, bitrate: int) -> str:
    return f"{codec}_{bitrate}"


def _parse_shard(s: str) -> tuple[int, int]:
    i_str, n_str = s.split("/")
    i, n = int(i_str), int(n_str)
    if not (0 <= i < n):
        raise ValueError(f"--shard {s}: i must satisfy 0 <= i < N")
    return i, n


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate codec-compressed test-set copies")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output_base", required=True)
    p.add_argument("--conditions", default=None,
                   help="Comma-separated condition labels (e.g. opus_16,mp3_64). "
                        "Default: all 6 conditions in CODEC_CONDITIONS.")
    p.add_argument("--shard", default=None,
                   help="i/N — process only every Nth row starting at i, "
                        "for splitting one condition across accounts.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    all_labels = {_condition_label(c, b): (c, b) for c, b in CODEC_CONDITIONS}
    if args.conditions:
        requested = [c.strip() for c in args.conditions.split(",")]
        unknown = [c for c in requested if c not in all_labels]
        if unknown:
            raise ValueError(f"Unknown condition(s) {unknown}. Valid: {sorted(all_labels)}")
        conditions = [(label, *all_labels[label]) for label in requested]
    else:
        conditions = [(label, c, b) for label, (c, b) in all_labels.items()]

    with open(args.manifest, "r", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == "test"]
    rows.sort(key=lambda r: r["utterance_id"])
    print(f"[init] {len(rows)} test-split rows loaded from manifest", flush=True)

    if args.shard:
        i, n = _parse_shard(args.shard)
        rows = [r for idx, r in enumerate(rows) if idx % n == i]
        print(f"[init] shard {i}/{n} -> {len(rows)} rows this run", flush=True)

    output_base = Path(args.output_base)
    summary = []

    for label, codec, bitrate in conditions:
        out_dir = output_base / label
        out_dir.mkdir(parents=True, exist_ok=True)

        done = skipped = 0
        failures: list[dict] = []
        total = len(rows)

        for idx, row in enumerate(rows):
            src = row["file_path"]
            dst = out_dir / Path(src).name

            if dst.exists() and dst.stat().st_size > 0:
                skipped += 1
            else:
                try:
                    samples = transcode(src, codec, bitrate)
                    sf.write(str(dst), samples, 16000)
                    done += 1
                except Exception as e:
                    failures.append({"utterance_id": row["utterance_id"], "file_path": src, "error": str(e)})

            if (idx + 1) % 200 == 0:
                print(f"  [{label}] {idx + 1}/{total} (done={done} skipped={skipped} failed={len(failures)})",
                      flush=True)

        if failures:
            fail_path = output_base / f"{label}_failures.csv"
            with open(fail_path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=["utterance_id", "file_path", "error"])
                w.writeheader()
                w.writerows(failures)
            print(f"[{label}] {len(failures)} failure(s) written to {fail_path}")

        expected = {Path(r["file_path"]).name for r in rows}
        actual = {p.name for p in out_dir.iterdir() if p.is_file()}
        missing = expected - actual
        if missing:
            print(f"[{label}] ⚠️  {len(missing)} expected file(s) missing from output dir "
                  f"(check {label}_failures.csv, then rerun this command to retry)")
        else:
            print(f"[{label}] ✅ all {len(expected)} expected file(s) present")

        print(f"[{label}] done={done} skipped={skipped} failed={len(failures)} total={total}\n", flush=True)
        summary.append((label, done, skipped, len(failures), total, len(missing)))

    print("=== summary ===")
    for label, done, skipped, n_fail, total, n_missing in summary:
        print(f"  {label:12s} done={done:6d} skipped={skipped:6d} failed={n_fail:4d} "
              f"missing={n_missing:4d} total={total}")


if __name__ == "__main__":
    main()
