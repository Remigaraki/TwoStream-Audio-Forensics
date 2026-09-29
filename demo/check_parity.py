"""Score parity: demo path vs. the committed evaluation CSVs, on the SAME audio.

For N test utterances, runs the exact demo path (bounded subprocess decode ->
preprocess -> hash-verified model on CPU) and compares against the `score`
column of results/preds_<MODEL>_<condition>.csv produced by predict_testset.py.
Only `score` is read; `pred` and test thresholds are not used for anything.

Tolerance: |demo - csv| <= 1e-4. The CSV stores 6 decimals and was produced on
a GPU in batches of 32; BatchNorm is in eval mode, so batching does not change
per-utterance outputs, leaving only float32 CPU/GPU kernel differences.

    python demo/check_parity.py --model C1 --manifest data/manifest.csv \
        --data_root /path/to/flac_T_and_D --limit 200
"""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo import inference as inf  # noqa: E402

TOL = 1e-4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="C1", choices=sorted(inf.MODELS))
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data_root", default=None, help="As in predict_testset.py: data_root/<basename>.")
    ap.add_argument("--condition", default="clean", help="Only 'clean' matches untranscoded audio.")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    ref_csv = ROOT / "results" / f"preds_{args.model}_{args.condition}.csv"
    with open(ref_csv, encoding="utf-8") as fh:
        ref = {r["utterance_id"]: float(r["score"]) for r in csv.DictReader(fh)}
    with open(args.manifest, encoding="utf-8") as fh:
        rows = sorted((r for r in csv.DictReader(fh) if r["split"] == "test"), key=lambda r: r["utterance_id"])

    model = inf.load_model(args.model)
    worst, n, fails = 0.0, 0, 0
    for r in rows[: args.limit]:
        path = Path(r.get("file_path") or r["path"])
        if args.data_root:
            path = Path(args.data_root) / path.name
        with tempfile.TemporaryDirectory() as tmp:
            s, _ = inf.score(model, inf.prepare(path, Path(tmp)).x)
        d = abs(s - ref[r["utterance_id"]])
        worst, n, fails = max(worst, d), n + 1, fails + (d > TOL)
        if d > TOL:
            print(f"  MISMATCH {r['utterance_id']}: demo={s:.6f} csv={ref[r['utterance_id']]:.6f}")
    print(f"[parity] {args.model}/{args.condition}: {n} clips, max |diff|={worst:.2e}, "
          f"{fails} over tol {TOL:g} -> {'PASS' if n and not fails else 'FAIL'}")
    sys.exit(1 if fails or not n else 0)


if __name__ == "__main__":
    main()
