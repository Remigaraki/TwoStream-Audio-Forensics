"""
McNemar's test for pairwise comparison of setup performance.

Compares Setup A vs C and Setup B vs C on each evaluation condition.
Reads score CSVs produced by src/eval/score_extractor.py and binarises
predictions at each setup's EER threshold before running the test.

McNemar's test checks whether two classifiers make significantly different
errors on the same samples. A p-value < 0.05 indicates a significant
difference in error patterns.

Usage:
    python -m scripts.mcnemar_test \\
        --results_dir /content/drive/MyDrive/ASVspoof5/results \\
        --output_csv  /content/drive/MyDrive/ASVspoof5/results/mcnemar_results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.metrics import compute_eer


# ---------------------------------------------------------------------------
# McNemar's test (with continuity correction for small n)
# ---------------------------------------------------------------------------

def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> tuple[float, float]:
    """
    McNemar's test comparing two binary classifiers on the same samples.

    Returns (chi2_statistic, p_value). Uses the mid-p corrected version
    which is more appropriate for small discordant-pair counts.
    """
    # Discordant pairs: A right / B wrong, and A wrong / B right
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    b = int(np.sum(correct_a & ~correct_b))   # A correct, B wrong
    c = int(np.sum(~correct_a & correct_b))   # A wrong, B correct

    n_discordant = b + c
    if n_discordant == 0:
        return 0.0, 1.0  # identical error patterns

    # Use exact binomial mid-p for small counts, chi2 approximation otherwise
    if n_discordant < 25:
        from scipy.stats import binom
        p_value = 2 * min(
            binom.cdf(min(b, c), n_discordant, 0.5),
            1 - binom.cdf(min(b, c) - 1, n_discordant, 0.5),
        )
        stat = float((abs(b - c) - 1) ** 2 / n_discordant) if n_discordant > 0 else 0.0
    else:
        stat = float((abs(b - c) - 1) ** 2 / n_discordant)
        p_value = float(1 - chi2.cdf(stat, df=1))

    return stat, p_value


# ---------------------------------------------------------------------------
# CSV loading helpers
# ---------------------------------------------------------------------------

def _load_scores(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_score) arrays from a score CSV.

    Legacy loader for the results_dir/scores_{setup}_{condition}.csv batch
    mode below — kept for backward compatibility with that naming scheme.
    """
    y_true, y_score = [], []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            y_true.append(int(row["label"]))
            y_score.append(float(row["score"]))
    return np.array(y_true), np.array(y_score)


def _load_predictions(csv_path: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (utterance_ids, y_true, y_score) from a predict_testset.py output CSV
    (columns: utterance_id, true_label, score, pred)."""
    uids, y_true, y_score = [], [], []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            uids.append(row["utterance_id"])
            y_true.append(int(row["true_label"]))
            y_score.append(float(row["score"]))
    return uids, np.array(y_true), np.array(y_score)


def _binarise_at_eer(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    _, threshold = compute_eer(y_true, y_score)
    return (y_score >= threshold).astype(int)


def _contingency_table(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Full 2x2 McNemar contingency table (both-correct/both-wrong included,
    not just the discordant pairs mcnemar_test() uses for the statistic)."""
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    return {
        "both_correct":   int(np.sum(correct_a & correct_b)),
        "a_correct_only": int(np.sum(correct_a & ~correct_b)),
        "b_correct_only": int(np.sum(~correct_a & correct_b)),
        "both_wrong":     int(np.sum(~correct_a & ~correct_b)),
    }


# ---------------------------------------------------------------------------
# Pairwise comparison of two predict_testset.py output CSVs
# ---------------------------------------------------------------------------

def compare_pair(csv_a: str, csv_b: str, name_a: str, name_b: str) -> dict:
    uids_a, y_true_a, score_a = _load_predictions(csv_a)
    uids_b, y_true_b, score_b = _load_predictions(csv_b)

    if uids_a != uids_b:
        raise ValueError(
            f"utterance_id order/content differs between {csv_a} and {csv_b} — "
            f"McNemar pairing requires identical row alignment"
        )
    if not np.array_equal(y_true_a, y_true_b):
        raise ValueError(f"true_label values differ between {csv_a} and {csv_b} for the same utterance_ids")

    y_true = y_true_a
    pred_a = _binarise_at_eer(y_true, score_a)
    pred_b = _binarise_at_eer(y_true, score_b)

    table = _contingency_table(y_true, pred_a, pred_b)
    stat, p_value = mcnemar_test(y_true, pred_a, pred_b)
    significant = p_value < 0.05

    print(f"\n=== {name_a} vs {name_b} (n={len(y_true)}) ===")
    print(f"                    {name_b} correct   {name_b} wrong")
    print(f"  {name_a} correct        {table['both_correct']:>6}          {table['a_correct_only']:>6}")
    print(f"  {name_a} wrong          {table['b_correct_only']:>6}          {table['both_wrong']:>6}")
    print(f"  chi2={stat:.4f}  p={p_value:.4f}  "
          f"{'SIGNIFICANT (p<0.05)' if significant else 'not significant (p>=0.05)'}")

    return {
        "comparison": f"{name_a}_vs_{name_b}",
        "n": len(y_true),
        **table,
        "chi2": stat,
        "p_value": p_value,
        "significant": significant,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_mcnemar(results_dir: str, output_csv: str) -> None:
    results_path = Path(results_dir)

    CONDITIONS = [
        "clean",
        "opus_16k", "opus_32k", "opus_64k",
        "mp3_64k",  "mp3_128k",
        "aac_128k",
    ]
    PAIRS = [("A", "C"), ("B", "C")]

    rows: list[dict] = []

    for condition in CONDITIONS:
        # Load scores for all three setups
        scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for setup in ("A", "B", "C"):
            csv_path = results_path / f"scores_{setup}_{condition}.csv"
            if not csv_path.exists():
                print(f"  Missing: {csv_path.name} — skipping condition '{condition}'")
                break
            scores[setup] = _load_scores(str(csv_path))
        else:
            y_true = scores["C"][0]  # ground truth is identical across setups

            for setup_a, setup_b in PAIRS:
                pred_a = _binarise_at_eer(y_true, scores[setup_a][1])
                pred_b = _binarise_at_eer(y_true, scores[setup_b][1])
                stat, p_value = mcnemar_test(y_true, pred_a, pred_b)
                sig = "yes" if p_value < 0.05 else "no"
                rows.append({
                    "condition":   condition,
                    "comparison":  f"{setup_a}_vs_{setup_b}",
                    "chi2":        f"{stat:.4f}",
                    "p_value":     f"{p_value:.4f}",
                    "significant": sig,
                })
                print(f"  [{condition}] Setup {setup_a} vs {setup_b}: "
                      f"chi2={stat:.4f}  p={p_value:.4f}  significant={sig}")

    # Write CSV
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["condition", "comparison", "chi2", "p_value", "significant"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nMcNemar results written to {out_path}  ({len(rows)} comparisons)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="McNemar's test: pairwise setup comparison")
    parser.add_argument("--csv_a", default=None, help="First predict_testset.py output CSV (pairwise mode)")
    parser.add_argument("--csv_b", default=None, help="Second predict_testset.py output CSV (pairwise mode)")
    parser.add_argument("--name_a", default=None, help="Display name for --csv_a")
    parser.add_argument("--name_b", default=None, help="Display name for --csv_b")
    parser.add_argument("--results_dir", default=None, help="Directory containing scores_{setup}_{condition}.csv files (legacy batch mode)")
    parser.add_argument("--output_csv",  default=None, help="Destination CSV for batch-mode results")
    args = parser.parse_args()

    if args.csv_a and args.csv_b:
        compare_pair(args.csv_a, args.csv_b, args.name_a or "A", args.name_b or "B")
    elif args.results_dir and args.output_csv:
        run_mcnemar(args.results_dir, args.output_csv)
    else:
        parser.error("either --csv_a/--csv_b (pairwise mode) or --results_dir/--output_csv (batch mode) is required")
