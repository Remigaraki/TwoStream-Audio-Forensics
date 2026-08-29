#!/usr/bin/env python3
"""
build_results.py -- assemble all thesis result tables and figures from prediction CSVs.

Reads:  results/preds_<MODEL>_<CONDITION>.csv  (cols: utterance_id,true_label,score,pred)
Writes: results/tables/*.csv, results/tables/*.md, results/figures/*.png

No GPU, no audio, no model loading. Pure post-processing.

Usage:
    python scripts/build_results.py
    python scripts/build_results.py --results-dir results --out-dir results
"""
import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# McNemar's test itself lives in scripts/mcnemar_test.py, which is canonical
# (it produced the run logs referenced in the thesis notes). Import it rather
# than reimplementing, so this script can never silently drift from it again.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcnemar_test import mcnemar_test as _mcnemar_stat, _binarise_at_eer

# ---------------------------------------------------------------- config

MODELS = ["A0", "A1", "B0", "C1", "C2"]
CONDITIONS = ["clean", "opus_16", "opus_32", "opus_64", "mp3_64", "mp3_128", "aac_128"]

# Pretty names for the thesis tables
MODEL_LABELS = {
    "A0": "A0 (RawNet2, no aug.)",
    "A1": "A1 (RawNet2 + codec aug.)",
    "B0": "B0 (statistical stream)",
    "C1": "C1 (fusion, attention)",
    "C2": "C2 (fusion, concat)",
}
CONDITION_LABELS = {
    "clean": "Clean",
    "opus_16": "Opus 16k",
    "opus_32": "Opus 32k",
    "opus_64": "Opus 64k",
    "mp3_64": "MP3 64k",
    "mp3_128": "MP3 128k",
    "aac_128": "AAC 128k",
}

# ASVspoof 5 Track 1 minDCF parameters (official evaluation plan).
# VERIFY these against the eval plan appendix before submission.
PI_SPOOF = 0.05
C_MISS = 1.0
C_FA = 10.0

MCNEMAR_PAIRS = [("C1", "A1"), ("C1", "B0"), ("C2", "C1"), ("A1", "A0")]

# ---------------------------------------------------------------- metrics


def _rates(scores_spoof, scores_bona):
    """Sweep every threshold; return (P_miss, P_fa) arrays.

    Convention here: `score` is P(spoof), higher = more spoof-like.
      - miss  = a bonafide utterance flagged as spoof
      - false alarm = a spoof utterance accepted as bonafide
    """
    all_scores = np.concatenate([scores_spoof, scores_bona])
    order = np.argsort(all_scores, kind="mergesort")
    labels = np.concatenate([np.ones_like(scores_spoof), np.zeros_like(scores_bona)])
    labels = labels[order]

    # threshold sweeps upward: accept-as-spoof if score >= tau
    tar_trial_sums = np.cumsum(labels)
    nontarget_sums = np.cumsum(1 - labels)

    n_spoof = len(scores_spoof)
    n_bona = len(scores_bona)

    # P_fa: spoof below threshold (accepted as bonafide)
    p_fa = np.concatenate([[1.0], tar_trial_sums / n_spoof])
    # P_miss: bonafide at/above threshold (rejected as bonafide)
    p_miss = np.concatenate([[0.0], 1 - nontarget_sums / n_bona])
    return p_miss, p_fa


def compute_eer(scores, labels):
    """EER. labels: 1=spoof, 0=bonafide. scores: higher = more spoof-like."""
    p_miss, p_fa = _rates(scores[labels == 1], scores[labels == 0])
    idx = np.nanargmin(np.abs(p_miss - p_fa))
    return float((p_miss[idx] + p_fa[idx]) / 2)


def compute_min_dcf(scores, labels, pi_spoof=PI_SPOOF, c_miss=C_MISS, c_fa=C_FA):
    """Normalised minimum detection cost function.

    cost(tau)  = c_miss * pi_bona * P_miss(tau) + c_fa * pi_spoof * P_fa(tau)
    normaliser = min(c_miss * pi_bona, c_fa * pi_spoof)

    With the ASVspoof 5 Track 1 defaults (pi_spoof=0.05, c_miss=1, c_fa=10):
      c_miss * pi_bona = 0.95, c_fa * pi_spoof = 0.50 -> normaliser = 0.50
      normalised cost  = 1.9 * P_miss + P_fa
    """
    pi_bona = 1.0 - pi_spoof
    p_miss, p_fa = _rates(scores[labels == 1], scores[labels == 0])
    cost = c_miss * pi_bona * p_miss + c_fa * pi_spoof * p_fa
    normaliser = min(c_miss * pi_bona, c_fa * pi_spoof)
    return float(np.min(cost) / normaliser)


def compute_cllr(scores, labels, eps=1e-12):
    """Cost of log-likelihood ratios. Treats `score` as P(spoof) directly.

    NOTE: sigmoid outputs are posteriors, not calibrated LLRs. Cllr > 1 means
    the scores are worse than an uninformative system -- report it as evidence
    of miscalibration, do not present it as a discrimination result.
    """
    p = np.clip(scores, eps, 1 - eps)
    llr = np.log(p / (1 - p))
    spoof_term = np.mean(np.log2(1 + np.exp(-llr[labels == 1])))
    bona_term = np.mean(np.log2(1 + np.exp(llr[labels == 0])))
    return float(0.5 * (spoof_term + bona_term))


def mcnemar(score_a, score_b, truth):
    """Paired McNemar's test -- thin wrapper around scripts.mcnemar_test's
    canonical implementation. Returns (n01, n10, chi2, p).

    Predictions are binarised at each side's own EER threshold from `score`
    (via mcnemar_test._binarise_at_eer), NOT read from the committed `pred`
    column: that column and a threshold recomputed from the score column can
    disagree on rare boundary rows (see scripts/mcnemar_test.py docstring /
    thesis notes on the McNemar discrepancy), and mcnemar_test.py's numbers
    are the canonical ones. The statistic/p-value formula itself (Yates
    continuity correction, chi2 vs. exact-binomial branch for small n) is
    untouched here -- it's whatever _mcnemar_stat (imported) does.
    """
    pred_a = _binarise_at_eer(truth, score_a)
    pred_b = _binarise_at_eer(truth, score_b)
    a_ok = pred_a == truth
    b_ok = pred_b == truth
    n01 = int(np.sum(a_ok & ~b_ok))  # A right, B wrong
    n10 = int(np.sum(~a_ok & b_ok))  # A wrong, B right
    chi2, p = _mcnemar_stat(truth, pred_a, pred_b)
    return n01, n10, chi2, p


# ---------------------------------------------------------------- io


def load(results_dir, model, condition):
    path = os.path.join(results_dir, f"preds_{model}_{condition}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # 'pred' is required for schema/provenance even though McNemar below now
    # recomputes its own predictions from 'score' rather than reading it.
    need = {"utterance_id", "true_label", "score", "pred"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df.sort_values("utterance_id").reset_index(drop=True)


def to_markdown(df, floatfmt="{:.4f}"):
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda v: "--" if pd.isna(v) else floatfmt.format(v))
    out = out.fillna("--")
    head = "| " + " | ".join(str(c) for c in out.columns) + " |"
    rule = "|" + "|".join("---" for _ in out.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in out.values]
    return "\n".join([head, rule] + rows)


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    tdir = os.path.join(args.out_dir, "tables")
    fdir = os.path.join(args.out_dir, "figures")
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)

    # ---- 1. per-condition metrics ------------------------------------
    rows = []
    missing = []
    for model in MODELS:
        for cond in CONDITIONS:
            df = load(args.results_dir, model, cond)
            if df is None:
                missing.append(f"{model}/{cond}")
                continue
            s = df["score"].to_numpy(float)
            y = df["true_label"].to_numpy(int)
            rows.append(
                {
                    "model": model,
                    "condition": cond,
                    "n": len(df),
                    "eer": compute_eer(s, y),
                    "min_dcf": compute_min_dcf(s, y),
                    "cllr": compute_cllr(s, y),
                }
            )
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise SystemExit("No prediction CSVs found -- check --results-dir")
    metrics.to_csv(os.path.join(tdir, "metrics_long.csv"), index=False)

    print(f"\nloaded {len(metrics)} of {len(MODELS)*len(CONDITIONS)} model/condition runs")
    if missing:
        print("MISSING:", ", ".join(missing))

    # ---- 2. wide tables, one per metric ------------------------------
    for metric, scale, fmt in [
        ("eer", 100.0, "{:.2f}"),
        ("min_dcf", 1.0, "{:.4f}"),
        ("cllr", 1.0, "{:.3f}"),
    ]:
        wide = metrics.pivot(index="model", columns="condition", values=metric)
        wide = wide.reindex(index=MODELS, columns=CONDITIONS) * scale
        wide.index = [MODEL_LABELS[m] for m in wide.index]
        wide.columns = [CONDITION_LABELS[c] for c in wide.columns]
        wide.to_csv(os.path.join(tdir, f"table_{metric}.csv"))
        md = to_markdown(wide.reset_index().rename(columns={"index": "Model"}), fmt)
        with open(os.path.join(tdir, f"table_{metric}.md"), "w") as fh:
            unit = " (%)" if metric == "eer" else ""
            fh.write(f"# {metric.upper()}{unit} by model and condition\n\n{md}\n")
        print(f"  wrote table_{metric}.csv / .md")

    # ---- 3. degradation relative to clean ----------------------------
    eer_wide = metrics.pivot(index="model", columns="condition", values="eer").reindex(
        index=MODELS, columns=CONDITIONS
    )
    if "clean" in eer_wide.columns and eer_wide["clean"].notna().any():
        deg = eer_wide.div(eer_wide["clean"], axis=0)
        deg.index = [MODEL_LABELS[m] for m in deg.index]
        deg.columns = [CONDITION_LABELS[c] for c in deg.columns]
        deg.to_csv(os.path.join(tdir, "table_degradation_ratio.csv"))
        with open(os.path.join(tdir, "table_degradation_ratio.md"), "w") as fh:
            fh.write(
                "# EER degradation ratio (condition EER / clean EER)\n\n"
                + to_markdown(deg.reset_index().rename(columns={"index": "Model"}), "{:.2f}")
                + "\n"
            )
        print("  wrote table_degradation_ratio.csv / .md")

    # ---- 4. C1 margin over A1 ----------------------------------------
    if {"C1", "A1"}.issubset(set(metrics["model"])):
        piv = metrics.pivot(index="condition", columns="model", values="eer")
        piv = piv.reindex([c for c in CONDITIONS if c in piv.index])
        if {"C1", "A1"}.issubset(piv.columns):
            margin = pd.DataFrame(
                {
                    "condition": [CONDITION_LABELS[c] for c in piv.index],
                    "A1_eer_pct": piv["A1"] * 100,
                    "C1_eer_pct": piv["C1"] * 100,
                    "margin_pts": (piv["A1"] - piv["C1"]) * 100,
                    "relative_reduction_pct": (1 - piv["C1"] / piv["A1"]) * 100,
                }
            ).reset_index(drop=True)
            margin.to_csv(os.path.join(tdir, "table_c1_margin.csv"), index=False)
            with open(os.path.join(tdir, "table_c1_margin.md"), "w") as fh:
                fh.write("# C1 advantage over A1 by condition\n\n" + to_markdown(margin, "{:.2f}") + "\n")
            print("  wrote table_c1_margin.csv / .md")

    # ---- 5. McNemar across every condition ---------------------------
    mrows = []
    for cond in CONDITIONS:
        loaded = {m: load(args.results_dir, m, cond) for m in MODELS}
        for a, b in MCNEMAR_PAIRS:
            da, db = loaded.get(a), loaded.get(b)
            if da is None or db is None:
                continue
            if not da["utterance_id"].equals(db["utterance_id"]):
                raise ValueError(f"{a} vs {b} @ {cond}: utterance_ids differ -- cannot pair")
            truth = da["true_label"].to_numpy(int)
            n01, n10, chi2, p = mcnemar(
                da["score"].to_numpy(float), db["score"].to_numpy(float), truth
            )
            mrows.append(
                {
                    "condition": cond,
                    "comparison": f"{a} vs {b}",
                    f"only_first_correct": n01,
                    f"only_second_correct": n10,
                    "chi2": chi2,
                    "p_value": p,
                    "significant": "yes" if p < 0.05 else "NO",
                }
            )
    if mrows:
        mc = pd.DataFrame(mrows)
        mc.to_csv(os.path.join(tdir, "table_mcnemar.csv"), index=False)
        with open(os.path.join(tdir, "table_mcnemar.md"), "w") as fh:
            fh.write("# McNemar paired significance tests\n\n" + to_markdown(mc, "{:.4f}") + "\n")
        print(f"  wrote table_mcnemar.csv / .md ({len(mc)} of {len(CONDITIONS)*len(MCNEMAR_PAIRS)} tests)")
        ns = mc[mc["significant"] == "NO"]
        if len(ns):
            print("\n  NON-SIGNIFICANT results (report these honestly):")
            for _, r in ns.iterrows():
                print(f"    {r['comparison']:12s} @ {r['condition']:9s} p={r['p_value']:.4f}")

    # ---- 6. figures --------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = [c for c in CONDITIONS if c in eer_wide.columns and eer_wide[c].notna().any()]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in MODELS:
        if m not in eer_wide.index:
            continue
        vals = [eer_wide.loc[m, c] * 100 if pd.notna(eer_wide.loc[m, c]) else np.nan for c in present]
        ax.plot(range(len(present)), vals, marker="o", label=MODEL_LABELS[m])
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([CONDITION_LABELS[c] for c in present], rotation=20, ha="right")
    ax.set_ylabel("EER (%)")
    ax.set_xlabel("Condition")
    ax.set_title("Detection performance across codec conditions")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, "degradation_curve_eer.png"), dpi=200)

    ax.set_yscale("log")
    ax.set_ylabel("EER (%, log scale)")
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, "degradation_curve_eer_log.png"), dpi=200)
    plt.close(fig)

    # A0 vs A1: the codec-augmentation control
    if {"A0", "A1"}.issubset(set(eer_wide.index)):
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(present))
        w = 0.38
        for i, m in enumerate(["A0", "A1"]):
            vals = [eer_wide.loc[m, c] * 100 if pd.notna(eer_wide.loc[m, c]) else 0 for c in present]
            ax.bar(x + (i - 0.5) * w, vals, w, label=MODEL_LABELS[m])
        ax.set_xticks(x)
        ax.set_xticklabels([CONDITION_LABELS[c] for c in present], rotation=20, ha="right")
        ax.set_ylabel("EER (%)")
        ax.set_title("Effect of codec augmentation (A0 vs A1)")
        ax.grid(alpha=0.3, axis="y")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(fdir, "control_a0_vs_a1.png"), dpi=200)
        plt.close(fig)

    print(f"\nfigures -> {fdir}")
    print(f"tables  -> {tdir}")
    print("\nCross-check before trusting: compare table_eer.md against the EERs printed")
    print("by predict_testset.py, and table_mcnemar.md against scripts/mcnemar_test.py.")


if __name__ == "__main__":
    main()
