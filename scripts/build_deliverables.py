#!/usr/bin/env python3
"""
build_deliverables.py -- assemble thesis_deliverables/ from results/tables/ and
results/figures/ (which scripts/build_results.py produces from results/preds_*.csv).

This is the single committed source for the three steps that used to be
one-off/ad hoc when this folder was first built by hand:
  - table_mcnemar.md reformatted for the manuscript (p-values as "< .001" / 3dp)
  - table_a1_vs_a0_chi2.csv/.md (Chapter 4, Table 4.5), built from
    table_eer.csv + table_mcnemar.csv, not hand-typed
  - degradation_curve_mindcf_log.png (no minDCF degradation figure existed
    before), styled to match degradation_curve_eer_log.png

Everything else in thesis_deliverables/ is a verbatim copy of an existing
results/tables/ file that already matches the manuscript formatting spec
(EER 2dp, minDCF 4dp, condition order clean..aac_128, model order A0..C2) or
an existing results/figures/ file (already restyled by build_results.py).
table_cllr is deliberately excluded -- Cllr is not reported in the manuscript.

Full regeneration, no manual steps:
    python scripts/build_results.py && python scripts/build_deliverables.py

Reads:  results/tables/*.csv, results/figures/*.png
Writes: thesis_deliverables/tables/*.csv, thesis_deliverables/tables/*.md,
        thesis_deliverables/figures/*.png
Never touches results/preds_*.csv, and never writes anywhere under results/.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_results import (
    MODELS,
    CONDITIONS,
    MODEL_LABELS,
    CONDITION_LABELS,
    MODEL_MARKERS,
    MODEL_LINESTYLES,
)

INV_MODEL_LABEL = {v: k for k, v in MODEL_LABELS.items()}
INV_CONDITION_LABEL = {v: k for k, v in CONDITION_LABELS.items()}

# Tables copied byte-for-byte from results/tables/ -- verified (by inspection,
# see commit history) to already match the manuscript formatting spec, so no
# reformatting step is applied.
VERBATIM_TABLES = [
    "table_eer.csv", "table_eer.md",
    "table_min_dcf.csv", "table_min_dcf.md",
    "table_degradation_ratio.csv", "table_degradation_ratio.md",
    "table_c1_margin.csv", "table_c1_margin.md",
    "metrics_long.csv",
]
# Figures copied byte-for-byte from results/figures/ -- build_results.py
# already applies the manuscript styling (per-model marker/linestyle, legend
# outside the axes, 300 DPI).
VERBATIM_FIGURES = [
    "control_a0_vs_a1.png",
    "degradation_curve_eer.png",
    "degradation_curve_eer_log.png",
]

REQUIRED_SOURCE_TABLES = VERBATIM_TABLES + ["table_mcnemar.csv"]
REQUIRED_SOURCE_FIGURES = VERBATIM_FIGURES


def fmt_p(p: float) -> str:
    """Manuscript p-value format: '< .001' below that threshold, else 3dp."""
    return "< .001" if p < 0.001 else f"{p:.3f}"


def _to_md_table(headers, rows) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in rows]
    return "\n".join([head, rule] + body)


# ---------------------------------------------------------------------
# all_reported_numbers.csv -- every EER / minDCF / discordant count /
# p-value in results/tables/, one row each, tagged with its source file.
# ---------------------------------------------------------------------
def build_all_reported_numbers(src_tables: Path, dst_path: Path) -> None:
    rows = []

    ml = pd.read_csv(src_tables / "metrics_long.csv")
    for _, r in ml.iterrows():
        rows.append({"metric": "eer", "model": r["model"], "condition": r["condition"],
                     "comparison": "", "value": r["eer"], "source_file": "results/tables/metrics_long.csv"})
        rows.append({"metric": "min_dcf", "model": r["model"], "condition": r["condition"],
                     "comparison": "", "value": r["min_dcf"], "source_file": "results/tables/metrics_long.csv"})

    eer_wide = pd.read_csv(src_tables / "table_eer.csv", index_col=0)
    for model_label, row in eer_wide.iterrows():
        model = INV_MODEL_LABEL.get(model_label, model_label)
        for cond_label, value in row.items():
            if pd.isna(value):
                continue
            cond = INV_CONDITION_LABEL.get(cond_label, cond_label)
            rows.append({"metric": "eer_pct", "model": model, "condition": cond,
                         "comparison": "", "value": value, "source_file": "results/tables/table_eer.csv"})

    mindcf_wide = pd.read_csv(src_tables / "table_min_dcf.csv", index_col=0)
    for model_label, row in mindcf_wide.iterrows():
        model = INV_MODEL_LABEL.get(model_label, model_label)
        for cond_label, value in row.items():
            if pd.isna(value):
                continue
            cond = INV_CONDITION_LABEL.get(cond_label, cond_label)
            rows.append({"metric": "min_dcf", "model": model, "condition": cond,
                         "comparison": "", "value": value, "source_file": "results/tables/table_min_dcf.csv"})

    margin = pd.read_csv(src_tables / "table_c1_margin.csv")
    for _, r in margin.iterrows():
        cond = INV_CONDITION_LABEL.get(r["condition"], r["condition"])
        rows.append({"metric": "eer_pct", "model": "A1", "condition": cond,
                     "comparison": "", "value": r["A1_eer_pct"], "source_file": "results/tables/table_c1_margin.csv"})
        rows.append({"metric": "eer_pct", "model": "C1", "condition": cond,
                     "comparison": "", "value": r["C1_eer_pct"], "source_file": "results/tables/table_c1_margin.csv"})

    mc = pd.read_csv(src_tables / "table_mcnemar.csv")
    for _, r in mc.iterrows():
        rows.append({"metric": "discordant_only_first_correct", "model": "", "condition": r["condition"],
                     "comparison": r["comparison"], "value": r["only_first_correct"],
                     "source_file": "results/tables/table_mcnemar.csv"})
        rows.append({"metric": "discordant_only_second_correct", "model": "", "condition": r["condition"],
                     "comparison": r["comparison"], "value": r["only_second_correct"],
                     "source_file": "results/tables/table_mcnemar.csv"})
        rows.append({"metric": "p_value", "model": "", "condition": r["condition"],
                     "comparison": r["comparison"], "value": r["p_value"],
                     "source_file": "results/tables/table_mcnemar.csv"})

    out = pd.DataFrame(rows, columns=["metric", "model", "condition", "comparison", "value", "source_file"])
    out = out.sort_values(["metric", "model", "condition", "comparison"]).reset_index(drop=True)
    out.to_csv(dst_path, index=False)


# ---------------------------------------------------------------------
# table_mcnemar.md -- reformatted for the manuscript (p as "< .001" / 3dp,
# chi2 to 2dp, condition-ordered). table_mcnemar.csv is copied verbatim
# (full precision) alongside it.
# ---------------------------------------------------------------------
def build_mcnemar_md(mc: pd.DataFrame, dst_path: Path) -> None:
    mc = mc.copy()
    mc["_cond_order"] = mc["condition"].map({c: i for i, c in enumerate(CONDITIONS)})
    mc = mc.sort_values(["_cond_order", "comparison"]).drop(columns="_cond_order")

    rows = [
        [r["condition"], r["comparison"], int(r["only_first_correct"]), int(r["only_second_correct"]),
         f"{r['chi2']:.2f}", fmt_p(r["p_value"]), r["significant"]]
        for _, r in mc.iterrows()
    ]
    md = _to_md_table(
        ["condition", "comparison", "only_first_correct", "only_second_correct", "chi2", "p_value", "significant"],
        rows,
    )
    dst_path.write_text("# McNemar paired significance tests\n\n" + md + "\n", encoding="utf-8")


# ---------------------------------------------------------------------
# table_a1_vs_a0_chi2.csv/.md -- Chapter 4, Table 4.5. Built from
# table_eer.csv + table_mcnemar.csv, not hand-typed.
# ---------------------------------------------------------------------
def build_a1_vs_a0_table(src_tables: Path, mc: pd.DataFrame, dst_dir: Path) -> None:
    eer = pd.read_csv(src_tables / "table_eer.csv", index_col=0)
    eer.index = [INV_MODEL_LABEL[i] for i in eer.index]
    eer.columns = [INV_CONDITION_LABEL[c] for c in eer.columns]

    a1_vs_a0 = mc[mc["comparison"] == "A1 vs A0"].set_index("condition")

    out_rows = []
    for cond in CONDITIONS:
        a0_eer = float(eer.loc["A0", cond])
        a1_eer = float(eer.loc["A1", cond])
        rel_reduction = (1 - a1_eer / a0_eer) * 100
        mrow = a1_vs_a0.loc[cond]
        out_rows.append({
            "condition": cond,
            "a0_eer_pct": a0_eer,
            "a1_eer_pct": a1_eer,
            "relative_reduction_pct": rel_reduction,
            "n01_a1_correct_a0_wrong": int(mrow["only_first_correct"]),
            "n10_a1_wrong_a0_correct": int(mrow["only_second_correct"]),
            "chi2": float(mrow["chi2"]),
            "p_value": float(mrow["p_value"]),
        })

    df = pd.DataFrame(out_rows)
    df.to_csv(dst_dir / "table_a1_vs_a0_chi2.csv", index=False)

    md_rows = [
        [
            CONDITION_LABELS[r["condition"]],
            f"{r['a0_eer_pct']:.2f}", f"{r['a1_eer_pct']:.2f}", f"{r['relative_reduction_pct']:.2f}",
            r["n01_a1_correct_a0_wrong"], r["n10_a1_wrong_a0_correct"],
            f"{r['chi2']:.2f}", fmt_p(r["p_value"]),
        ]
        for r in out_rows
    ]
    md = _to_md_table(
        ["Condition", "A0 EER (%)", "A1 EER (%)", "Relative reduction (%)",
         "n01 (A1 right, A0 wrong)", "n10 (A1 wrong, A0 right)", "chi2", "p"],
        md_rows,
    )
    (dst_dir / "table_a1_vs_a0_chi2.md").write_text(
        "# A1 vs A0: effect of codec augmentation, with McNemar significance\n\n" + md + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# degradation_curve_mindcf_log.png -- styled to match
# degradation_curve_eer_log.png (same marker/linestyle/legend conventions).
# ---------------------------------------------------------------------
def build_mindcf_figure(src_tables: Path, dst_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = pd.read_csv(src_tables / "metrics_long.csv")
    mindcf_wide = metrics.pivot(index="model", columns="condition", values="min_dcf").reindex(
        index=MODELS, columns=CONDITIONS
    )
    present = [c for c in CONDITIONS if c in mindcf_wide.columns and mindcf_wide[c].notna().any()]

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for m in MODELS:
        if m not in mindcf_wide.index:
            continue
        vals = [mindcf_wide.loc[m, c] if pd.notna(mindcf_wide.loc[m, c]) else np.nan for c in present]
        ax.plot(
            range(len(present)), vals,
            marker=MODEL_MARKERS[m], linestyle=MODEL_LINESTYLES[m],
            markersize=7, linewidth=1.8,
            markeredgecolor="white", markeredgewidth=0.8,
            label=MODEL_LABELS[m],
        )
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([CONDITION_LABELS[c] for c in present], rotation=20, ha="right")
    ax.set_xlabel("Condition")
    ax.set_title("Detection cost (minDCF) across codec conditions")
    ax.grid(alpha=0.3)
    ax.set_yscale("log")
    ax.set_ylabel("minDCF (log scale)")
    # Deviates from plain legend()/"best": on this data range "best" puts the
    # legend box on top of the A0/B0 curves in the MP3 region. Outside the
    # axes instead -- same convention build_results.py now uses everywhere.
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(dst_dir / "degradation_curve_mindcf_log.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="thesis_deliverables")
    args = ap.parse_args()

    src_tables = Path(args.results_dir) / "tables"
    src_figures = Path(args.results_dir) / "figures"
    dst_tables = Path(args.out_dir) / "tables"
    dst_figures = Path(args.out_dir) / "figures"
    dst_tables.mkdir(parents=True, exist_ok=True)
    dst_figures.mkdir(parents=True, exist_ok=True)

    missing_tables = [f for f in REQUIRED_SOURCE_TABLES if not (src_tables / f).exists()]
    missing_figures = [f for f in REQUIRED_SOURCE_FIGURES if not (src_figures / f).exists()]
    if missing_tables or missing_figures:
        raise SystemExit(
            f"missing required results/ files -- run scripts/build_results.py first.\n"
            f"  missing tables:  {missing_tables}\n"
            f"  missing figures: {missing_figures}"
        )

    for name in VERBATIM_TABLES:
        shutil.copy2(src_tables / name, dst_tables / name)
    for name in VERBATIM_FIGURES:
        shutil.copy2(src_figures / name, dst_figures / name)
    print(f"copied {len(VERBATIM_TABLES)} verbatim tables, {len(VERBATIM_FIGURES)} verbatim figures")

    build_all_reported_numbers(src_tables, dst_tables / "all_reported_numbers.csv")
    print("wrote all_reported_numbers.csv")

    mc = pd.read_csv(src_tables / "table_mcnemar.csv")
    shutil.copy2(src_tables / "table_mcnemar.csv", dst_tables / "table_mcnemar.csv")
    build_mcnemar_md(mc, dst_tables / "table_mcnemar.md")
    print("wrote table_mcnemar.csv (verbatim) / table_mcnemar.md (reformatted)")

    build_a1_vs_a0_table(src_tables, mc, dst_tables)
    print("wrote table_a1_vs_a0_chi2.csv / .md")

    build_mindcf_figure(src_tables, dst_figures)
    print("wrote degradation_curve_mindcf_log.png")

    n_tables = len(list(dst_tables.iterdir()))
    n_figures = len(list(dst_figures.iterdir()))
    print(f"\nthesis_deliverables/tables/  -> {n_tables} files")
    print(f"thesis_deliverables/figures/ -> {n_figures} files")
    print(f"total: {n_tables + n_figures} files")


if __name__ == "__main__":
    main()
