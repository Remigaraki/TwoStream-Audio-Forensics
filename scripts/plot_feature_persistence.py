"""
Feature persistence analysis — EER vs bitrate plot.

Reads score CSVs produced by src/eval/score_extractor.py and plots EER (%)
for each setup across the 7 evaluation conditions (clean + 6 codec).

The x-axis orders conditions by effective bitrate (clean treated as lossless).
One line per setup (A, B, C). Output: PNG + PDF saved to --output_dir.

Usage:
    python -m scripts.plot_feature_persistence \\
        --results_dir /content/drive/MyDrive/ASVspoof5/results \\
        --output_dir  /content/drive/MyDrive/ASVspoof5/figures
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.utils.metrics import compute_eer, compute_t_dcf


# Ordered by effective bitrate (low → high), clean last as lossless baseline
CONDITION_ORDER = [
    ("opus_16k",  "Opus\n16k"),
    ("opus_32k",  "Opus\n32k"),
    ("mp3_64k",   "MP3\n64k"),
    ("opus_64k",  "Opus\n64k"),
    ("mp3_128k",  "MP3\n128k"),
    ("aac_128k",  "AAC\n128k"),
    ("clean",     "Clean\n(lossless)"),
]

SETUP_STYLES = {
    "A": {"color": "#2196F3", "marker": "o", "label": "Setup A (Stream 1 only)"},
    "B": {"color": "#FF9800", "marker": "s", "label": "Setup B (Stream 2 only)"},
    "C": {"color": "#4CAF50", "marker": "^", "label": "Setup C (Fused)"},
}


def _load_scores(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_score = [], []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            y_true.append(int(row["label"]))
            y_score.append(float(row["score"]))
    return np.array(y_true), np.array(y_score)


def collect_results(results_dir: str, setups: list[str]) -> dict:
    results_path = Path(results_dir)
    data: dict[str, dict[str, dict]] = {s: {} for s in setups}

    for setup in setups:
        for cond_key, _ in CONDITION_ORDER:
            csv_path = results_path / f"scores_{setup}_{cond_key}.csv"
            if not csv_path.exists():
                print(f"  Missing: {csv_path.name}")
                data[setup][cond_key] = {"eer": float("nan"), "tdcf": float("nan")}
                continue
            y_true, y_score = _load_scores(str(csv_path))
            try:
                eer,  _ = compute_eer(y_true, y_score)
                tdcf, _ = compute_t_dcf(y_true, y_score)
            except Exception as exc:
                print(f"  WARNING [{setup}/{cond_key}]: {exc}")
                eer, tdcf = float("nan"), float("nan")
            data[setup][cond_key] = {"eer": eer, "tdcf": tdcf}

    return data


def plot_eer(data: dict, setups: list[str], output_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(CONDITION_ORDER))
    cond_keys  = [k for k, _ in CONDITION_ORDER]
    x_labels   = [label for _, label in CONDITION_ORDER]

    fig, ax = plt.subplots(figsize=(10, 5))

    for setup in setups:
        style = SETUP_STYLES.get(setup, {"color": "gray", "marker": "x", "label": f"Setup {setup}"})
        eer_vals = [data[setup].get(cond, {}).get("eer", float("nan")) * 100 for cond in cond_keys]
        ax.plot(x, eer_vals, marker=style["marker"], color=style["color"],
                label=style["label"], linewidth=2, markersize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("EER (%)", fontsize=12)
    ax.set_xlabel("Evaluation Condition", fontsize=12)
    ax.set_title("Feature Persistence Under Lossy Compression", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(bottom=0)

    # Shade the codec region vs lossless
    ax.axvspan(len(CONDITION_ORDER) - 1.5, len(CONDITION_ORDER) - 0.5, alpha=0.06,
               color="green", label="_nolegend_")
    ax.axvspan(-0.5, len(CONDITION_ORDER) - 1.5, alpha=0.04,
               color="red", label="_nolegend_")

    fig.tight_layout()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        dest = out_path / f"feature_persistence_eer.{ext}"
        fig.savefig(dest, dpi=150)
        print(f"Saved: {dest}")

    plt.close(fig)


def plot_tdcf(data: dict, setups: list[str], output_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(CONDITION_ORDER))
    cond_keys = [k for k, _ in CONDITION_ORDER]
    x_labels  = [label for _, label in CONDITION_ORDER]

    fig, ax = plt.subplots(figsize=(10, 5))

    for setup in setups:
        style = SETUP_STYLES.get(setup, {"color": "gray", "marker": "x", "label": f"Setup {setup}"})
        tdcf_vals = [data[setup].get(cond, {}).get("tdcf", float("nan")) for cond in cond_keys]
        ax.plot(x, tdcf_vals, marker=style["marker"], color=style["color"],
                label=style["label"], linewidth=2, markersize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("min-tDCF", fontsize=12)
    ax.set_xlabel("Evaluation Condition", fontsize=12)
    ax.set_title("min-tDCF Under Lossy Compression", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(bottom=0)

    fig.tight_layout()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        dest = out_path / f"feature_persistence_tdcf.{ext}"
        fig.savefig(dest, dpi=150)
        print(f"Saved: {dest}")

    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot EER and min-tDCF vs codec condition")
    parser.add_argument("--results_dir", required=True, help="Directory containing score CSVs")
    parser.add_argument("--output_dir",  required=True, help="Directory to write PNG/PDF figures")
    parser.add_argument("--setups", nargs="+", default=["A", "B", "C"],
                        choices=["A", "B", "C"], help="Which setups to include")
    args = parser.parse_args()

    data = collect_results(args.results_dir, args.setups)
    plot_eer(data, args.setups, args.output_dir)
    plot_tdcf(data, args.setups, args.output_dir)
    print("Done.")
