"""
Score extractor — loads a checkpoint and writes a score CSV for downstream evaluation.

Output CSV columns
------------------
    file_id : filename stem from the protocol
    score   : P(spoof) — softmax probability for the spoof class
    label   : ground-truth label (0=bonafide, 1=spoof)

Usage
-----
    python -m src.eval.score_extractor \\
        --setup C \\
        --checkpoint checkpoints/best_setup_C.pt \\
        --data_dir   /content/drive/MyDrive/data/ASVspoof5 \\
        --protocol   protocols/eval.txt \\
        --pca_path   checkpoints/pca.pkl \\
        --output_csv results/scores_setup_C_clean.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from data.dataset_loader import ASVspoofDataset
from src.stream2.extract_bispectrum import Stream2ForensicFeaturePipeline
from src.train import build_model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_scores(
    setup: str,
    checkpoint_path: str,
    data_dir: str,
    protocol_path: str,
    pca_path: str,
    output_csv: str,
    batch_size: int = 32,
    num_workers: int = 2,
    device: torch.device | None = None,
) -> None:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Model ---
    model = build_model(setup)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(val_eer={ckpt.get('metrics', {}).get('val_eer', float('nan')):.4f})")

    # --- Stream 2 pipeline ---
    pipeline = Stream2ForensicFeaturePipeline.load(pca_path)

    # --- Dataset (eval mode → deterministic center crop) ---
    dataset = ASVspoofDataset(
        data_dir,
        protocol_path,
        stream2_pipeline=pipeline,
        return_stream2=True,
        training=False,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    # --- Inference ---
    all_file_ids = dataset.file_list
    all_scores: list[float] = []
    all_labels: list[int] = []

    for batch in loader:
        audio, stats, labels = batch
        audio = audio.to(device)
        stats = stats.to(device)

        logits = model(audio, stats)                        # [B, 2]
        probs = F.softmax(logits, dim=1)[:, 1]             # P(spoof)
        all_scores.extend(probs.cpu().tolist())
        all_labels.extend(labels.tolist())

    # --- Write CSV ---
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file_id", "score", "label"])
        for file_id, score, label in zip(all_file_ids, all_scores, all_labels):
            writer.writerow([file_id, f"{score:.6f}", label])

    print(f"Scores written to {output_path}  ({len(all_scores)} samples)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract spoof scores from a trained checkpoint")
    p.add_argument("--setup", required=True, choices=["A", "B", "C"])
    p.add_argument("--checkpoint", required=True, help="Path to best_setup_X.pt")
    p.add_argument("--data_dir", required=True, help="Root directory containing audio files")
    p.add_argument("--protocol", required=True, help="Eval protocol file")
    p.add_argument("--pca_path", required=True, help="Path to fitted pca.pkl")
    p.add_argument("--output_csv", required=True, help="Destination CSV file")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    extract_scores(
        setup=args.setup,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        protocol_path=args.protocol,
        pca_path=args.pca_path,
        output_csv=args.output_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
