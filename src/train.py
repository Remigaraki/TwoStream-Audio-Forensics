"""
Training entry point supporting all three experimental setups.

Usage
-----
    # Full fused model (Setup C), 80 epochs
    python -m src.train --setup C \\
        --data_dir /content/drive/MyDrive/data/ASVspoof5 \\
        --protocol_train protocols/train.txt \\
        --epochs 80 --batch_size 32 --use_wandb

    # Stream 1 ablation (Setup A), 50 epochs
    python -m src.train --setup A \\
        --data_dir /content/drive/MyDrive/data/ASVspoof5 \\
        --protocol_train protocols/train.txt \\
        --epochs 50 --batch_size 32

Setups
------
    A — Stream 1 only  (RawNet2Encoder + classifier, stats input ignored)
    B — Stream 2 only  (StatisticalStream + classifier, audio input ignored)
    C — Full fusion    (TwoStreamFusionNet)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn

from data.dataset_loader import ASVspoofDataset, fit_stream2_pipeline_from_protocol
from src.fusion.two_stream_net import StatisticalStream, TwoStreamFusionNet
from src.stream1.rawnet2 import RawNet2Encoder
from src.stream2.extract_bispectrum import Stream2ForensicFeaturePipeline
from src.training.train_loop import (
    TrainingConfig,
    build_two_stream_loaders,
    fit_two_stream_model,
    set_seed,
)
from src.utils.logger import setup_logger


# ---------------------------------------------------------------------------
# Per-setup model wrappers
# ---------------------------------------------------------------------------

class _SetupAModel(nn.Module):
    """Stream 1 only — stats tensor is accepted but ignored."""

    def __init__(self):
        super().__init__()
        self.encoder = RawNet2Encoder(output_dim=256)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, audio: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(audio))


class _SetupBModel(nn.Module):
    """Stream 2 only — audio tensor is accepted but ignored."""

    def __init__(self):
        super().__init__()
        self.stream2 = StatisticalStream(input_dim=248, output_dim=64)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),
        )

    def forward(self, audio: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.stream2(stats))


def build_model(setup: str) -> nn.Module:
    """Return the appropriate model for setup A, B, or C."""
    setup = setup.upper()
    if setup == "A":
        return _SetupAModel()
    if setup == "B":
        return _SetupBModel()
    if setup == "C":
        return TwoStreamFusionNet()
    raise ValueError(f"Unknown setup '{setup}'. Choose A, B, or C.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TwoStream deepfake audio detection — training entry point"
    )
    p.add_argument(
        "--setup", required=True, choices=["A", "B", "C"],
        help="A=Stream1 only  B=Stream2 only  C=Full fusion",
    )
    p.add_argument("--data_dir", required=True, help="Root directory containing audio files")
    p.add_argument("--protocol_train", required=True, help="Training protocol file")
    p.add_argument("--pca_path", default=None,
                   help="Path to a pre-fitted pca.pkl. "
                        "If absent, PCA is fitted from --protocol_train and saved automatically.")
    p.add_argument("--max_fit_samples", type=int, default=2000,
                   help="Max audio samples used to fit the Stream 2 PCA (default: 2000)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    p.add_argument("--checkpoint_dir", default="checkpoints",
                   help="Directory for saving best checkpoints and logs")
    p.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb_project", default="twostream-audio-forensics")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    set_seed(args.seed)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = checkpoint_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        f"train_setup_{args.setup}",
        log_dir / f"setup_{args.setup}.log",
    )
    logger.info(
        "Setup %s | epochs=%d | batch=%d | lr=%g | scheduler=%s",
        args.setup, args.epochs, args.batch_size, args.lr, args.scheduler,
    )

    # ------------------------------------------------------------------
    # Stream 2 pipeline (PCA) — load or fit
    # ------------------------------------------------------------------
    pca_path = Path(args.pca_path) if args.pca_path else checkpoint_dir / "pca.pkl"
    if pca_path.exists():
        logger.info("Loading Stream 2 pipeline from %s", pca_path)
        pipeline = Stream2ForensicFeaturePipeline.load(pca_path)
    else:
        logger.info(
            "Fitting Stream 2 pipeline on up to %d samples from %s …",
            args.max_fit_samples, args.protocol_train,
        )
        pipeline = fit_stream2_pipeline_from_protocol(
            args.data_dir,
            args.protocol_train,
            max_items=args.max_fit_samples,
        )
        pipeline.save(pca_path)
        logger.info("Pipeline saved to %s", pca_path)

    # ------------------------------------------------------------------
    # Dataset & loaders
    # ------------------------------------------------------------------
    train_dataset = ASVspoofDataset(
        args.data_dir,
        args.protocol_train,
        stream2_pipeline=pipeline,
        return_stream2=True,
        training=True,
    )
    logger.info("Training dataset: %d samples", len(train_dataset))

    train_loader, val_loader = build_two_stream_loaders(
        train_dataset,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(args.setup)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model — Setup %s | %.2fM parameters", args.setup, n_params / 1e6)

    # ------------------------------------------------------------------
    # Training config
    # ------------------------------------------------------------------
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_path=str(checkpoint_dir / f"best_setup_{args.setup}.pt"),
        log_dir=str(log_dir),
        scheduler_type=args.scheduler,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=f"setup_{args.setup}",
        num_workers=args.num_workers,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    history = fit_two_stream_model(model, train_loader, val_loader, config, logger=logger)

    valid_eers = [e["val_eer"] for e in history if not math.isnan(e["val_eer"])]
    best_eer = min(valid_eers) if valid_eers else float("nan")
    logger.info("Training complete. Best val EER: %.4f", best_eer)
    print(f"\nSetup {args.setup} done. Best val EER: {best_eer:.4f}")
    print(f"Checkpoint: {checkpoint_dir / f'best_setup_{args.setup}.pt'}")


if __name__ == "__main__":
    main()
