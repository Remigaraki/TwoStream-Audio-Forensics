"""Training entry point for TwoStream deepfake audio detection.

Usage
-----
    python src/train.py --setup A --manifest path/to/manifest.csv \
        --pca_path data/pca_model/pca.pkl --ckpt_dir checkpoints/ --epochs 50

    python src/train.py --setup C --manifest path/to/manifest.csv \
        --pca_path data/pca_model/pca.pkl --ckpt_dir checkpoints/ --epochs 80 --use_wandb

Setups
------
    A — Stream 1 only  (RawNet2Encoder + classifier)
    B — Stream 2 only  (Stream2 + classifier)
    C — Full fusion    (TwoStreamFusionNet)

Special flags
-------------
    --synthetic_dry_run : generate random tensors instead of loading from disk
                          (useful for CI / unit tests)
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fusion.two_stream_net import TwoStreamFusionNet
from src.eval.metrics import compute_eer


# ---------------------------------------------------------------------------
# Synthetic dataset (for --synthetic_dry_run)
# ---------------------------------------------------------------------------

class _SyntheticDataset(Dataset):
    """100 random [1, 64000] waveforms with random 0/1 labels."""

    def __init__(self, n: int = 100, seed: int = 42):
        rng = torch.Generator()
        rng.manual_seed(seed)
        self.data = torch.randn(n, 1, 64000, generator=rng)
        self.labels = torch.randint(0, 2, (n,), generator=rng).float()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Manifest dataset
# ---------------------------------------------------------------------------

class _ManifestDataset(Dataset):
    """CSV manifest dataset. Each row: path,label,split"""

    def __init__(self, manifest_path: str, split: str, data_root: str | None = None):
        self.records = []
        self.data_root = Path(data_root) if data_root else None
        with open(manifest_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("split", "train") == split:
                    self.records.append(row)

        if not self.records:
            raise ValueError(f"No records for split='{split}' in {manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        import soundfile as sf
        import torchaudio.functional as F_audio

        rec = self.records[idx]
        label = float(int(rec["label"]))

        audio_path = rec.get("file_path") or rec["path"]
        if self.data_root is not None:
            audio_path = str(self.data_root / Path(audio_path).name)

        waveform_np, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        waveform = torch.from_numpy(waveform_np).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 2 and waveform.shape[1] != 1:
            waveform = waveform.T

        if sr != 16000:
            waveform = F_audio.resample(waveform, sr, 16000)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        target = 64000
        length = waveform.shape[-1]
        if length > target:
            waveform = waveform[:, :target]
        elif length < target:
            waveform = nn.functional.pad(waveform, (0, target - length))

        return waveform, torch.tensor(label)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TwoStream training entry point")
    p.add_argument("--setup", required=True, choices=["A", "B", "C"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--manifest", required=False, default=None)
    p.add_argument("--pca_path", default=None)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--resume_from", default=None)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--data_root", default=None,
                   help="Override directory for BOTH train and val audio files (by basename).")
    p.add_argument("--train_data_root", default=None,
                   help="Override directory for train audio files only (by basename).")
    p.add_argument("--val_data_root", default=None,
                   help="Override directory for val audio files only (by basename).")
    p.add_argument("--synthetic_dry_run", action="store_true",
                   help="Use synthetic random tensors (skips --manifest)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    if args.synthetic_dry_run:
        full_ds = _SyntheticDataset(n=100)
        n_val = 20
        n_train = len(full_ds) - n_val
        train_ds, val_ds = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        if args.manifest is None:
            raise ValueError("--manifest is required unless --synthetic_dry_run is set")
        train_root = args.train_data_root or args.data_root
        val_root   = args.val_data_root   or args.data_root
        train_ds = _ManifestDataset(args.manifest, split="train", data_root=train_root)
        val_ds   = _ManifestDataset(args.manifest, split="val",   data_root=val_root)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    pca_path = args.pca_path
    if args.setup in ("B", "C") and pca_path is None:
        raise ValueError(f"--pca_path is required for setup {args.setup}")

    model = TwoStreamFusionNet(
        pca_path=pca_path,
        setup=args.setup,
    ).to(device)

    # ------------------------------------------------------------------
    # Optimizer / loss / scheduler
    # ------------------------------------------------------------------
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch = 0
    best_eer = float("inf")
    if args.resume_from and Path(args.resume_from).exists():
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_eer = ckpt.get("val_eer", float("inf"))
        print(f"Resumed from {args.resume_from} (epoch {start_epoch})")

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.use_wandb:
        import wandb
        wandb.init(project="twostream-audio-forensics", config=vars(args))

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------
    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)
    for epoch in range(start_epoch, args.epochs):
        # Train
        model.train()
        running_loss = 0.0
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.float().to(device)

            optimizer.zero_grad()
            preds = model(batch_x).squeeze(1)  # [B]
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(batch_x)

            if step % 50 == 0:
                print(f"  [train] epoch {epoch+1} batch {step}/{n_train_batches}  loss={loss.item():.4f}", flush=True)

        train_loss = running_loss / len(train_ds)

        # Validate
        model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for step, (batch_x, batch_y) in enumerate(val_loader):
                batch_x = batch_x.to(device)
                preds = model(batch_x).squeeze(1).cpu().numpy()
                all_scores.extend(preds.tolist())
                all_labels.extend(batch_y.numpy().tolist())
                if step % 100 == 0:
                    print(f"  [val]   epoch {epoch+1} batch {step}/{n_val_batches}", flush=True)

        val_eer = compute_eer(
            np.array(all_scores, dtype=np.float32),
            np.array(all_labels, dtype=np.int32),
        )

        # Scheduler step
        scheduler.step(val_eer)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val EER: {val_eer:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # Save latest
        latest_ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_eer": val_eer,
        }
        torch.save(latest_ckpt, ckpt_dir / "latest.pt")

        # Save best
        if val_eer < best_eer:
            best_eer = val_eer
            torch.save(latest_ckpt, ckpt_dir / "best.pt")

        if args.use_wandb:
            import wandb
            wandb.log({"train_loss": train_loss, "val_eer": val_eer, "lr": current_lr})

    if args.use_wandb:
        import wandb
        wandb.finish()

    print(f"\nTraining complete. Best val EER: {best_eer:.4f}")
    print(f"Checkpoints saved to: {ckpt_dir.resolve()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()
    train(args)
