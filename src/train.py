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
from src.pipeline.augment import transcode, CODEC_CONDITIONS
from src.pipeline.specaugment import spec_augment
from src.stream2.pca_compressor import BispectralPCA


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

    def __init__(
        self,
        manifest_path: str,
        split: str,
        data_root: str | None = None,
        augment: str = "none",
        augment_prob: float = 0.5,
        seed: int = 42,
    ):
        self.records = []
        self.data_root = Path(data_root) if data_root else None
        with open(manifest_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("split", "train") == split:
                    self.records.append(row)

        if not self.records:
            raise ValueError(f"No records for split='{split}' in {manifest_path}")

        # Augmentation is OFF by default ("none") so A0/B0/C0 reproduce exactly.
        # The RNGs here are seeded independently of the (fixed, baked-in-the-
        # manifest) train/val/test split seed — this only controls which
        # samples get augmented and how, not which samples land in which split.
        if augment not in ("none", "codec", "specaugment"):
            raise ValueError(f"augment must be 'none', 'codec', or 'specaugment', got '{augment}'")
        self.augment = augment
        self.augment_prob = augment_prob
        self._rng = random.Random(seed)
        self._torch_gen = torch.Generator().manual_seed(seed)

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

        if self.augment == "codec" and self._rng.random() < self.augment_prob:
            codec, bitrate = self._rng.choice(CODEC_CONDITIONS)
            try:
                waveform_np = transcode(audio_path, codec, bitrate)
                sr = 16000  # transcode() always returns 16kHz
            except Exception:
                waveform_np, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        else:
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

        if self.augment == "specaugment" and self._rng.random() < self.augment_prob:
            waveform = spec_augment(waveform, generator=self._torch_gen)

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

    # --- Ablation flags (all default to baseline-reproducing values) ---------
    p.add_argument("--augment", choices=["none", "codec", "specaugment"], default="none",
                   help="Train-time augmentation for Stream A inputs only (A1/A2 ablations). "
                        "OFF by default so A0 reproduces exactly.")
    p.add_argument("--augment_prob", type=float, default=0.5,
                   help="Per-sample probability of applying --augment.")
    p.add_argument("--seed", type=int, default=42,
                   help="Augmentation RNG seed. Independent of the manifest train/val/test "
                        "split seed (fixed at manifest-generation time, never changed here) — "
                        "this only controls which samples get augmented and how.")
    p.add_argument("--pca_k", type=int, default=128,
                   help="Expected PCA component count K for --pca_path (B1 ablation: 64/128/256). "
                        "Hard-fails if it doesn't match the loaded pca.pkl's actual K.")
    p.add_argument("--features", choices=["lfcc", "lfcc+cqcc"], default="lfcc",
                   help="Stream B feature set (B2 ablation adds CQCC). Setup B/C only.")
    p.add_argument("--fusion", choices=["attention", "concat"], default="attention",
                   help="Setup C fusion head: cross-modal attention (default) or plain "
                        "concat of E_raw+E_stat (C2 ablation).")
    p.add_argument("--freeze_streams", action=argparse.BooleanOptionalAction, default=True,
                   help="Setup C only, used with --init_stream1_from/--init_stream2_from: "
                        "freeze the transplanted Stream A/B weights so only the fusion head "
                        "trains (C1/C2 ablations isolate the fusion's contribution).")
    p.add_argument("--init_stream1_from", default=None,
                   help="Setup C only: checkpoint (.pt) to load Stream 1 weights from "
                        "(e.g. best A checkpoint), for C1/C2 ablations.")
    p.add_argument("--init_stream2_from", default=None,
                   help="Setup C only: checkpoint (.pt) to load Stream 2 weights from "
                        "(e.g. best B checkpoint), for C1/C2 ablations.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}", flush=True)

    # ------------------------------------------------------------------
    # Ablation flag validation (fail fast, before any data/model loading)
    # ------------------------------------------------------------------
    if args.augment != "none" and args.setup != "A":
        raise ValueError(f"--augment is only supported for --setup A (got setup={args.setup})")
    if args.fusion != "attention" and args.setup != "C":
        raise ValueError(f"--fusion only applies to --setup C (got setup={args.setup})")
    if args.features != "lfcc" and args.setup not in ("B", "C"):
        raise ValueError(f"--features only applies to --setup B or C (got setup={args.setup})")
    if (args.init_stream1_from or args.init_stream2_from) and args.setup != "C":
        raise ValueError("--init_stream1_from/--init_stream2_from only apply to --setup C")

    print(
        f"[init] augment={args.augment} augment_prob={args.augment_prob} "
        f"aug_seed={args.seed} (independent of the manifest train/val/test split seed, "
        f"which is fixed at manifest-generation time and unaffected by this flag)",
        flush=True,
    )

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
        print(f"[init] reading manifest …", flush=True)
        train_ds = _ManifestDataset(
            args.manifest, split="train", data_root=train_root,
            augment=args.augment, augment_prob=args.augment_prob, seed=args.seed,
        )
        print(f"[init] train_ds={len(train_ds)} samples", flush=True)
        # Augmentation is train-only — val must stay clean for comparable EER across variants.
        val_ds = _ManifestDataset(args.manifest, split="val", data_root=val_root)
        print(f"[init] val_ds={len(val_ds)} samples", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    print(f"[init] loaders ready — {len(train_loader)} train batches, {len(val_loader)} val batches", flush=True)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    pca_path = args.pca_path
    if args.setup in ("B", "C") and pca_path is None:
        raise ValueError(f"--pca_path is required for setup {args.setup}")

    # B1 ablation: hard-fail if --pca_k doesn't match the loaded pca.pkl's actual K,
    # rather than silently padding/truncating (pca_compressor pads transform() output,
    # which would mask a mismatched checkpoint).
    if pca_path is not None:
        loaded_pca = BispectralPCA.load(pca_path)
        if loaded_pca.n_components != args.pca_k:
            raise ValueError(
                f"--pca_k={args.pca_k} does not match {pca_path} "
                f"(fitted with n_components={loaded_pca.n_components})"
            )

    print(f"[init] building model (setup={args.setup}, fusion={args.fusion}, features={args.features}) …", flush=True)
    model = TwoStreamFusionNet(
        pca_path=pca_path,
        setup=args.setup,
        fusion=args.fusion,
        features=args.features,
    )

    # C1/C2 ablations: transplant pretrained Stream A / Stream B weights into
    # the fusion model, optionally freezing them so only the fusion head trains.
    if args.init_stream1_from:
        sd = torch.load(args.init_stream1_from, map_location="cpu")["model_state"]
        stream1_sd = {k[len("stream1."):]: v for k, v in sd.items() if k.startswith("stream1.")}
        model.stream1.load_state_dict(stream1_sd)
        print(f"[init] loaded stream1 weights from {args.init_stream1_from}", flush=True)
    if args.init_stream2_from:
        sd = torch.load(args.init_stream2_from, map_location="cpu")["model_state"]
        stream2_sd = {k[len("stream2."):]: v for k, v in sd.items() if k.startswith("stream2.")}
        model.stream2.load_state_dict(stream2_sd)
        print(f"[init] loaded stream2 weights from {args.init_stream2_from}", flush=True)

    freeze_streams = args.freeze_streams and (args.init_stream1_from or args.init_stream2_from)
    if freeze_streams:
        if args.init_stream1_from:
            for p in model.stream1.parameters():
                p.requires_grad = False
        if args.init_stream2_from:
            for p in model.stream2.parameters():
                p.requires_grad = False
        print("[init] streams frozen — only the fusion head will train", flush=True)

    print(f"[init] moving model to {device} …", flush=True)
    model = model.to(device)
    print(f"[init] model ready", flush=True)

    # ------------------------------------------------------------------
    # Optimizer / loss / scheduler
    # ------------------------------------------------------------------
    criterion = nn.BCELoss()
    weight_decay = 1e-3 if args.setup == "B" else 1e-4
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # LR warmup for Setup C: linearly ramp from 1e-6 to target LR over 10 epochs
    warmup_epochs = 10 if args.setup == "C" else 0
    warmup_start_lr = 1e-6

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
        print(f"Resumed from {args.resume_from} (epoch {start_epoch})", flush=True)

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
    import time
    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)

    for epoch in range(start_epoch, args.epochs):

        # LR warmup
        if epoch < warmup_epochs:
            warmup_lr = warmup_start_lr + (args.lr - warmup_start_lr) * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        # Train
        model.train()
        if freeze_streams:
            # Keep frozen streams in eval mode so their BatchNorm running
            # stats don't drift while only the fusion head is being trained.
            if args.init_stream1_from:
                model.stream1.eval()
            if args.init_stream2_from:
                model.stream2.eval()
        running_loss = 0.0
        label_sum = 0.0
        t0 = time.time()
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.float().to(device)

            optimizer.zero_grad()
            preds = model(batch_x).squeeze(1)  # [B]
            loss = criterion(preds, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * len(batch_x)
            label_sum += batch_y.mean().item()

            if step % 50 == 0:
                elapsed = time.time() - t0 + 1e-6
                bps = (step + 1) / elapsed
                print(f"  [train] epoch {epoch+1} batch {step}/{n_train_batches}"
                      f"  loss={loss.item():.4f}  {bps:.1f} batches/sec", flush=True)

        train_loss = running_loss / len(train_ds)
        mean_label = label_sum / n_train_batches
        train_elapsed = time.time() - t0
        train_bps = n_train_batches / (train_elapsed + 1e-6)

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

        # Scheduler step (skip during warmup)
        if epoch >= warmup_epochs:
            scheduler.step(val_eer)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1:3d} | "
            f"Loss: {train_loss:.4f} | "
            f"Val EER: {val_eer:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Batches/sec: {train_bps:.1f} | "
            f"Mean label: {mean_label:.3f}",
            flush=True,
        )

        if mean_label > 0.97:
            print("  ⚠️  WARNING: mean label > 0.97 — DataLoader may not be sampling bonafide clips", flush=True)
        if train_bps < 1.0:
            print("  ⚠️  WARNING: < 1 batch/sec — check that SSD copy ran before training", flush=True)

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
            print(f"  ✅ New best EER: {best_eer:.4f} — saved best.pt", flush=True)

        if args.use_wandb:
            import wandb
            wandb.log({"train_loss": train_loss, "val_eer": val_eer, "lr": current_lr})

    if args.use_wandb:
        import wandb
        wandb.finish()

    print(f"\nTraining complete. Best val EER: {best_eer:.4f}", flush=True)
    print(f"Checkpoints saved to: {ckpt_dir.resolve()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()
    train(args)
