"""Fit BispectralPCA from real data or synthetic waveforms.

Usage (synthetic, for testing):
    python scripts/fit_pca.py --synthetic --save_path data/pca_model/pca.pkl

Usage (real data):
    python scripts/fit_pca.py --manifest path/to/manifest.json --split train \
        --save_path data/pca_model/pca.pkl --n_samples 2000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stream2.bispectrum import estimate_bispectrum
from src.stream2.pca_compressor import BispectralPCA


def _generate_synthetic_bispectra(n: int = 500) -> list[np.ndarray]:
    """Generate n random 64000-sample waveforms and compute their bispectra."""
    rng = np.random.default_rng(42)
    bispectra = []
    for i in range(n):
        waveform = rng.standard_normal(64000).astype(np.float32)
        b_mat = estimate_bispectrum(waveform)  # (128, 128)
        bispectra.append(b_mat)
        if (i + 1) % 50 == 0:
            print(f"  computed {i+1}/{n} bispectra", flush=True)
    return bispectra


def _load_bispectra_from_manifest(
    manifest_path: str, split: str, n_samples: int, data_root: str | None = None
) -> list[np.ndarray]:
    import csv
    import soundfile as sf

    with open(manifest_path, "r", encoding="utf-8", newline="") as fh:
        records = list(csv.DictReader(fh))

    records = [r for r in records if r.get("split") == split and r.get("label") == "0"]
    records = records[:n_samples]
    if not records:
        raise ValueError(f"No bonafide records for split='{split}' in {manifest_path}")

    bispectra = []
    for i, rec in enumerate(records):
        fpath = rec.get("file_path") or rec.get("path")
        if fpath is None:
            raise KeyError(f"Manifest has neither 'file_path' nor 'path' column. Keys: {list(rec.keys())}")
        if data_root and not Path(fpath).is_absolute():
            fpath = str(Path(data_root) / fpath)
        waveform_np, sr = sf.read(fpath, dtype="float32", always_2d=False)
        if waveform_np.ndim == 2:
            waveform_np = waveform_np.mean(axis=1)
        # Resample to 16000 if needed
        if sr != 16000:
            import torchaudio.functional as F
            import torch
            w = torch.from_numpy(waveform_np).unsqueeze(0)
            w = F.resample(w, sr, 16000)
            waveform_np = w.squeeze(0).numpy()
        # Pad / trim to 64000
        target = 64000
        if len(waveform_np) > target:
            waveform_np = waveform_np[:target]
        elif len(waveform_np) < target:
            waveform_np = np.pad(waveform_np, (0, target - len(waveform_np)))
        b_mat = estimate_bispectrum(waveform_np)
        bispectra.append(b_mat)
        if (i + 1) % 50 == 0:
            print(f"  computed {i+1}/{len(records)} bispectra", flush=True)
    return bispectra


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit BispectralPCA")
    parser.add_argument("--manifest", default=None, help="Path to manifest JSON")
    parser.add_argument("--split", default="train", help="Manifest split to use")
    parser.add_argument(
        "--save_path", default="data/pca_model/pca.pkl", help="Output .pkl path"
    )
    parser.add_argument(
        "--n_samples", type=int, default=2000, help="Max samples to use"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use 500 synthetic random waveforms instead of real data"
    )
    parser.add_argument(
        "--data_root", default=None,
        help="Root directory to prepend to relative paths in the manifest"
    )
    parser.add_argument(
        "--n_components", type=int, default=128,
        help="Number of PCA components K (B1 ablation sweeps 64/128/256). "
             "Default 128 matches the production pca.pkl."
    )
    args = parser.parse_args()

    print("=" * 60)
    if args.synthetic:
        print("Mode: SYNTHETIC (500 random waveforms)")
        print("Generating bispectra …")
        bispectra = _generate_synthetic_bispectra(500)
    else:
        if args.manifest is None:
            parser.error("--manifest is required unless --synthetic is set")
        print(f"Mode: REAL DATA  manifest={args.manifest}  split={args.split}")
        print(f"Loading up to {args.n_samples} samples …")
        bispectra = _load_bispectra_from_manifest(
            args.manifest, args.split, args.n_samples, args.data_root
        )

    print(f"\nFitting BispectralPCA (K={args.n_components}) on {len(bispectra)} bispectra …")
    pca = BispectralPCA(n_components=args.n_components)
    pca.fit(bispectra)

    ev = float(pca._pca.explained_variance_ratio_.sum())
    print(f"Explained variance ratio: {ev:.4f}")
    if ev < 0.95:
        print(
            f"WARNING: Explained variance {ev:.4f} < 0.95. "
            "Consider providing more training samples."
        )
    else:
        print("OK: Explained variance >= 0.95")

    save_path = Path(args.save_path)
    pca.save(save_path)
    print(f"\nPCA saved to: {save_path.resolve()}")
    print(f"Components: {pca._pca.n_components_}")
    print("=" * 60)


if __name__ == "__main__":
    main()
