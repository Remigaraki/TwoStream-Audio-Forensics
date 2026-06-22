"""
Tests for the pipeline layer: codec augmentation and HearingRealityDataset.

Tests
-----
1. test_transcode_length     — all 6 codec conditions, output length ≈ 64000 (±5%)
2. test_transcode_rms        — RMS > 0.001 for all 6 conditions
3. test_transcode_no_nan     — no NaN for all 6 conditions
4. test_dataset_item         — Dataset[0] → [1, 64000], integer label
5. test_no_split_overlap     — no file-path overlap between train / val / test splits
6. test_local_root_flat      — local_root + flat=True remaps paths to a flat directory
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 16000
DURATION = 4.0
N = int(SR * DURATION)  # 64000


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _write_sine_wav(path: str, freq: float = 440.0, sr: int = SR, duration: float = DURATION) -> None:
    """Write a mono 16-bit PCM sine-wave WAV to *path*."""
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    samples = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sine_wav(tmp_path_factory) -> str:
    """Temporary 4-second sine-wave WAV file."""
    p = tmp_path_factory.mktemp("audio") / "sine.wav"
    _write_sine_wav(str(p))
    return str(p)


@pytest.fixture(scope="module")
def tiny_manifest(tmp_path_factory) -> Path:
    """
    Create a small manifest JSON + WAV files for Dataset tests.
    Returns path to the manifest JSON.
    """
    base = tmp_path_factory.mktemp("dataset")

    # Create 6 WAV files: 3 per split (train/val/test) x 2 labels
    splits_and_labels = [
        ("train", 0), ("train", 1), ("train", 0),
        ("val",   0), ("val",   1),
        ("test",  0), ("test",  1),
    ]
    records = []
    for split, label in splits_and_labels:
        fname = base / f"{split}_{label}_{len(records)}.wav"
        _write_sine_wav(str(fname), freq=440 + label * 100)
        records.append({"path": str(fname), "label": label, "split": split})

    manifest = base / "manifest.json"
    manifest.write_text(json.dumps(records), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# 1-3. transcode tests (skip when ffmpeg not available)
# ---------------------------------------------------------------------------

pytestmark_ffmpeg = pytest.mark.skipif(
    not _has_ffmpeg(), reason="ffmpeg not found in PATH"
)

from src.pipeline.augment import transcode, CODEC_CONDITIONS


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not found in PATH")
@pytest.mark.parametrize("codec,bitrate", CODEC_CONDITIONS)
def test_transcode_length(sine_wav, codec, bitrate):
    """Transcoded output length should be within ±5% of 64000."""
    samples = transcode(sine_wav, codec, bitrate)
    assert isinstance(samples, np.ndarray), "Expected np.ndarray"
    lo, hi = int(N * 0.95), int(N * 1.05)
    assert lo <= len(samples) <= hi, (
        f"codec={codec} bitrate={bitrate}: length {len(samples)} not in [{lo}, {hi}]"
    )


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not found in PATH")
@pytest.mark.parametrize("codec,bitrate", CODEC_CONDITIONS)
def test_transcode_rms(sine_wav, codec, bitrate):
    """RMS of transcoded audio must exceed 0.001 (not silent)."""
    samples = transcode(sine_wav, codec, bitrate)
    rms = float(np.sqrt(np.mean(samples ** 2)))
    assert rms > 0.001, f"codec={codec} bitrate={bitrate}: RMS={rms:.6f} too low"


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not found in PATH")
@pytest.mark.parametrize("codec,bitrate", CODEC_CONDITIONS)
def test_transcode_no_nan(sine_wav, codec, bitrate):
    """Transcoded audio must not contain NaN or Inf."""
    samples = transcode(sine_wav, codec, bitrate)
    assert not np.isnan(samples).any(), f"NaN in output for codec={codec} bitrate={bitrate}"
    assert not np.isinf(samples).any(), f"Inf in output for codec={codec} bitrate={bitrate}"


# ---------------------------------------------------------------------------
# 4. Dataset item shape
# ---------------------------------------------------------------------------

def test_dataset_item(tiny_manifest):
    """Dataset[0] must return a [1, 64000] tensor and an integer label."""
    from src.pipeline.dataset import HearingRealityDataset

    ds = HearingRealityDataset(tiny_manifest, split="train", augment=False)
    assert len(ds) > 0, "Dataset is empty"

    waveform, label = ds[0]

    assert isinstance(waveform, torch.Tensor), "Waveform must be a torch.Tensor"
    assert waveform.shape == (1, N), f"Expected [1, {N}], got {waveform.shape}"
    assert isinstance(label, int), f"Label must be int, got {type(label)}"


# ---------------------------------------------------------------------------
# CSV manifest fixture (used by tests 5–6)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def csv_manifest(tmp_path_factory) -> Path:
    """
    Manifest CSV + WAV files whose file_path values use a fake Drive prefix,
    but whose audio lives in a separate flat directory — simulating the
    /content/flac_train staging pattern.
    """
    base      = tmp_path_factory.mktemp("csv_dataset")
    flat_dir  = base / "flat_audio"
    flat_dir.mkdir()

    fake_drive = "/content/drive/MyDrive/Processed"

    splits_and_labels = [
        ("train", 0), ("train", 1), ("train", 0),
        ("val",   0), ("val",   1),
        ("test",  0), ("test",  1),
    ]
    rows = []
    for split, label in splits_and_labels:
        fname = f"{split}_{label}_{len(rows)}.wav"
        _write_sine_wav(str(flat_dir / fname), freq=440 + label * 100)
        rows.append({
            "file_path": f"{fake_drive}/asvspoof5/{split}/{fname}",
            "label":     label,
            "split":     split,
            "vocoder_type":   "bonafide" if label == 0 else "A01",
            "dataset_source": "asvspoof5",
        })

    manifest = base / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return manifest, flat_dir, fake_drive


# ---------------------------------------------------------------------------
# 5. No file-path overlap between splits
# ---------------------------------------------------------------------------

def test_no_split_overlap(tiny_manifest):
    """No audio file should appear in more than one split."""
    from src.pipeline.dataset import HearingRealityDataset

    splits = ["train", "val", "test"]
    path_sets = {}
    for split in splits:
        try:
            ds = HearingRealityDataset(tiny_manifest, split=split, augment=False)
            path_sets[split] = {ds._records[i]["path"] for i in range(len(ds))}
        except ValueError:
            path_sets[split] = set()

    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1:]:
            overlap = path_sets[s1] & path_sets[s2]
            assert not overlap, (
                f"File path overlap between '{s1}' and '{s2}': {overlap}"
            )


# ---------------------------------------------------------------------------
# 6. local_root + flat=True path remapping
# ---------------------------------------------------------------------------

def test_local_root_flat(csv_manifest):
    """
    With local_root + flat=True, each file_path in the manifest (which carries
    a fake Drive prefix and subdirectory structure) must be resolved to
    local_root/<filename> — matching the flat staged files on local SSD.
    """
    from src.pipeline.dataset import HearingRealityDataset

    manifest, flat_dir, fake_drive = csv_manifest

    ds = HearingRealityDataset(
        manifest_path=manifest,
        split="train",
        augment=False,
        local_root=flat_dir,
        drive_root=fake_drive,
        flat=True,
    )
    assert len(ds) > 0, "Dataset is empty"

    waveform, label = ds[0]

    assert isinstance(waveform, torch.Tensor), "Waveform must be a torch.Tensor"
    assert waveform.shape == (1, N), f"Expected [1, {N}], got {waveform.shape}"
    assert isinstance(label, int), f"Label must be int, got {type(label)}"
