"""CPU inference for the demo: trusted-asset loading + the evaluated preprocessing.

Preprocessing mirrors src/train.py::_ManifestDataset.__getitem__ (augment="none"),
which scripts/predict_testset.py uses for every reported score:
    sf.read(float32) -> [C, N] -> torchaudio resample to 16 kHz -> channel mean
    -> first 64000 samples, or zero-pad.
No normalisation, trimming or aggregation is added. The only difference is that
we decode a bounded prefix (4 s + margin) instead of the whole file; the
resampler is local, so the first 64000 output samples are unchanged
(checked in demo/tests/test_demo.py against _ManifestDataset itself).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as F_audio

DEMO_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("DEMO_ASSET_ROOT", DEMO_DIR.parent))
sys.path.insert(0, str(DEMO_DIR.parent))

from src.fusion.two_stream_net import TwoStreamFusionNet  # noqa: E402
from demo.artifact_identity import model_identity, verify_reports  # noqa: E402

TARGET_SR = 16000
TARGET_LEN = 64000                      # 4 s, as in _ManifestDataset
DECODE_SECONDS = 4.5                    # 4 s + resampler margin (kernel spans < 2 ms)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MIN_DURATION_S, MAX_DURATION_S = 0.5, 600.0
SILENCE_PEAK = 1e-4                     # -80 dBFS; input-quality check, not a speech detector
DECODE_TIMEOUT_S = 20
QUEUE_TIMEOUT_S = 90
LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"

# Trusted artifacts, pinned by SHA-256 (hashes of the committed files on main).
MODELS = {
    "C1": {
        "label": "C1 - two-stream fusion, cross-modal attention",
        "setup": "C", "fusion": "attention",
        "checkpoint": ("checkpoints/setup_c/best_c1_attention.pt",
                       "3047175ddc1b2f35ed7f31541c24870f3adffdf5099a0f5dc257f651dc3a930d"),
        "pca": ("data/pca_model/pca.pkl",
                "224766ac0d971c36439d2b1c3b7d6ede5c16d7e9567762fd738710dc11978c31"),
    },
    "A1": {
        "label": "A1 - RawNet2 only, codec augmentation",
        "setup": "A", "fusion": "attention",  # fusion is ignored for setup A
        "checkpoint": ("checkpoints/setup_a/best_a_ep13_eer0105_0713_0401.pt",
                       "5f48571c9bce368d7ec20b7e0c02abea0d241d9c4bf04ccd967bb1fcaba8da0e"),
        "pca": None,
    },
}
THRESHOLDS_PATH = DEMO_DIR / "thresholds.json"

# ponytail: one inference at a time; raise DEMO_MAX_CONCURRENT on bigger hardware
_SLOTS = threading.BoundedSemaphore(int(os.environ.get("DEMO_MAX_CONCURRENT", "1")))
torch.set_num_threads(int(os.environ.get("DEMO_TORCH_THREADS", "2")))


class AssetError(RuntimeError):
    """A model/PCA file is missing, an LFS pointer, or fails its hash."""


class InputError(ValueError):
    """The upload cannot be analysed; message is safe to show the user."""


class BusyError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_asset(rel_path: str, expected_sha: str) -> Path:
    path = ROOT / rel_path
    if not path.is_file():
        raise AssetError(f"Missing model asset: {rel_path} (looked under {ROOT})")
    with open(path, "rb") as fh:
        if fh.read(len(LFS_MAGIC)) == LFS_MAGIC:
            raise AssetError(f"{rel_path} is a Git LFS pointer, not the binary. Run `git lfs pull`.")
    actual = sha256_file(path)
    if actual != expected_sha:
        raise AssetError(f"{rel_path} SHA-256 mismatch: expected {expected_sha[:12]}..., got {actual[:12]}...")
    return path


def load_model(name: str) -> TwoStreamFusionNet:
    """Hash-verify, then build and strictly load. Hashes are checked before any unpickling."""
    spec = MODELS[name]
    ckpt_path = verify_asset(*spec["checkpoint"])
    pca_path = verify_asset(*spec["pca"]) if spec["pca"] else None
    model = TwoStreamFusionNet(pca_path=str(pca_path) if pca_path else None,
                               setup=spec["setup"], fusion=spec["fusion"])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model.eval()


def load_thresholds() -> dict:
    data = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    for name in MODELS:
        t = data[name]
        if not (0.0 <= float(t["threshold"]) <= 1.0) or t["status"] not in ("provisional", "validation"):
            raise AssetError(f"Invalid threshold entry for {name} in {THRESHOLDS_PATH.name}")
        if t['status'] == 'validation':
            current = model_identity(name, MODELS[name])
            if t.get('provenance', {}).get('identity') != current:
                raise AssetError(f'{name} threshold preprocessing/checkpoint/PCA identity mismatch')
            reports_path = THRESHOLDS_PATH.with_suffix('.verification.json')
            try:
                reports = json.loads(reports_path.read_text(encoding='utf-8'))
                verify_reports(THRESHOLDS_PATH, reports, {name: current})
            except (OSError, ValueError) as exc:
                raise AssetError(f'{name} threshold is not parity verified: {exc}') from exc
    return data


def preprocess(waveform_np: np.ndarray, sr: int) -> torch.Tensor:
    """Verbatim logic of _ManifestDataset.__getitem__ (augment='none'). Returns [1, 1, 64000] float32."""
    waveform = torch.from_numpy(waveform_np).float()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 2 and waveform.shape[1] != 1:
        waveform = waveform.T
    if sr != TARGET_SR:
        waveform = F_audio.resample(waveform, sr, TARGET_SR)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    length = waveform.shape[-1]
    if length > TARGET_LEN:
        waveform = waveform[:, :TARGET_LEN]
    elif length < TARGET_LEN:
        waveform = torch.nn.functional.pad(waveform, (0, TARGET_LEN - length))
    return waveform.unsqueeze(0).contiguous()


def decode_bounded(path: Path, workdir: Path) -> tuple[np.ndarray, dict]:
    """Decode at most DECODE_SECONDS in an isolated, time-limited subprocess."""
    out = workdir / "decoded.npy"
    try:
        proc = subprocess.run(
            [sys.executable, str(DEMO_DIR / "decode.py"), str(path), str(out), str(DECODE_SECONDS)],
            capture_output=True, timeout=DECODE_TIMEOUT_S, text=True,
        )
    except subprocess.TimeoutExpired:
        raise InputError("Decoding timed out; the file may be malformed.") from None
    if proc.returncode != 0 or not proc.stdout.strip():
        raise InputError("The file could not be decoded (corrupt or unsupported).")
    meta = json.loads(proc.stdout.strip().splitlines()[-1])
    if "error" in meta:
        raise InputError(meta["detail"])
    return np.load(out, allow_pickle=False), meta


@dataclass
class Prepared:
    x: torch.Tensor           # [1, 1, 64000]
    meta: dict                # decoder header
    duration_s: float         # whole recording
    analyzed_s: float         # real audio inside the 4 s window
    padded_s: float           # zero padding appended


def prepare(path: Path, workdir: Path) -> Prepared:
    size = path.stat().st_size
    if size == 0:
        raise InputError("The file is empty.")
    if size > MAX_UPLOAD_BYTES:
        raise InputError(f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    data, meta = decode_bounded(path, workdir)
    sr = meta["samplerate"]
    duration = meta["total_frames"] / sr if meta["total_frames"] > 0 else data.shape[0] / sr
    if data.shape[0] == 0:
        raise InputError("The file contains no audio samples.")
    if duration < MIN_DURATION_S:
        raise InputError(f"Recording is {duration:.2f} s; minimum is {MIN_DURATION_S} s.")
    if duration > MAX_DURATION_S:
        raise InputError(f"Recording is {duration / 60:.1f} min; maximum is {MAX_DURATION_S / 60:.0f} min.")
    if not np.isfinite(data).all():
        raise InputError("The audio contains NaN or infinite sample values.")

    x = preprocess(data, sr)
    real = min(data.shape[0] * TARGET_SR / sr, TARGET_LEN) / TARGET_SR
    if float(x.abs().max()) < SILENCE_PEAK:  # padding is zeros, so this is the real segment's peak
        raise InputError("The analysed segment is silent or near-silent (peak below -80 dBFS).")
    return Prepared(x=x, meta=meta, duration_s=duration, analyzed_s=real, padded_s=TARGET_LEN / TARGET_SR - real)


def score(model: TwoStreamFusionNet, x: torch.Tensor) -> tuple[float, float]:
    """Returns (score, seconds). The network already ends in Sigmoid: no second sigmoid here."""
    if not _SLOTS.acquire(timeout=QUEUE_TIMEOUT_S):
        raise BusyError("The demo is busy with other requests. Please try again shortly.")
    try:
        t0 = time.perf_counter()
        with torch.inference_mode():
            s = float(model(x).reshape(-1)[0])
        return s, time.perf_counter() - t0
    finally:
        _SLOTS.release()


def decide(s: float, threshold: float) -> str:
    # Same direction as predict_testset.py: score >= threshold -> spoof (label 1).
    return "Likely synthetic or spoofed speech" if s >= threshold else "Likely bona fide speech"


def analyze_file(path: Path, model: TwoStreamFusionNet) -> dict:
    """CLI/test helper: decode in a private temp dir, score, clean up."""
    with tempfile.TemporaryDirectory(prefix="demo-") as tmp:
        p = prepare(Path(path), Path(tmp))
        s, secs = score(model, p.x)
    return {"score": s, "seconds": secs, "analyzed_s": p.analyzed_s, "duration_s": p.duration_s}


if __name__ == "__main__":
    # python demo/inference.py <audio> [C1|A1]
    m = load_model(sys.argv[2] if len(sys.argv) > 2 else "C1")
    print(json.dumps(analyze_file(Path(sys.argv[1]), m), indent=2))
