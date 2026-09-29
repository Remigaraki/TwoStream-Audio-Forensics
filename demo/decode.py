"""Bounded audio decoder, run as a subprocess by demo/inference.py.

Untrusted media is parsed here, not in the Streamlit server process, so a
hung or crashing decoder is killed by a timeout instead of taking the app down.
Only numpy + soundfile are imported to keep start-up cheap.

    python demo/decode.py <audio_path> <out.npy> <max_seconds>

Reads at most `max_seconds` of audio with exactly the sf.read call used by
_ManifestDataset (dtype=float32, always_2d=False) and prints a JSON header.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np
import soundfile as sf

ALLOWED_FORMATS = {"WAV", "WAVEX", "FLAC", "MP3"}
MIN_SR, MAX_SR, MAX_CHANNELS = 8000, 192000, 8


def _limit_resources() -> None:
    try:
        import resource  # POSIX only; Windows local runs rely on the timeout alone
        resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
        resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
    except (ImportError, ValueError, OSError):
        pass


def main(path: str, out_path: str, max_seconds: float) -> dict:
    try:
        info = sf.info(path)
    except Exception:
        return {"error": "unreadable", "detail": "Not a decodable WAV, FLAC or MP3 file."}
    if info.format not in ALLOWED_FORMATS:
        return {"error": "format", "detail": f"Container '{info.format}' is not WAV, FLAC or MP3."}
    if not (MIN_SR <= info.samplerate <= MAX_SR):
        return {"error": "samplerate", "detail": f"Sample rate {info.samplerate} Hz is outside {MIN_SR}-{MAX_SR} Hz."}
    if not (1 <= info.channels <= MAX_CHANNELS):
        return {"error": "channels", "detail": f"{info.channels} channels is outside 1-{MAX_CHANNELS}."}

    n = min(info.frames, math.ceil(max_seconds * info.samplerate)) if info.frames > 0 else 0
    data, sr = sf.read(path, frames=n, dtype="float32", always_2d=False)
    np.save(out_path, data, allow_pickle=False)
    return {
        "format": info.format, "samplerate": sr, "channels": info.channels,
        "total_frames": info.frames, "read_frames": int(data.shape[0]),
    }


if __name__ == "__main__":
    _limit_resources()
    print(json.dumps(main(sys.argv[1], sys.argv[2], float(sys.argv[3]))))
