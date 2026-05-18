"""Codec augmentation via ffmpeg + pydub for robustness training.

transcode(input_path, codec, bitrate) -> np.ndarray  (float32, 16kHz, mono)

Supported codec conditions:
    ('opus', 16), ('opus', 32), ('opus', 64)
    ('mp3',  64), ('mp3', 128)
    ('aac', 128)
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Tuple

import numpy as np
import torch
import torchaudio.functional as F
from pydub import AudioSegment

# All codec conditions used in robustness evaluation / augmentation
CODEC_CONDITIONS: list[Tuple[str, int]] = [
    ("opus",  16),
    ("opus",  32),
    ("opus",  64),
    ("mp3",   64),
    ("mp3",  128),
    ("aac",  128),
]

# Map codec name -> (ffmpeg_codec, file_extension)
_CODEC_MAP = {
    "opus": ("libopus", ".ogg"),
    "mp3":  ("libmp3lame", ".mp3"),
    "aac":  ("aac", ".m4a"),
}


def transcode(input_path: str, codec: str, bitrate: int) -> np.ndarray:
    """
    Transcode *input_path* through the given codec at *bitrate* kbps,
    then decode back to a float32 mono 16kHz numpy array.

    Parameters
    ----------
    input_path : path to the source audio file
    codec      : one of 'opus', 'mp3', 'aac'
    bitrate    : target bitrate in kbps (e.g. 16, 32, 64, 128)

    Returns
    -------
    np.ndarray  shape (N,), float32, 16kHz, values in [-1, 1]
    """
    if codec not in _CODEC_MAP:
        raise ValueError(f"Unsupported codec '{codec}'. Choose from {list(_CODEC_MAP)}")

    ffmpeg_codec, ext = _CODEC_MAP[codec]
    bitrate_str = f"{bitrate}k"

    tmp_encoded = None
    try:
        # Write transcoded file to a temp path
        fd, tmp_encoded = tempfile.mkstemp(suffix=ext)
        os.close(fd)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vn",                     # drop video
            "-acodec", ffmpeg_codec,
            "-b:a", bitrate_str,
            "-ar", "16000",            # resample during encode
            "-ac", "1",                # mono
            tmp_encoded,
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed (codec={codec}, bitrate={bitrate}k):\n"
                + result.stderr.decode(errors="replace")
            )

        # Decode with pydub
        seg = AudioSegment.from_file(tmp_encoded)
        seg = seg.set_channels(1).set_frame_rate(16000)

        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        # Normalise to [-1, 1] based on bit depth
        max_val = float(2 ** (seg.sample_width * 8 - 1))
        samples /= max_val

        # Ensure 16kHz via torchaudio if pydub frame_rate differs
        actual_sr = seg.frame_rate
        if actual_sr != 16000:
            t = torch.from_numpy(samples).unsqueeze(0)
            t = F.resample(t, actual_sr, 16000)
            samples = t.squeeze(0).numpy()

        return samples.astype(np.float32)

    finally:
        if tmp_encoded is not None and os.path.exists(tmp_encoded):
            try:
                os.remove(tmp_encoded)
            except OSError:
                pass
