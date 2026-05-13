"""
Torture Chamber — apply lossy codec compression to audio files.

Supported codec conditions (6 total, matching the evaluation protocol):
    Opus  16 kbps, 32 kbps, 64 kbps
    MP3   64 kbps, 128 kbps
    AAC   128 kbps

Usage — single condition:
    python -m data.torture_pipeline \\
        --input_dir  data/raw/ASVspoof5/eval \\
        --output_dir data/lossy/eval \\
        --codec opus --bitrate 32k

Usage — all 6 conditions at once:
    python -m data.torture_pipeline \\
        --input_dir  data/raw/ASVspoof5/eval \\
        --output_dir data/lossy/eval \\
        --all
"""

import argparse
import shutil
from pathlib import Path

from pydub import AudioSegment


CODEC_CONFIG = {
    "mp3":  {"format": "mp3",  "codec": "libmp3lame", "extension": "mp3"},
    "opus": {"format": "ogg",  "codec": "libopus",    "extension": "ogg"},
    "aac":  {"format": "adts", "codec": "aac",        "extension": "aac"},
}

# The 6 evaluation conditions required by the protocol
ALL_CONDITIONS: list[tuple[str, str]] = [
    ("opus", "16k"),
    ("opus", "32k"),
    ("opus", "64k"),
    ("mp3",  "64k"),
    ("mp3",  "128k"),
    ("aac",  "128k"),
]

_AUDIO_EXTENSIONS = {".flac", ".wav", ".ogg", ".mp3", ".aac"}


# ---------------------------------------------------------------------------
# ffmpeg resolution
# ---------------------------------------------------------------------------

def _resolve_ffmpeg_binary() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    local_app_data = Path.home() / "AppData" / "Local"
    search_roots = [
        local_app_data / "Microsoft" / "WinGet" / "Packages",
        local_app_data / "CapCut" / "Apps",
        local_app_data / "Overwolf",
    ]
    for root in search_roots:
        if not root.exists():
            continue
        for candidate in root.rglob("ffmpeg.exe"):
            return str(candidate)

    raise FileNotFoundError(
        "ffmpeg not found. Install it (e.g. winget install ffmpeg) and add it to PATH."
    )


# ---------------------------------------------------------------------------
# Core compression
# ---------------------------------------------------------------------------

def compress_audio(input_path: str, output_path: str, codec: str, bitrate: str) -> None:
    """Transcode a single audio file to the specified codec and bitrate."""
    if codec not in CODEC_CONFIG:
        raise ValueError(f"Unsupported codec '{codec}'. Choose from: {sorted(CODEC_CONFIG)}")

    config = CODEC_CONFIG[codec]
    AudioSegment.converter = _resolve_ffmpeg_binary()
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format=config["format"], codec=config["codec"], bitrate=bitrate)


def process_directory(input_dir: str, output_dir: str, codec: str, bitrate: str) -> int:
    """
    Recursively transcode all audio files in input_dir to output_dir,
    preserving the relative subdirectory structure.

    Returns the number of files processed.
    """
    if codec not in CODEC_CONFIG:
        raise ValueError(f"Unsupported codec '{codec}'. Choose from: {sorted(CODEC_CONFIG)}")

    input_root = Path(input_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    extension = CODEC_CONFIG[codec]["extension"]

    processed = 0
    for input_path in input_root.rglob("*"):
        if input_path.suffix.lower() not in _AUDIO_EXTENSIONS:
            continue

        # Preserve subdirectory structure under output_root
        rel = input_path.relative_to(input_root)
        output_path = output_root / rel.with_suffix(f".{extension}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        compress_audio(str(input_path), str(output_path), codec, bitrate)
        processed += 1

    return processed


def run_all_conditions(input_dir: str, output_base_dir: str) -> None:
    """
    Run all 6 codec conditions in sequence, writing each to a separate subdirectory:
        <output_base_dir>/opus_16k/
        <output_base_dir>/opus_32k/
        ...
    """
    for codec, bitrate in ALL_CONDITIONS:
        condition_label = f"{codec}_{bitrate}"
        output_dir = str(Path(output_base_dir) / condition_label)
        print(f"[{condition_label}] Compressing {input_dir} → {output_dir}")
        n = process_directory(input_dir, output_dir, codec, bitrate)
        print(f"[{condition_label}] Done — {n} file(s) processed")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Torture Chamber: lossy audio compression")
    parser.add_argument("--input_dir",  required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--all", action="store_true",
                        help="Run all 6 codec conditions (ignores --codec/--bitrate)")
    parser.add_argument("--codec",   choices=sorted(CODEC_CONFIG), default="mp3")
    parser.add_argument("--bitrate", default="64k")
    args = parser.parse_args()

    if args.all:
        run_all_conditions(args.input_dir, args.output_dir)
    else:
        n = process_directory(args.input_dir, args.output_dir, args.codec, args.bitrate)
        print(f"Done — {n} file(s) processed")
