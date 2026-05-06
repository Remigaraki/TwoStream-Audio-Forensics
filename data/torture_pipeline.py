import argparse
import os
import shutil
from pathlib import Path

from pydub import AudioSegment


CODEC_CONFIG = {
    "mp3": {"format": "mp3", "codec": "libmp3lame", "extension": "mp3"},
    "opus": {"format": "ogg", "codec": "libopus", "extension": "ogg"},
    "aac": {"format": "adts", "codec": "aac", "extension": "aac"},
}


def _resolve_ffmpeg_binary():
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    local_app_data = Path.home() / "AppData" / "Local"
    preferred_roots = [
        local_app_data / "Microsoft" / "WinGet" / "Packages",
        local_app_data / "CapCut" / "Apps",
        local_app_data / "Overwolf",
    ]

    for root in preferred_roots:
        if not root.exists():
            continue
        for candidate in root.rglob("ffmpeg.exe"):
            return str(candidate)

    raise FileNotFoundError(
        "ffmpeg was not found. Install it with winget or add it to PATH before running codec augmentation."
    )


def compress_audio(input_path: str, output_path: str, codec: str, bitrate: str):
    """
    Apply lossy compression using pydub/ffmpeg to simulate transport damage.
    """
    if codec not in CODEC_CONFIG:
        raise ValueError(f"Unsupported codec '{codec}'. Choose from: {sorted(CODEC_CONFIG)}")

    config = CODEC_CONFIG[codec]
    AudioSegment.converter = _resolve_ffmpeg_binary()
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format=config["format"], codec=config["codec"], bitrate=bitrate)


def process_directory(input_dir: str, output_dir: str, codec: str, bitrate: str):
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if codec not in CODEC_CONFIG:
        raise ValueError(f"Unsupported codec '{codec}'. Choose from: {sorted(CODEC_CONFIG)}")

    extension = CODEC_CONFIG[codec]["extension"]

    for path in input_root.iterdir():
        if path.suffix.lower() not in {".flac", ".wav", ".ogg", ".mp3", ".aac"}:
            continue

        output_name = f"{path.stem}_{codec}_{bitrate}.{extension}"
        output_path = output_root / output_name
        compress_audio(str(path), str(output_path), codec, bitrate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Torture Pipeline: Audio Compression")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--codec", type=str, choices=sorted(CODEC_CONFIG), default="mp3")
    parser.add_argument("--bitrate", type=str, default="64k")
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir, args.codec, args.bitrate)
