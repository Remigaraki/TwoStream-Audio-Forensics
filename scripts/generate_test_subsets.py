"""
Generate the 6 codec-compressed eval subsets required by the evaluation protocol.

Runs all 6 conditions (Opus 16/32/64 kbps, MP3 64/128 kbps, AAC 128 kbps) on the
clean eval audio directory and writes each condition to a separate subdirectory:

    <output_base>/
        opus_16k/   opus_32k/   opus_64k/
        mp3_64k/    mp3_128k/
        aac_128k/

Preserves relative subdirectory structure from the input directory.
Skips a condition if its output directory already contains files.

Usage (local):
    python -m scripts.generate_test_subsets \\
        --input_dir  data/raw/ASVspoof5/eval \\
        --output_dir data/lossy/eval

Usage (Colab):
    !python -m scripts.generate_test_subsets \\
        --input_dir  /content/drive/MyDrive/ASVspoof5/eval \\
        --output_dir /content/drive/MyDrive/ASVspoof5/tortured_eval
"""

import argparse
from pathlib import Path

from data.torture_pipeline import ALL_CONDITIONS, process_directory


def generate_test_subsets(input_dir: str, output_base_dir: str, skip_existing: bool = True) -> None:
    input_path  = Path(input_dir)
    output_base = Path(output_base_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    for codec, bitrate in ALL_CONDITIONS:
        label      = f"{codec}_{bitrate}"
        output_dir = output_base / label

        if skip_existing and output_dir.exists():
            existing = sum(1 for _ in output_dir.rglob("*") if _.is_file())
            if existing > 0:
                print(f"[{label}] Already has {existing} file(s) — skipping.")
                continue

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{label}] Compressing {input_path} → {output_dir} …")
        n = process_directory(str(input_path), str(output_dir), codec, bitrate)
        print(f"[{label}] Done — {n} file(s) processed.")

    print("\nAll codec conditions complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 6 codec-compressed eval subsets for the evaluation protocol"
    )
    parser.add_argument("--input_dir",  required=True, help="Clean eval audio root directory")
    parser.add_argument("--output_dir", required=True, help="Base output directory for all conditions")
    parser.add_argument(
        "--no_skip", action="store_true",
        help="Reprocess conditions even if output directory already has files"
    )
    args = parser.parse_args()

    generate_test_subsets(
        input_dir=args.input_dir,
        output_base_dir=args.output_dir,
        skip_existing=not args.no_skip,
    )
