import os
from pydub import AudioSegment
import argparse

def compress_audio(input_path: str, output_path: str, codec: str, bitrate: str):
    """
    Applies lossy compression using pydub/ffmpeg to simulate 'torture' conditions.
    """
    audio = AudioSegment.from_file(input_path)
    # codec can be 'mp3' or 'libopus', format can be 'mp3' or 'ogg'
    format_out = 'mp3' if codec == 'mp3' else 'ogg'
    codec_out = 'libmp3lame' if codec == 'mp3' else 'libopus'
    
    audio.export(output_path, format=format_out, codec=codec_out, bitrate=bitrate)

def process_directory(input_dir: str, output_dir: str, codec: str, bitrate: str):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(('.flac', '.wav')):
            input_path = os.path.join(input_dir, filename)
            output_name = os.path.splitext(filename)[0] + f"_{codec}_{bitrate}.{ 'mp3' if codec == 'mp3' else 'ogg'}"
            output_path = os.path.join(output_dir, output_name)
            compress_audio(input_path, output_path, codec, bitrate)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Torture Pipeline: Audio Compression")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--codec", type=str, choices=['mp3', 'opus'], default='mp3')
    parser.add_argument("--bitrate", type=str, default="64k")
    args = parser.parse_args()
    
    process_directory(args.input_dir, args.output_dir, args.codec, args.bitrate)
