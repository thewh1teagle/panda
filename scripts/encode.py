"""
uv run hf download --repo-type dataset thewh1teagle/hebrew-tts-dataset --include "heb-female-audio-ipa2-v2.7z" --local-dir .
7z x heb-female-audio-ipa2-v2.7z
mv heb-female-audio-ipa2-v2 dataset/audio/
uv run -m scripts.encode dataset/audio/metadata.csv dataset/audio/metadata_encoded.csv
"""
import argparse
import sys
import torch
import librosa
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from tqdm import tqdm
from src.codec_io import batch_wav_to_codes

parser = argparse.ArgumentParser()
parser.add_argument("metadata", help="path to metadata.csv (id|phonemes)")
parser.add_argument("output", nargs="?", default="dataset/metadata_encoded.csv", help="output path (default: dataset/metadata_encoded.csv)")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

metadata = Path(args.metadata)
audio_dir = metadata.parent / "wav"
out_path = Path(args.output)

rows = []
with open(metadata) as f:
    for line in f:
        parts = line.strip().split("|")
        rows.append(parts)

# sort by duration so batches have similar-length audio (minimizes padding waste)
rows.sort(key=lambda r: librosa.get_duration(path=audio_dir / f"{r[0]}.wav"))

with open(out_path, "w") as out:
    for i in tqdm(range(0, len(rows), args.batch_size)):
        batch = rows[i:i + args.batch_size]
        paths = [audio_dir / f"{r[0]}.wav" for r in batch]
        results = batch_wav_to_codes(paths, device=args.device)
        for (id_, phonemes, *_), flat in zip(batch, results):
            out.write(f"{id_}|{phonemes}|{' '.join(str(t) for t in flat)}\n")

print(f"Saved to {out_path}")
