"""
uv run -m scripts.pretokenize dataset/audio/metadata_encoded.csv dataset/pretokenized
"""
import argparse
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from transformers import PreTrainedTokenizerFast
from datasets import Dataset
from tqdm import tqdm
from src.codec import interleave, deinterleave
from src.constants import MAX_LEN, VAL_MAX

parser = argparse.ArgumentParser()
parser.add_argument("metadata", help="path to metadata_encoded.csv (id|phonemes|audio_tokens)")
parser.add_argument("output", help="path to save HF dataset (train/ and val/ saved inside)")
args = parser.parse_args()

tokenizer = PreTrainedTokenizerFast.from_pretrained(
    Path(__file__).parent.parent / "src" / "tokenizer"
)

vocab = tokenizer.get_vocab()
BOS = vocab["<s>"]
EOS = vocab["</s>"]
TEXT = vocab["<text>"]
GENERATE = vocab["<generate>"]
AUDIO = vocab["<audio>"]

rows = []
with open(args.metadata) as f:
    for line in f:
        parts = line.strip().split("|")
        if len(parts) == 3:
            rows.append(parts)

random.shuffle(rows)

sequences = []
skipped = 0

for id_, phonemes, audio_tokens_str in tqdm(rows):
    ipa_ids = [vocab[c] for c in phonemes if c in vocab]
    raw = [int(t) for t in audio_tokens_str.split()]
    c0, c1, c2 = deinterleave(raw)
    audio_ids = interleave(c0, c1, c2)
    # <s> <text> {phonemes} <generate> <audio> {audio_tokens} </s>
    ids = [BOS, TEXT] + ipa_ids + [GENERATE, AUDIO] + audio_ids + [EOS]

    if len(ids) > MAX_LEN:
        skipped += 1
        continue

    sequences.append({"input_ids": ids})

print(f"Kept {len(sequences)}, skipped {skipped} (>{MAX_LEN} tokens)")

val = sequences[:VAL_MAX]
train = sequences[VAL_MAX:]

output = Path(args.output)
Dataset.from_list(train).save_to_disk(output / "train")
Dataset.from_list(val).save_to_disk(output / "val")
print(f"Train: {len(train)}, val: {len(val)} — saved to {output}")
