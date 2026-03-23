# panda

LLM based text-to-speech

## Training

```bash
# 1. build tokenizer
uv run scripts/build_tokenizer.py

# 2. encode audio (SNAC)
uv run scripts/encode.py dataset/metadata.csv dataset/metadata_encoded.csv

# 3. pretokenize (applies delay pattern, splits train/val)
uv run scripts/pretokenize.py dataset/metadata_encoded.csv dataset/pretokenized

# 4. train
uv run src/train.py dataset/pretokenized checkpoints/
```

## Inference

```bash
uv run src/infer.py checkpoints/epoch_10 "həlˈoʊ" output.wav
```