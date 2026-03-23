# Panda TTS — Methodology & Structure

## What it is
Codec language model for TTS. Takes speaker ID + IPA phonemes, generates SNAC audio tokens. Architecture: Qwen3-0.6B with random weights and custom vocab.

## Prompt format
```
<s><speaker><speaker_N><text> IPA tokens <generate><audio_0>...<audio_X></s>
```

## File structure
```
scripts/
  build_tokenizer.py   # build WordLevel tokenizer from known vocab, save to tokenizer/
  encode.py            # wav → SNAC audio tokens, append to metadata
  pretokenize.py       # build full token sequences, save HF dataset to data/pretokenized/

src/
  config.py            # Qwen3Config + vocab/token ID constants
  model.py             # Qwen3ForCausalLM init with random weights
  train.py             # load dataset, train with cross-entropy (loss on audio tokens only)
  infer.py             # load model, generate from sid + phonemes, SNAC decode → wav

tokenizer/             # saved HF tokenizer artifact
data/pretokenized/     # saved HF dataset (load_from_disk in train)
```

## Dataset format
Input: `id|sid|phonemes`
After encode: `id|sid|phonemes|audio_tokens`

## Vocabulary
- Special: `<pad> <s> </s> <unk> <speaker> <speaker_N> <text> <generate> <audio>`
- IPA: all unique chars collected from phonemes column
- Audio: `<audio_0>` … `<audio_4095>`
- WordLevel tokenizer — no BPE, vocab is fully known upfront
