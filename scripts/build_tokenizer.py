"""
Build a WordLevel tokenizer with a fully known vocabulary:
  - special tokens
  - IPA/punctuation symbols from piper ljspeech config
  - speaker tokens (up to MAX_SPEAKERS)
  - audio tokens (SNAC: 0..4095)

uv run scripts/build_tokenizer.py
"""
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast
from tokenizers import Regex
import json, os

SPECIAL = ["<pad>", "<unk>", "<s>", "</s>", "<speaker>", "<text>", "<generate>", "<audio>"]

# 157 symbols from piper ljspeech phoneme_id_map
IPA_CHARS = list(' !"#$\'(),-.0123456789:;?X^_abcdefghijklmnopqrstuvwxyzæçðøħŋœǀǁǂǃɐɑɒɓɔɕɖɗɘəɚɛɜɞɟɠɡɢɣɤɥɦɧɨɪɫɬɭɮɯɰɱɲɳɴɵɶɸɹɺɻɽɾʀʁʂʃʄʈʉʊʋʌʍʎʏʐʑʒʔʕʘʙʛʜʝʟʡʢʦʰʲˈˌːˑ˞ˤ̧̩̪̯̺̻̃βεθχᵻ↑↓ⱱ')

MAX_SPEAKERS = 2048
SPEAKER_TOKENS = [f"<speaker_{i}>" for i in range(MAX_SPEAKERS)]

MAX_AUDIO = 4096
AUDIO_TOKENS = [f"<audio_{i}>" for i in range(MAX_AUDIO)]

vocab = {}
for tok in SPECIAL + IPA_CHARS + SPEAKER_TOKENS + AUDIO_TOKENS:
    if tok not in vocab:
        vocab[tok] = len(vocab)

print(f"Vocab size: {len(vocab)}")

tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
tokenizer.pre_tokenizer = Split(pattern=Regex(r"\S+"), behavior="isolated")

hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="<unk>",
    pad_token="<pad>",
    bos_token="<s>",
    eos_token="</s>",
)

out_dir = os.path.join(os.path.dirname(__file__), "..", "src", "tokenizer")
os.makedirs(out_dir, exist_ok=True)
hf_tokenizer.save_pretrained(out_dir)
print(f"Saved to {out_dir}")

# sanity check
ids = hf_tokenizer.encode("<s> <speaker> <speaker_42> <text> h ə l <generate> <audio_0> <audio_1> </s>")
print("Test encode:", ids)
print("Test decode:", hf_tokenizer.decode(ids))
