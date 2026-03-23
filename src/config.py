import argparse
from transformers import Qwen3Config, PreTrainedTokenizerFast
import os

TOKENIZER_DIR = os.path.join(os.path.dirname(__file__), "tokenizer")

tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
vocab = tokenizer.get_vocab()

VOCAB_SIZE = len(vocab)
BOS = vocab["<s>"]
EOS = vocab["</s>"]
PAD = vocab["<pad>"]
TEXT = vocab["<text>"]
GENERATE = vocab["<generate>"]
AUDIO = vocab["<audio>"]
AUDIO_OFFSET = vocab["<audio_0>"]

parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="path to pretokenized dataset")
parser.add_argument("output", help="path to save checkpoints")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--warmup_pct", type=float, default=0.05)
parser.add_argument("--save_every", type=int, default=1)
parser.add_argument("--bf16", action="store_true", default=True)
parser.add_argument("--grad_checkpoint", action="store_true", default=True)
args = parser.parse_args()

def get_model_config():
    config = Qwen3Config.from_pretrained("Qwen/Qwen3-0.6B")
    config.vocab_size = VOCAB_SIZE
    return config
