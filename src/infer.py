import argparse
import torch
import soundfile as sf
from transformers import Qwen3ForCausalLM, PreTrainedTokenizerFast

from src.config import BOS, EOS, TEXT, GENERATE, AUDIO, TOKENIZER_DIR
from src.constants import SAMPLE_RATE
from src.codec_io import decode

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", help="path to model checkpoint")
parser.add_argument("phonemes", help="IPA phoneme string")
parser.add_argument("output", help="output wav path")
parser.add_argument("--max_tokens", type=int, default=1024)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
vocab = tokenizer.get_vocab()

model = Qwen3ForCausalLM.from_pretrained(args.checkpoint).to(device).eval()

ipa_ids = [vocab[c] for c in args.phonemes if c in vocab]
prompt = [BOS, TEXT] + ipa_ids + [GENERATE, AUDIO]
input_ids = torch.tensor([prompt], dtype=torch.long).to(device)

with torch.inference_mode():
    output = model.generate(
        input_ids,
        max_new_tokens=args.max_tokens,
        do_sample=True,
        temperature=1.0,
        eos_token_id=EOS,
    )

generated = output[0, len(prompt):].tolist()
if EOS in generated:
    generated = generated[:generated.index(EOS)]

audio = decode(generated)
sf.write(args.output, audio, SAMPLE_RATE)
print(f"saved to {args.output}")
