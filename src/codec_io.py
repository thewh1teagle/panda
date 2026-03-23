import torch
import numpy as np
import librosa
from snac import SNAC

from src.constants import SAMPLE_RATE
from src.codec import codes_to_flat

_model = None
_device = None


def get_snac(device: str = "cpu"):
    global _model, _device
    if _model is None or _device != device:
        _model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)
        _device = device
    return _model



def wav_to_codes(wav_path: str, device: str = "cpu") -> list[int]:
    """Load wav, encode with SNAC, return flat raw codes (no delay, no offset)."""
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    tensor = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        codes = get_snac(device).encode(tensor)
    return codes_to_flat(codes, 0)


def batch_wav_to_codes(wav_paths: list[str], device: str = "cpu") -> list[list[int]]:
    """Load and encode a batch of wavs, return list of flat raw codes."""
    audios = [librosa.load(p, sr=SAMPLE_RATE, mono=True)[0] for p in wav_paths]
    max_len = max(a.shape[0] for a in audios)
    padded = np.zeros((len(audios), 1, max_len), dtype=np.float32)
    for i, a in enumerate(audios):
        padded[i, 0, :a.shape[0]] = a
    tensor = torch.from_numpy(padded).to(device)
    with torch.inference_mode():
        codes = get_snac(device).encode(tensor)
    return [codes_to_flat(codes, i) for i in range(len(wav_paths))]


def decode(audio_ids: list[int]) -> np.ndarray:
    """Reconstruct audio from flat token list."""
    from src.codec import tokens_to_codes
    codes = tokens_to_codes(audio_ids)
    with torch.inference_mode():
        audio = get_snac().decode(codes)
    return audio[0, 0].numpy()
