import torch
from src.config import AUDIO_OFFSET


def codes_to_flat(codes, idx: int) -> list[int]:
    """Extract depth-first flat raw codes for one item in a batch."""
    c0 = codes[0][idx].tolist()
    c1 = codes[1][idx].tolist()
    c2 = codes[2][idx].tolist()
    flat = []
    for i in range(len(c0)):
        flat.append(c0[i])
        flat.append(c1[i * 2])
        flat.append(c2[i * 4])
        flat.append(c2[i * 4 + 1])
        flat.append(c1[i * 2 + 1])
        flat.append(c2[i * 4 + 2])
        flat.append(c2[i * 4 + 3])
    return flat


def deinterleave(raw: list[int]) -> tuple[list, list, list]:
    """Split depth-first flat codes (no offset, no delay) back into c0/c1/c2."""
    n = len(raw) // 7
    c0, c1, c2 = [], [], []
    for i in range(n):
        base = i * 7
        c0.append(raw[base])
        c1.append(raw[base + 1])
        c2.append(raw[base + 2])
        c2.append(raw[base + 3])
        c1.append(raw[base + 4])
        c2.append(raw[base + 5])
        c2.append(raw[base + 6])
    return c0, c1, c2


def interleave(c0: list, c1: list, c2: list) -> list[int]:
    """Flatten raw integer code lists with delay pattern applied."""
    PAD = 0
    c1 = [PAD] + c1[:-1]
    c2 = [PAD, PAD] + c2[:-2]

    n = len(c0)
    flat = []
    for i in range(n):
        flat.append(AUDIO_OFFSET + c0[i])
        flat.append(AUDIO_OFFSET + c1[i * 2])
        flat.append(AUDIO_OFFSET + c2[i * 4])
        flat.append(AUDIO_OFFSET + c2[i * 4 + 1])
        flat.append(AUDIO_OFFSET + c1[i * 2 + 1])
        flat.append(AUDIO_OFFSET + c2[i * 4 + 2])
        flat.append(AUDIO_OFFSET + c2[i * 4 + 3])
    return flat


def tokens_to_codes(audio_ids: list[int]) -> list:
    """Undo delay and reconstruct 3 SNAC code levels from flat token list."""
    ids = [t - AUDIO_OFFSET for t in audio_ids]
    n = len(ids) // 7
    c0, c1, c2 = [], [], []
    for i in range(n):
        base = i * 7
        c0.append(ids[base])
        c1.append(ids[base + 1])
        c2.append(ids[base + 2])
        c2.append(ids[base + 3])
        c1.append(ids[base + 4])
        c2.append(ids[base + 5])
        c2.append(ids[base + 6])

    # undo delay
    c1 = c1[1:]
    c2 = c2[2:]
    n = min(len(c0), len(c1) // 2, len(c2) // 4)
    return [
        torch.tensor([c0[:n]]),
        torch.tensor([c1[:n * 2]]),
        torch.tensor([c2[:n * 4]]),
    ]
