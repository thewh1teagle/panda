import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from src.config import PAD, args


def collate(batch):
    seqs = [torch.tensor(x["input_ids"]) for x in batch]
    max_len = max(s.size(0) for s in seqs)
    input_ids = torch.full((len(seqs), max_len), PAD, dtype=torch.long)
    attention_mask = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, :s.size(0)] = s
        attention_mask[i, :s.size(0)] = 1
    return input_ids, attention_mask


def get_loader():
    ds = load_from_disk(args.dataset + "/train")
    return DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)


def get_val_loader():
    ds = load_from_disk(args.dataset + "/val")
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
