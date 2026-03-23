import torch
from src.config import PAD, GENERATE
from src.constants import IGNORE_INDEX


def evaluate(model, loader, device) -> float:
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_ids, attention_mask in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            labels = input_ids.clone()
            for i in range(labels.size(0)):
                ids = input_ids[i].tolist()
                gen_pos = ids.index(GENERATE) if GENERATE in ids else len(ids)
                labels[i, :gen_pos + 1] = IGNORE_INDEX
            labels[input_ids == PAD] = IGNORE_INDEX

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

    return total_loss / len(loader)
