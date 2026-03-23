import os
import torch
from tqdm import tqdm

from src.model import get_model
from src.data import get_loader, get_val_loader
from torch.optim.lr_scheduler import OneCycleLR
from src.config import PAD, GENERATE, EOS, args
from src.constants import IGNORE_INDEX
from src.eval import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

loader = get_loader()
val_loader = get_val_loader()

model = get_model().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
total_steps = args.epochs * len(loader)
scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps,
                       pct_start=args.warmup_pct)
os.makedirs(args.output, exist_ok=True)

for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    for input_ids, attention_mask in tqdm(loader, desc=f"epoch {epoch+1}", leave=True):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # labels: same as input_ids but mask everything up to and including <generate>
        labels = input_ids.clone()
        for i in range(labels.size(0)):
            ids = input_ids[i].tolist()
            gen_pos = ids.index(GENERATE) if GENERATE in ids else len(ids)
            labels[i, :gen_pos + 1] = IGNORE_INDEX  # ignore prompt tokens in loss
        labels[input_ids == PAD] = IGNORE_INDEX

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    val_loss = evaluate(model, val_loader, device)
    current_lr = scheduler.get_last_lr()[0]
    print(f"epoch {epoch+1} loss: {avg_loss:.4f} val_loss: {val_loss:.4f} lr: {current_lr:.2e}")

    if (epoch + 1) % args.save_every == 0:
        ckpt_path = os.path.join(args.output, f"epoch_{epoch+1}")
        model.save_pretrained(ckpt_path)
        print(f"saved checkpoint to {ckpt_path}")
