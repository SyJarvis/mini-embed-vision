# -*- coding: utf-8
import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    get_linear_schedule_with_warmup,
    TrainingArguments
)
from torch.cuda.amp import GradScaler
from tqdm.auto import tqdm

from model import MiniEmbedVisionModel, MiniEmbedVisionConfig
from data import MultimodalDataset, collate_fn
from retrieval import compute_recall_at_k  


def train_epoch(model, dataloader, optimizer, scheduler, device, scaler, gradient_accumulation_steps=1, use_ddp=False):
    model.train()
    total_loss = 0.0
    num_steps = 0
    pbar = tqdm(dataloader, desc="Train", leave=False)
    
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                return_loss=True,
                gather_for_ddp=use_ddp  # 启用 DDP gather
            )
            loss = outputs.loss

        if loss is None:
            raise RuntimeError("Loss is None. Make sure both text and image are provided.")

        loss = loss / gradient_accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * gradient_accumulation_steps
        num_steps += 1
        pbar.set_postfix({"loss": total_loss / num_steps})

    return total_loss / num_steps


@torch.no_grad()
def evaluate_embeddings(model, dataloader, device):
    model.eval()
    text_embs = []
    img_embs = []
    for batch in tqdm(dataloader, desc="Eval", leave=False):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)

        text_emb = model.encode_text(input_ids, attention_mask)
        img_emb = model.encode_image(pixel_values)

        text_embs.append(text_emb.cpu())
        img_embs.append(img_emb.cpu())

    text_embs = torch.cat(text_embs, dim=0).numpy()
    img_embs = torch.cat(img_embs, dim=0).numpy()
    return text_embs, img_embs


def main():
    # === 配置 ===
    TEXT_MODEL = "BAAI/bge-base-zh-v1.5"
    VISION_MODEL = "openai/clip-vit-base-patch32"
    ROOT_DIR = "data/coco128"
    EMBED_DIM = 768
    BATCH_SIZE = 64
    EPOCHS = 100
    LR = 2e-5
    GRAD_ACCUM = 1
    USE_LORA = False
    FREEZE_TEXT = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_DIR = "./checkpoints"
    USE_DDP = torch.distributed.is_initialized()  # 自动检测 DDP

    # === Tokenizer & Processor ===
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, use_fast=True)
    image_processor = CLIPImageProcessor.from_pretrained(VISION_MODEL)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # === Dataset ===
    train_dataset = MultimodalDataset(
        root_dir=ROOT_DIR,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_text_len=128
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=(not USE_DDP),  # DDP 时不要 shuffle，由 sampler 处理
        collate_fn=lambda b: collate_fn(b, pad_token_id=pad_token_id),
        num_workers=4,
        pin_memory=True
    )
    val_loader = None

    # === Model ===
    config = MiniEmbedVisionConfig(
        embed_dim=EMBED_DIM,
        text_model_name=TEXT_MODEL,
        vision_model_name=VISION_MODEL,
        freeze_text=FREEZE_TEXT,
        use_lora=USE_LORA,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05
    )
    model = MiniEmbedVisionModel(config)
    model.to(DEVICE)

    # === Optimizer ===
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)

    # === Scheduler ===
    total_steps = math.ceil(len(train_loader) / GRAD_ACCUM) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(500, total_steps // 10),
        num_training_steps=total_steps
    )

    # === AMP ===
    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    # === Training Loop ===
    os.makedirs(SAVE_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        loss = train_epoch(
            model, train_loader, optimizer, scheduler, DEVICE, scaler,
            gradient_accumulation_steps=GRAD_ACCUM,
            use_ddp=USE_DDP
        )
        print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {loss:.4f}")

        if val_loader is not None:
            text_embs, img_embs = evaluate_embeddings(model, val_loader, DEVICE)
            recalls = compute_recall_at_k(text_embs, img_embs, k_list=[1, 5, 10])
            print("Validation recalls:", recalls)

        # === Save checkpoint ===
        if (epoch + 1) % 10 == 0:
            model.eval()
            ckp = os.path.join(SAVE_DIR, f"model_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckp)

if __name__ == "__main__":
    main()