# -*- coding: utf-8 -*-
"""mini-embed-vision 训练脚本（ImageNet-1k 中文描述数据集）。

单卡:
    python train.py
多卡 DDP:
    torchrun --nproc_per_node=3 train.py --batch_size 128
冒烟测试:
    python train.py --smoke
"""
import argparse
import math
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    get_linear_schedule_with_warmup,
)
from torchvision import transforms
from tqdm.auto import tqdm

from data import MultimodalParquetDataset, collate_fn
from model import MiniEmbedVisionModel, MiniEmbedVisionConfig
from retrieval import compute_recall_at_k


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/imagenet-1k-zh-captions")
    p.add_argument("--text-model", default="BAAI/bge-base-zh-v1.5")
    p.add_argument("--vision-model", default="openai/clip-vit-base-patch32")
    p.add_argument("--vision-backend", choices=["transformers", "timm"], default="transformers")
    p.add_argument("--timm-model-name", default="eva02_base_patch14_224")
    p.add_argument("--embed-dim", type=int, default=768)
    p.add_argument("--batch-size", type=int, default=32, help="每卡 batch size")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-text-len", type=int, default=512)
    p.add_argument("--val-ratio", type=float, default=0.02)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--save-dir", default="./checkpoints")
    p.add_argument("--save-every", type=int, default=1, help="每 N 个 epoch 保存一次")
    p.add_argument("--val-every", type=int, default=1, help="每 N 个 epoch 验证一次")
    p.add_argument("--patience", type=int, default=3, help="Early stop：R@1 连续 N 轮无提升则停。0=关闭")
    p.add_argument("--use-lora", action="store_true")
    p.add_argument("--enable-reranker", action="store_true")
    p.add_argument("--reranker-hidden-dim", type=int, default=256)
    p.add_argument("--reranker-dropout", type=float, default=0.1)
    p.add_argument("--reranker-loss-weight", type=float, default=0.2)
    p.add_argument("--wandb", action="store_true", help="Enable W&B logging on rank 0")
    p.add_argument("--wandb-project", default="mini-embed-vision")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-run-id", default=None, help="Stable W&B run id; reuse it when resuming training")
    p.add_argument("--resume", default=None, help="恢复训练的 checkpoint 路径")
    p.add_argument("--smoke", action="store_true", help="冒烟测试：少量数据跑 1 epoch")
    return p.parse_args()


def setup_distributed():
    """torchrun 会注入 RANK/WORLD_SIZE 环境变量。"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, True
    return 0, 0, 1, False


def is_master(rank):
    return rank == 0


def print_rank0(*args, **kwargs):
    if is_master(int(os.environ.get("RANK", 0))):
        print(*args, **kwargs)


def setup_wandb(args, rank):
    if not args.wandb or not is_master(rank):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is enabled but not installed. Install it with: pip install wandb") from exc

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        id=args.wandb_run_id,
        name=args.wandb_run_name,
        config=vars(args),
        resume="allow",
        save_code=False,
    )
    return run


def build_model(args, device):
    config = MiniEmbedVisionConfig(
        embed_dim=args.embed_dim,
        text_model_name=args.text_model,
        vision_model_name=args.vision_model,
        vision_backend=args.vision_backend,
        timm_model_name=args.timm_model_name,
        enable_reranker=args.enable_reranker,
        reranker_hidden_dim=args.reranker_hidden_dim,
        reranker_dropout=args.reranker_dropout,
        reranker_loss_weight=args.reranker_loss_weight,
        freeze_text=True,
        use_lora=args.use_lora,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
    )
    model = MiniEmbedVisionModel(config)
    return model.to(device)


def train_one_epoch(model, dataloader, optimizer, scheduler, device, scaler,
                    gradient_accumulation_steps=1, use_ddp=False, epoch=0, log_every=20,
                    args=None, wandb_run=None):
    model.train()
    total_loss = 0.0
    num_steps = 0
    pbar = tqdm(dataloader, desc=f"Train E{epoch+1}", leave=False, disable=not is_master(int(os.environ.get("RANK", 0))))
    optimizer.zero_grad()

    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=scaler is not None):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                return_loss=True,
                return_rerank_loss=args.enable_reranker,
                gather_for_ddp=use_ddp,
            )
            loss = outputs.loss

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
        rerank_loss = float(outputs.rerank_loss.detach()) if outputs.rerank_loss is not None else None
        num_steps += 1
        if num_steps % log_every == 0 and is_master(int(os.environ.get("RANK", 0))):
            pbar.set_postfix({"loss": total_loss / num_steps, "lr": scheduler.get_last_lr()[0]})
            if wandb_run is not None:
                global_step = epoch * len(dataloader) + step + 1
                metrics = {
                    "train/loss": total_loss / num_steps,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                }
                if rerank_loss is not None:
                    metrics["train/rerank_loss"] = rerank_loss
                wandb_run.log(metrics, step=global_step)

    return total_loss / max(num_steps, 1)


@torch.no_grad()
def evaluate_embeddings(model, dataloader, device):
    model.eval()
    text_embs = []
    img_embs = []
    for batch in tqdm(dataloader, desc="Eval", leave=False, disable=not is_master(int(os.environ.get("RANK", 0)))):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)

        encode_text = getattr(model, "encode_text", None) or model.module.encode_text
        encode_image = getattr(model, "encode_image", None) or model.module.encode_image
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            text_emb = encode_text(input_ids, attention_mask)
            img_emb = encode_image(pixel_values)

        text_embs.append(text_emb.float().cpu())
        img_embs.append(img_emb.float().cpu())

    text_embs = torch.cat(text_embs, dim=0).numpy()
    img_embs = torch.cat(img_embs, dim=0).numpy()
    return text_embs, img_embs


def main():
    args = parse_args()
    rank, local_rank, world_size, use_ddp = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    print_rank0(f"device={device}, world_size={world_size}, use_ddp={use_ddp}")
    wandb_run = setup_wandb(args, rank)

    # === Tokenizer & Processor ===
    tokenizer = AutoTokenizer.from_pretrained(args.text_model, use_fast=True)
    if args.vision_backend == "timm":
        image_processor = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
    else:
        image_processor = CLIPImageProcessor.from_pretrained(args.vision_model)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # === Dataset ===
    train_dataset = MultimodalParquetDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_text_len=args.max_text_len,
        split="train",
        val_ratio=args.val_ratio,
    )
    val_dataset = MultimodalParquetDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_text_len=args.max_text_len,
        split="val",
        val_ratio=args.val_ratio,
    )
    print_rank0(f"train={len(train_dataset)}, val={len(val_dataset)}")

    if args.smoke:
        train_dataset.ds = train_dataset.ds.select(range(min(512, len(train_dataset))))
        val_dataset.ds = val_dataset.ds.select(range(min(64, len(val_dataset))))
        print_rank0(f"[smoke] train={len(train_dataset)}, val={len(val_dataset)}")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=lambda b: collate_fn(b, pad_token_id=pad_token_id),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    # 验证只在 rank0 上做，无需 sampler
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_token_id=pad_token_id),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # === Model ===
    model = build_model(args, device)

    if use_ddp:
        # 冻结参数不产生梯度，DDP 需要放宽 unused-parameter 检查
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # === Optimizer（只优化可训练参数）===
    core = model.module if use_ddp else model
    trainable_params = [p for p in core.parameters() if p.requires_grad]
    print_rank0(f"trainable params: {sum(p.numel() for p in trainable_params):,}")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(10, int(total_steps * args.warmup_ratio)),
        num_training_steps=total_steps,
    )

    # bf16 autocast 不需要 GradScaler
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    scaler = None
    print_rank0(f"bf16 autocast: {use_bf16}")

    # === Resume ===
    start_epoch = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        core.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"]
        print_rank0(f"resumed from {args.resume} (epoch {start_epoch})")

    # === Training Loop ===
    os.makedirs(args.save_dir, exist_ok=True)
    best_recall = -1.0
    epochs_no_improve = 0
    best_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        if use_ddp:
            train_sampler.set_epoch(epoch)
        loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, scaler,
            gradient_accumulation_steps=args.grad_accum,
            use_ddp=use_ddp,
            epoch=epoch,
            args=args,
            wandb_run=wandb_run,
        )
        print_rank0(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {loss:.4f} | lr: {scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % args.val_every == 0:
            text_embs, img_embs = evaluate_embeddings(model, val_loader, device)
            recalls = compute_recall_at_k(text_embs, img_embs, k_list=[1, 5, 10])
            print_rank0(f"Epoch {epoch+1} | Val Recall@1/5/10: {recalls}")
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "val/R@1": recalls["R@1"],
                        "val/R@5": recalls["R@5"],
                        "val/R@10": recalls["R@10"],
                        "epoch": epoch + 1,
                    },
                    step=(epoch + 1) * len(train_loader),
                )

            if use_ddp:
                dist.barrier()

            if is_master(rank):
                r1 = recalls.get("R@1", 0.0) if isinstance(recalls, dict) else recalls[0]
                if r1 > best_recall:
                    best_recall = r1
                    best_epoch = epoch + 1
                    epochs_no_improve = 0
                    path = os.path.join(args.save_dir, "best.pth")
                    torch.save({
                        "model": core.state_dict(),
                        "epoch": epoch + 1,
                        "recall": recalls,
                    }, path)
                    if wandb_run is not None:
                        wandb_run.summary["best/R@1"] = best_recall
                        wandb_run.summary["best/epoch"] = best_epoch
                    print_rank0(f"saved best → {path} (R@1={best_recall:.4f})")
                else:
                    epochs_no_improve += 1
                    print_rank0(f"no improve ({epochs_no_improve}/{args.patience}) | best R@1={best_recall:.4f} @ E{best_epoch}")
            if use_ddp:
                # rank0 决定是否 early stop，broadcast 同步给所有 rank（避免 DDP 死锁）
                if is_master(rank):
                    should_stop = torch.tensor([1.0 if (args.patience > 0 and epochs_no_improve >= args.patience) else 0.0], device=device)
                else:
                    should_stop = torch.tensor([0.0], device=device)
                dist.broadcast(should_stop, src=0)
                if should_stop.item() >= 1.0:
                    print_rank0(f"Early stop at epoch {epoch+1} (no R@1 improve for {args.patience} epochs). best R@1={best_recall:.4f} @ E{best_epoch}")
                    break

        if (epoch + 1) % args.save_every == 0 and is_master(rank):
            path = os.path.join(args.save_dir, f"model_epoch{epoch+1}.pth")
            torch.save({
                "model": core.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
            }, path)
            print_rank0(f"saved → {path}")

    if use_ddp:
        dist.destroy_process_group()
    if wandb_run is not None:
        wandb_run.finish()
    print_rank0("done.")


if __name__ == "__main__":
    main()
