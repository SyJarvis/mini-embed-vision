#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用自训练 checkpoint 跑真实图文检索 demo。

从 imagenet-1k-zh-captions 验证集取 N 对图文：
  1) 以文搜图：每条描述在 N 张图里检索，看正确图排第几
  2) 以图搜文：每张图在 N 条描述里检索，看正确描述排第几
  3) 展示若干 query 的 top-k 结果（含是否命中）

用法:
    python example/retrieval_demo.py
    python example/retrieval_demo.py --checkpoint checkpoints_140k/best.pth --n 200
"""
import argparse
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, CLIPImageProcessor

from data import MultimodalParquetDataset, collate_fn, hash_source_id
from model import MiniEmbedVisionModel, MiniEmbedVisionConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints_140k/best.pth")
    p.add_argument("--data-dir", default="data/imagenet-1k-zh-captions")
    p.add_argument("--text-model", default="BAAI/bge-base-zh-v1.5")
    p.add_argument("--vision-model", default="openai/clip-vit-base-patch32")
    p.add_argument("--n", type=int, default=200, help="取多少对图文做 demo")
    p.add_argument("--show", type=int, default=5, help="展示多少个 query 的 top-k")
    p.add_argument("--topk", type=int, default=5)
    return p.parse_args()


@torch.no_grad()
def encode_all(model, dataset, tokenizer, image_processor, device, batch_size=64):
    """编码全部样本，返回 (text_embs[N,D], img_embs[N,D])，已 L2 归一化。"""
    from torch.utils.data import DataLoader
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_token_id=pad_token_id),
        num_workers=4, pin_memory=True,
    )
    text_embs, img_embs = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            t = model.encode_text(input_ids, attention_mask).float()
            i = model.encode_image(pixel_values).float()
        text_embs.append(t.cpu())
        img_embs.append(i.cpu())
    return torch.cat(text_embs, 0), torch.cat(img_embs, 0)


def recall(sim, k_list=(1, 5, 10)):
    """sim[i,j] = query i 与候选 j 的相似度，正样本在对角线。返回各 k 的 recall。"""
    ranks = sim.argsort(dim=1, descending=True)
    gt = torch.arange(sim.size(0)).unsqueeze(1)
    hits = (ranks == gt)
    out = {}
    for k in k_list:
        out[f"R@{k}"] = hits[:, :k].any(dim=1).float().mean().item()
    # median rank
    rank_pos = (ranks == gt).nonzero()[:, 1].float()
    out["median_rank"] = rank_pos.median().item()
    return out


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 加载模型 + checkpoint ===
    tokenizer = AutoTokenizer.from_pretrained(args.text_model, use_fast=True)
    image_processor = CLIPImageProcessor.from_pretrained(args.vision_model)
    config = MiniEmbedVisionConfig(
        text_model_name=args.text_model, vision_model_name=args.vision_model,
        freeze_text=True,
    )
    model = MiniEmbedVisionModel(config).to(device).eval()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded checkpoint: {args.checkpoint} (epoch={ckpt.get('epoch') if isinstance(ckpt,dict) else '?'})")
    if unexpected:
        print(f"  unexpected keys: {len(unexpected)} (可忽略)")
    if missing:
        print(f"  missing keys: {len(missing)}")

    # === 取验证集前 N 对（确定性，便于复现） ===
    ds = load_dataset("parquet", data_dir=args.data_dir, split="train")
    n_total = len(ds)
    val_indices = [i for i in range(n_total) if hash_source_id(ds[i]["source_id"]) % 10000 < 200][:args.n]
    # 包一层只暴露这 N 个样本的 dataset
    sub = ds.select(val_indices)

    class Wrap:
        def __init__(self, d): self.d = d; self.len = len(d)
        def __len__(self): return self.len
        def __getitem__(self, idx):
            row = self.d[idx]
            import io as _io
            img = row["image"]
            if not isinstance(img, __import__('PIL').Image.Image):
                img = __import__('PIL').Image.open(_io.BytesIO(img)).convert("RGB")
            tok = tokenizer(row["text"], truncation=True, max_length=128,
                            return_attention_mask=True, return_tensors="pt")
            pix = image_processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
            return {"input_ids": tok["input_ids"].squeeze(0),
                    "attention_mask": tok["attention_mask"].squeeze(0),
                    "pixel_values": pix,
                    "text": row["text"], "label_name": row.get("label_name", "")}

    wrap = Wrap(sub)
    print(f"\n取 {len(wrap)} 对验证图文做检索 demo")

    text_embs, img_embs = encode_all(model, wrap, tokenizer, image_processor, device)
    # text_embs, img_embs: [N, 768] 已归一化
    sim_t2i = text_embs @ img_embs.T   # 以文搜图
    sim_i2t = img_embs @ text_embs.T   # 以图搜文

    r_t2i = recall(sim_t2i)
    r_i2t = recall(sim_i2t)
    print("\n" + "="*60)
    print(f"{'指标':<10}{'以文搜图':>12}{'以图搜文':>12}")
    print("-"*60)
    for k in ["R@1", "R@5", "R@10", "median_rank"]:
        label = k if k == "median_rank" else k
        print(f"{label:<10}{r_t2i[k]:>12.4f}{r_i2t[k]:>12.4f}")
    print("="*60)

    # === 展示若干 query 的 top-k ===
    print(f"\n展示 {args.show} 个「以文搜图」query 的 top-{args.topk}：")
    for qi in range(min(args.show, len(wrap))):
        query_text = wrap[qi]["text"]
        order = sim_t2i[qi].argsort(descending=True)[:args.topk].tolist()
        hit = "✓命中" if order[0] == qi else f"✗ 正解排第{order.index(qi)+1}位"
        print(f"\n[Query #{qi}] {hit}")
        print(f"  查询描述: {query_text[:60]}")
        for rank, ci in enumerate(order, 1):
            mark = " ← 正解" if ci == qi else ""
            nm = wrap[ci]["label_name"] or ""
            print(f"  {rank}. [{nm[:25]}] {wrap[ci]['text'][:50]}{mark}")

    print(f"\n展示 {args.show} 个「以图搜文」query 的 top-{args.topk}：")
    for qi in range(min(args.show, len(wrap))):
        order = sim_i2t[qi].argsort(descending=True)[:args.topk].tolist()
        hit = "✓命中" if order[0] == qi else f"✗ 正解排第{order.index(qi)+1}位"
        nm = wrap[qi]["label_name"] or ""
        print(f"\n[Image #{qi}] {hit}  类别: {nm[:25]}")
        for rank, ci in enumerate(order, 1):
            mark = " ← 正解" if ci == qi else ""
            print(f"  {rank}. {wrap[ci]['text'][:55]}{mark}")


if __name__ == "__main__":
    main()
