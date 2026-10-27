"""Join MMEB-train pairs, en→zh translations and image files into a bilingual training set.

Inputs (produced by extract-mmeb + translate):
  - pairs parquet: {source_id, text_en, image_path, subset} (deduped by extract-mmeb)
  - translations jsonl: {source_id: mmeb/text/<hash>, text_zh, text_en} keyed by text hash
  - mmeb images root containing images/<subset>/Train/*.jpg

Output: standard bilingual schema (image/text/text_en/label/label_name/source_id)
plus auto-generated dataset_infos.json + README.md.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from tqdm.auto import tqdm

from .common import iter_jsonl, normalize_text, write_standard_metadata
from .pack_bilingual import write_shard


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pairs", type=Path, required=True, help="extract-mmeb 输出的 pairs parquet")
    parser.add_argument("--translations", type=Path, required=True, help="translate en2zh 输出的 JSONL")
    parser.add_argument("--images-root", type=Path, required=True,
                        help="MMEB-train 根目录（parquet 里的 image_path 为 images/... 相对路径）")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--dataset-name", default="mmeb-bilingual")
    parser.add_argument("--description", default=(
        "Bilingual (Chinese/English) text-image pairs harvested from TIGER-Lab/MMEB-train: "
        "MSCOCO and VisualNews retrieval captions, SUN397 scene names, ImageNet/VOC class names. "
        "Chinese side is machine-translated from English."
    ))
    parser.add_argument("--limit", type=int, default=0, help="Only pack first N pairs (0 = all)")


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def build_translation_index(path: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    for _, record in iter_jsonl(path):
        if not isinstance(record, dict) or not record.get("ok"):
            continue
        key = str(record.get("source_id", "")).removeprefix("mmeb/text/")
        zh = normalize_text(record.get("text_zh"))
        if key and zh:
            index[key] = zh
    return index


def run(args: argparse.Namespace) -> None:
    index = build_translation_index(args.translations)
    print({"translations": len(index)})

    pairs = pq.read_table(args.pairs).to_pylist()
    if args.limit > 0:
        pairs = pairs[: args.limit]
    print({"pairs": len(pairs)})

    shard_size = max(1, args.shard_size)
    num_shards = -(-len(pairs) // shard_size)

    total = joined = missing_tr = missing_img = 0
    batch: list[dict[str, Any]] = []
    shard_index = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.output_dir.glob("train-*.parquet"):
        stale.unlink()

    progress = tqdm(desc="Join MMEB", unit="pair", total=len(pairs))
    for pair in pairs:
        total += 1
        zh = index.get(text_key(pair["text_en"]))
        if zh is None:
            missing_tr += 1
            progress.update(1)
            continue
        image_file = args.images_root / pair["image_path"]
        if not image_file.exists():
            missing_img += 1
            progress.update(1)
            continue
        batch.append({
            "image": {"bytes": image_file.read_bytes(), "path": image_file.name},
            "text": zh,
            "text_en": normalize_text(pair["text_en"]),
            "label": -1,
            "label_name": "",
            "source_id": pair["source_id"],
        })
        joined += 1
        if len(batch) == shard_size:
            write_shard(batch, args.output_dir, shard_index, num_shards)
            shard_index += 1
            batch.clear()
        progress.update(1)
    if batch:
        write_shard(batch, args.output_dir, shard_index, num_shards)
        shard_index += 1
    progress.close()

    write_standard_metadata(
        args.output_dir, args.dataset_name, args.description,
        num_rows=joined, num_shards=shard_index,
        build_steps=(
            "1. 来源：TIGER-Lab/MMEB-train（子集：MSCOCO_i2t/t2i、VisualNews_i2t/t2i、SUN397、ImageNet_1K、VOC2007）\n"
            "2. 提取：`python -m data_generate extract-mmeb`（剥离指令模板，按 (图片, 文本) 去重）\n"
            "3. 中文：`python -m data_generate translate --direction en2zh`（LLM 机器翻译，仅翻唯一文本）\n"
            "4. 打包：`python -m data_generate join-mmeb`（本数据集）"
        ),
        license_yaml="license: apache-2.0",
        license_note="\n上游数据遵循各自许可（MMEB-train: apache-2.0；COCO 2014 遵循 COCO Terms of Use）。",
    )
    print(
        {
            "output_dir": str(args.output_dir),
            "total_pairs": total,
            "joined_rows": joined,
            "missing_translation": missing_tr,
            "missing_image": missing_img,
            "num_shards": shard_index,
        }
    )
