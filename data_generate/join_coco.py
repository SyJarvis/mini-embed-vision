"""Join COCO 2017 detection parquet images with en→zh translations into a bilingual training set.

Input:
  - images-dir: detection-datasets/coco parquet shards (columns: image_id, image, ...)
  - translations: translate 子命令 --direction en2zh 产出的 JSONL (text_en + text_zh)

Output schema matches pack_bilingual exactly (image/text/text_en/label/label_name/source_id),
plus auto-generated dataset_infos.json and README.md in the standardized format
(numbers are computed from the files actually written, so they can never go stale).
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from tqdm.auto import tqdm

from .common import iter_jsonl, normalize_text
from .pack_bilingual import features, write_shard


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--images-dir", type=Path, required=True,
                        help="Directory with detection-datasets/coco train-*.parquet shards")
    parser.add_argument("--translations", type=Path, required=True,
                        help="translate 子命令 en2zh 产出的 JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--dataset-name", default="coco-2017-bilingual")
    parser.add_argument("--description", default=(
        "COCO 2017 train images with bilingual (Chinese/English) captions. "
        "One official English caption per image, machine-translated to Chinese."
    ))


def build_translation_index(path: Path) -> dict[str, tuple[str, str]]:
    index: dict[str, tuple[str, str]] = {}
    for _, record in iter_jsonl(path):
        if not isinstance(record, dict) or not record.get("ok"):
            continue
        source_id = record.get("source_id")
        zh, en = record.get("text_zh"), record.get("text_en")
        if isinstance(source_id, str) and isinstance(zh, str) and isinstance(en, str):
            zh, en = normalize_text(zh), normalize_text(en)
            if zh and en:
                index[source_id] = (zh, en)
    return index


def write_standard_metadata(output_dir: Path, dataset_name: str, description: str,
                            num_rows: int, num_shards: int) -> None:
    """Delegate to the shared writer with COCO-specific provenance."""
    from .common import write_standard_metadata as _write

    _write(
        output_dir, dataset_name, description, num_rows, num_shards,
        build_steps=(
            "1. 图片：detection-datasets/coco（COCO 2017 train）\n"
            "2. 英文 caption：ChristophSchuhmann/MS_COCO_2017_URL_TEXT，每图取第 1 句\n"
            "3. 中文：`python -m data_generate translate --direction en2zh`（LLM 机器翻译）\n"
            "4. 打包：`python -m data_generate join-coco`（本数据集）"
        ),
        license_yaml="license: other\nlicense_name: coco\nlicense_link: https://cocodataset.org/#termsofuse",
        license_note="\nCOCO 数据遵循 [COCO Terms of Use](https://cocodataset.org/#termsofuse)。",
    )


def run(args: argparse.Namespace) -> None:
    index = build_translation_index(args.translations)
    print({"translations": len(index)})

    shards = sorted(args.images_dir.glob("train-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No train-*.parquet under {args.images_dir}")

    total_images = sum(pq.read_metadata(s).num_rows for s in shards)
    shard_size = max(1, args.shard_size)
    expected_rows = min(total_images, len(index))
    num_shards = -(-expected_rows // shard_size)

    total = joined = 0
    batch: list[dict[str, Any]] = []
    shard_index = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.output_dir.glob("train-*.parquet"):
        stale.unlink()

    progress = tqdm(desc="Join COCO", unit="img", total=total_images)
    for shard_path in shards:
        table = pq.read_table(shard_path, columns=["image_id", "image"])
        for record in table.to_pylist():
            total += 1
            source_id = f"coco/2017/{record['image_id']:012d}"
            pair = index.get(source_id)
            if pair is None:
                progress.update(1)
                continue
            zh, en = pair
            image = record["image"]
            batch.append({
                "image": {"bytes": image.get("bytes"), "path": image.get("path")},
                "text": zh,
                "text_en": en,
                "label": -1,
                "label_name": "",
                "source_id": source_id,
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

    write_standard_metadata(args.output_dir, args.dataset_name, args.description,
                            num_rows=joined, num_shards=shard_index)
    print(
        {
            "output_dir": str(args.output_dir),
            "total_images": total,
            "joined_rows": joined,
            "missing_translation": total - joined,
            "num_shards": shard_index,
        }
    )
