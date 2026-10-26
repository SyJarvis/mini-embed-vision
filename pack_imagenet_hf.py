#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pack ImageNet-1k Chinese captions into a local Hugging Face parquet dataset."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from PIL import Image
from tqdm.auto import tqdm

DEFAULT_CAPTIONS = Path("data/imagenet-1k-captions/train_captions.jsonl")
DEFAULT_DATASET_DIR = Path("/mnt/data3/datasets/imagenet-1k")
DEFAULT_OUTPUT_DIR = Path("data/imagenet-1k-zh-captions")
DEFAULT_SHARD_SIZE = 2000
DEFAULT_JPEG_QUALITY = 95

README_TEMPLATE = """---
language:
- zh
task_categories:
- image-to-text
- image-text-to-text
size_categories:
- 10K<n<100K
pretty_name: ImageNet-1k Chinese Captions
---

# ImageNet-1k Chinese Captions

ImageNet-1k 子集的中文图像描述数据集。图片来自本地 ImageNet-1k Arrow 分片，描述由视觉语言模型生成。

## 加载

```python
from datasets import load_dataset

ds = load_dataset("data/imagenet-1k-zh-captions", split="train")
print(ds[0]["text"])
print(ds[0]["image"])
```

如果目录自动识别失败，改用：

```python
ds = load_dataset("parquet", data_dir="data/imagenet-1k-zh-captions", split="train")
```

## 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `image` | Image | 原始图像 |
| `text` | string | 中文描述 |
| `label` | int64 | ImageNet-1k 类别 ID |
| `label_name` | string | ImageNet-1k 类别名 |
| `source_id` | string | 原始样本 ID，格式 `split/shard/row` |

## 数据量

- 训练集: {num_rows} 条
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack ImageNet captions into a Hugging Face parquet dataset.")
    parser.add_argument("--captions", type=Path, default=DEFAULT_CAPTIONS)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    parser.add_argument("--limit", type=int, default=0, help="Pack at most N records. Use 0 for all.")
    return parser.parse_args()


def load_caption_records(path: Path, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
            text = (item.get("text") or "").strip()
            if not text:
                continue
            source_id = item.get("id")
            shard = item.get("shard")
            row = item.get("row")
            if not source_id or not shard or row is None:
                continue
            if source_id in seen:
                continue
            seen.add(source_id)
            records.append(
                {
                    "source_id": source_id,
                    "shard": shard,
                    "row": int(row),
                    "text": text,
                    "label": item.get("label"),
                    "label_name": item.get("label_name"),
                }
            )
            if limit > 0 and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"No valid caption records found in {path}")
    return records


def image_to_jpeg_bytes(image: Image.Image, quality: int) -> bytes:
    rgb = image.convert("RGB")
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def iter_packed_rows(
    records: list[dict[str, Any]],
    dataset_dir: Path,
    jpeg_quality: int,
) -> Iterator[dict[str, Any]]:
    from datasets import Dataset

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["shard"]].append(record)

    for shard_name in sorted(grouped):
        shard_path = dataset_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Arrow shard not found: {shard_path}")
        dataset = Dataset.from_file(str(shard_path))
        items = sorted(grouped[shard_name], key=lambda item: item["row"])
        for item in items:
            row_idx = item["row"]
            if row_idx < 0 or row_idx >= len(dataset):
                raise IndexError(f"{shard_name} row {row_idx} is out of range 0..{len(dataset) - 1}")
            image = dataset[row_idx]["image"]
            if not isinstance(image, Image.Image):
                image = Image.open(io.BytesIO(image)).convert("RGB") if isinstance(image, (bytes, bytearray)) else Image.fromarray(image)
            yield {
                "image": {"bytes": image_to_jpeg_bytes(image, jpeg_quality), "path": None},
                "text": item["text"],
                "label": -1 if item["label"] is None else int(item["label"]),
                "label_name": "" if item["label_name"] is None else str(item["label_name"]),
                "source_id": item["source_id"],
            }


def dataset_features():
    from datasets import Features, Image as HFImage, Value

    return Features(
        {
            "image": HFImage(),
            "text": Value("string"),
            "label": Value("int64"),
            "label_name": Value("string"),
            "source_id": Value("string"),
        }
    )


def write_parquet_shard(rows: list[dict[str, Any]], path: Path) -> None:
    from datasets import Dataset

    dataset = Dataset.from_list(rows, features=dataset_features())
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(path))


def write_dataset_infos(output_dir: Path, num_rows: int, num_shards: int) -> None:
    files = sorted(output_dir.glob(f"train-*-of-{num_shards:05d}.parquet"))
    dataset_size = sum(os.path.getsize(path) for path in files)
    payload = {
        "default": {
            "description": "ImageNet-1k subset with Chinese image captions.",
            "features": {
                "image": {"decode": True, "_type": "Image"},
                "text": {"dtype": "string", "_type": "Value"},
                "label": {"dtype": "int64", "_type": "Value"},
                "label_name": {"dtype": "string", "_type": "Value"},
                "source_id": {"dtype": "string", "_type": "Value"},
            },
            "splits": {
                "train": {
                    "name": "train",
                    "num_examples": num_rows,
                    "dataset_name": "imagenet-1k-zh-captions",
                }
            },
            "dataset_size": dataset_size,
            "num_rows": num_rows,
        }
    }
    path = output_dir / "dataset_infos.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_readme(output_dir: Path, num_rows: int) -> None:
    (output_dir / "README.md").write_text(README_TEMPLATE.format(num_rows=num_rows), encoding="utf-8")


def pack(args: argparse.Namespace) -> None:
    if not args.captions.is_file():
        raise SystemExit(f"Caption file not found: {args.captions}")
    if not args.dataset_dir.is_dir():
        raise SystemExit(f"Dataset directory not found: {args.dataset_dir}")

    records = load_caption_records(args.captions, args.limit)
    num_rows = len(records)
    shard_size = max(1, args.shard_size)
    num_shards = math.ceil(num_rows / shard_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for stale in args.output_dir.glob("train-*.parquet"):
        stale.unlink()

    print(f"Packing {num_rows} records into {num_shards} parquet shard(s)")
    batch: list[dict[str, Any]] = []
    shard_index = 0
    written = 0
    progress = tqdm(iter_packed_rows(records, args.dataset_dir, args.jpeg_quality), total=num_rows, desc="Pack")
    for row in progress:
        batch.append(row)
        if len(batch) < shard_size and written + len(batch) < num_rows:
            continue
        out_path = args.output_dir / f"train-{shard_index:05d}-of-{num_shards:05d}.parquet"
        write_parquet_shard(batch, out_path)
        written += len(batch)
        print(f"Wrote {out_path} ({len(batch)} rows)")
        shard_index += 1
        batch = []

    if written != num_rows:
        raise RuntimeError(f"Expected {num_rows} rows, wrote {written}")

    write_dataset_infos(args.output_dir, num_rows=num_rows, num_shards=num_shards)
    write_readme(args.output_dir, num_rows=num_rows)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "num_rows": num_rows,
                "num_shards": num_shards,
                "shard_size": shard_size,
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    pack(parse_args())


if __name__ == "__main__":
    main()
