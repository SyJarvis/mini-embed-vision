from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Iterator

from PIL import Image


JSONLRecord = dict[str, Any]


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip("'").strip('"')


def iter_jsonl(path: Path) -> Iterator[tuple[int, JSONLRecord]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc


def read_jsonl(path: Path) -> list[JSONLRecord]:
    return [record for _, record in iter_jsonl(path)]


def append_jsonl(path: Path, record: JSONLRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    for _, record in iter_jsonl(path):
        sample_id = record.get("id")
        if isinstance(sample_id, str) and sample_id:
            done.add(sample_id)
    return done


def load_image(reference: str | Path) -> Image.Image:
    path = Path(reference)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def resize_image(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def encode_image_b64(
    image: Image.Image,
    max_side: int = 768,
    jpeg_quality: int = 85,
) -> str:
    rgb = image.convert("RGB")
    resized = resize_image(rgb, max_side)
    buffer = io.BytesIO()
    resized.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    cleaned = text.strip().replace("```", "").strip().strip("\"'“”‘’")
    return " ".join(cleaned.split())


def as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def write_standard_metadata(output_dir: Path, dataset_name: str, description: str,
                            num_rows: int, num_shards: int, build_steps: str,
                            license_yaml: str = "license: apache-2.0",
                            license_note: str = "") -> None:
    """Generate dataset_infos.json + README.md from the shards actually on disk.

    All numbers (rows, shards, byte size) are computed from the written files,
    so the metadata can never go stale. ``build_steps`` is a markdown fragment
    describing how the dataset was produced.
    """
    import json

    shards = sorted(output_dir.glob("train-*.parquet"))
    dataset_size = sum(s.stat().st_size for s in shards)
    from .pack_bilingual import features

    features_dict = features().to_dict()
    infos = {
        dataset_name: {
            "description": description,
            "features": features_dict,
            "splits": {
                "train": {
                    "name": "train",
                    "num_bytes": dataset_size,
                    "num_examples": num_rows,
                    "dataset_name": dataset_name,
                }
            },
            "download_size": dataset_size,
            "dataset_size": dataset_size,
        }
    }
    (output_dir / "dataset_infos.json").write_text(
        json.dumps(infos, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    size_mb = dataset_size / 2**20
    readme = f"""---
{license_yaml}
task_categories:
- image-to-text
language:
- zh
- en
tags:
- image-captioning
- bilingual
- multimodal-embedding
size_categories:
- 100K<n<1M
---

# {dataset_name}

{description}

## 数据规模

| 项 | 值 |
|----|----|
| 行数 | {num_rows:,} |
| 分片数 | {num_shards} |
| 总体积 | {size_mb:,.0f} MB |

## 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| image | Image | 原始图片字节 |
| text | string | 中文描述（英文文本机器翻译） |
| text_en | string | 英文文本 |
| label | int64 | 恒为 -1 |
| label_name | string | 空字符串 |
| source_id | string | 数据溯源 ID |

## 构建流程

{build_steps}

## 加载方式

```python
from datasets import load_dataset
ds = load_dataset("parquet", data_dir="{output_dir}", split="train")
```
{license_note}
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
