from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from PIL import Image

from .common import JSONLRecord


def iter_manifest_records(manifest: Path) -> Iterator[JSONLRecord]:
    from .common import iter_jsonl

    for line_no, record in iter_jsonl(manifest):
        if not isinstance(record, dict):
            raise ValueError(f"Manifest row {line_no} is not an object")
        sample_id = record.get("id")
        image_path = record.get("image_path")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"Manifest row {line_no} is missing a non-empty id")
        if not isinstance(image_path, str) or not image_path:
            raise ValueError(f"Manifest row {line_no} is missing image_path")
        yield record


def iter_manifest_images(manifest: Path, image_root: Path | None = None) -> Iterator[JSONLRecord]:
    from .common import load_image

    root = image_root or manifest.parent
    for record in iter_manifest_records(manifest):
        path = Path(record["image_path"])
        if not path.is_absolute():
            path = root / path
        item = dict(record)
        item["image_path"] = str(path)
        item["image"] = load_image(path)
        yield item


def iter_arrow_images(source_dir: Path, split: str) -> Iterator[JSONLRecord]:
    from datasets import Dataset

    pattern = re.compile(rf"^imagenet-1k-{re.escape(split)}-\d+-of-\d+\.arrow$")
    shards = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and pattern.match(path.name)
    )
    if not shards:
        raise FileNotFoundError(f"No ImageNet {split} Arrow shards in {source_dir}")

    for shard_path in shards:
        dataset = Dataset.from_file(str(shard_path))
        label_feature = dataset.features.get("label")
        for row_index in range(len(dataset)):
            row = dataset[row_index]
            label = row.get("label")
            label_name = None
            if label_feature is not None and hasattr(label_feature, "int2str") and label is not None:
                try:
                    label_name = label_feature.int2str(label)
                except Exception:
                    label_name = None
            image = row.get("image")
            if not isinstance(image, Image.Image):
                raise TypeError(f"Unsupported image type at {shard_path}:{row_index}: {type(image)!r}")
            yield {
                "id": f"{split}/{shard_path.name}/{row_index}",
                "split": split,
                "shard": shard_path.name,
                "row": row_index,
                "label": label,
                "label_name": label_name,
                "image": image,
            }


def iter_images(source: str, source_dir: Path, manifest: Path | None, split: str) -> Iterator[JSONLRecord]:
    if source == "manifest":
        if manifest is None:
            raise ValueError("--manifest is required when --source manifest")
        yield from iter_manifest_images(manifest, image_root=source_dir)
    elif source == "imagenet-arrow":
        yield from iter_arrow_images(source_dir, split)
    else:
        raise ValueError(f"Unsupported source: {source}")


def load_arrow_image(source_dir: Path, shard: str, row: int) -> Image.Image:
    from datasets import Dataset

    path = source_dir / shard
    if not path.is_file():
        raise FileNotFoundError(f"Arrow shard not found: {path}")
    dataset = Dataset.from_file(str(path))
    if row < 0 or row >= len(dataset):
        raise IndexError(f"{shard}:{row} is out of range 0..{len(dataset) - 1}")
    image = dataset[row]["image"]
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise TypeError(f"Unsupported image type in {shard}:{row}: {type(image)!r}")
