from __future__ import annotations

import argparse
import io
import math
from pathlib import Path
from typing import Any, Iterator

from tqdm.auto import tqdm

from .common import as_int, iter_jsonl, load_image, normalize_text
from .sources import load_arrow_image


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--source", choices=["manifest", "imagenet-arrow"], required=True)
    parser.add_argument("--source-dir", type=Path, default=Path("."), help="Image root or ImageNet Arrow directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--limit", type=int, default=0)


def image_bytes(image, quality: int) -> dict[str, Any]:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    return {"bytes": buffer.getvalue(), "path": None}


def valid_caption(record: dict[str, Any]) -> bool:
    return isinstance(record.get("id"), str) and bool(normalize_text(record.get("text")))


def iter_rows(records: list[dict[str, Any]], args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    if args.source == "manifest":
        for record in records:
            path = Path(record["image_path"])
            if not path.is_absolute():
                path = args.source_dir / path
            yield {
                "image": image_bytes(load_image(path), args.jpeg_quality),
                "text": normalize_text(record["text"]),
                "label": as_int(record.get("label")),
                "label_name": str(record.get("label_name") or ""),
                "source_id": record["id"],
            }
        return

    for record in records:
        image = load_arrow_image(args.source_dir, record["shard"], as_int(record["row"]))
        yield {
            "image": image_bytes(image, args.jpeg_quality),
            "text": normalize_text(record["text"]),
            "label": as_int(record.get("label")),
            "label_name": str(record.get("label_name") or ""),
            "source_id": record["id"],
        }


def features():
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


def write_shard(rows: list[dict[str, Any]], path: Path) -> None:
    from datasets import Dataset

    dataset = Dataset.from_list(rows, features=features())
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(path))


def run(args: argparse.Namespace) -> None:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, record in iter_jsonl(args.captions):
        if not valid_caption(record) or record["id"] in seen:
            continue
        seen.add(record["id"])
        records.append(record)
        if args.limit > 0 and len(records) >= args.limit:
            break
    if not records:
        raise ValueError(f"No valid caption records in {args.captions}")
    if args.source == "manifest" and any("image_path" not in record for record in records):
        raise ValueError("Manifest caption records must contain image_path")
    if args.source == "imagenet-arrow" and any("shard" not in record or "row" not in record for record in records):
        raise ValueError("ImageNet caption records must contain shard and row")

    shard_size = max(1, args.shard_size)
    num_shards = math.ceil(len(records) / shard_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.output_dir.glob("train-*.parquet"):
        stale.unlink()

    batch: list[dict[str, Any]] = []
    written = 0
    shard_index = 0
    progress = tqdm(iter_rows(records, args), total=len(records), desc="Pack", unit="row")
    for row in progress:
        batch.append(row)
        if len(batch) == shard_size or written + len(batch) == len(records):
            path = args.output_dir / f"train-{shard_index:05d}-of-{num_shards:05d}.parquet"
            write_shard(batch, path)
            written += len(batch)
            shard_index += 1
            batch.clear()
    if written != len(records):
        raise RuntimeError(f"Expected {len(records)} rows, wrote {written}")
    print({"output_dir": str(args.output_dir), "num_rows": written, "num_shards": num_shards})
