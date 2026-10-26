"""Join the packed zh caption parquet with zh→en translations into a bilingual training set.

Output schema adds a ``text_en`` column to the standard pack schema; rows
without a translation keep an empty ``text_en`` and remain usable as zh-only
samples. Image bytes are copied verbatim (no re-encode).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from tqdm.auto import tqdm

from .common import iter_jsonl, normalize_text


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--parquet-dir", type=Path, required=True, help="Packed zh caption parquet directory")
    parser.add_argument("--translations", type=Path, required=True, help="translate 子命令产出的 zh→en JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--require-en", action="store_true", help="Drop rows without an English translation")


def build_translation_index(path: Path) -> dict[str, str]:
    index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for _, record in iter_jsonl(path):
        if not isinstance(record, dict) or not record.get("ok"):
            continue
        source_id = record.get("source_id")
        zh, en = record.get("text_zh"), record.get("text_en")
        if isinstance(source_id, str) and isinstance(zh, str) and isinstance(en, str):
            if normalize_text(zh) and normalize_text(en):
                index[source_id].append((normalize_text(zh), normalize_text(en)))
    return {sid: entries[0][1] for sid, entries in index.items()}


def features():
    from datasets import Features, Image as HFImage, Value

    return Features(
        {
            "image": HFImage(),
            "text": Value("string"),
            "text_en": Value("string"),
            "label": Value("int64"),
            "label_name": Value("string"),
            "source_id": Value("string"),
        }
    )


def row_from_arrow(row: dict[str, Any]) -> dict[str, Any]:
    image = row["image"]
    if not isinstance(image, dict):
        raise TypeError(f"Expected packed image struct, got {type(image)!r}")
    return {
        "image": {"bytes": image.get("bytes"), "path": image.get("path")},
        "text": normalize_text(row["text"]) or "",
        "text_en": "",
        "label": int(row.get("label") if row.get("label") is not None else -1),
        "label_name": str(row.get("label_name") or ""),
        "source_id": str(row["source_id"]),
    }


def run(args: argparse.Namespace) -> None:
    index = build_translation_index(args.translations)
    print({"translations": len(index)})

    shards = sorted(args.parquet_dir.glob("train-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No train-*.parquet under {args.parquet_dir}")

    total_rows = sum(pq.read_metadata(shard).num_rows for shard in shards)
    shard_size = max(1, args.shard_size)
    num_shards = -(-total_rows // shard_size)

    total = joined = dropped = 0
    batch: list[dict[str, Any]] = []
    shard_index = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.output_dir.glob("train-*.parquet"):
        stale.unlink()

    progress = tqdm(desc="Pack bilingual", unit="row", total=total_rows)
    for shard_path in shards:
        table = pq.read_table(shard_path)
        records = table.to_pylist()
        for record in records:
            total += 1
            row = row_from_arrow(record)
            text_en = index.get(row["source_id"])
            if text_en is None and args.require_en:
                dropped += 1
                progress.update(1)
                continue
            if text_en is not None:
                row["text_en"] = text_en
                joined += 1
            batch.append(row)
            progress.update(1)
            if len(batch) == shard_size:
                write_shard(batch, args.output_dir, shard_index, num_shards)
                shard_index += 1
                batch.clear()
    if batch:
        write_shard(batch, args.output_dir, shard_index, num_shards)
        shard_index += 1
    progress.close()
    print(
        {
            "output_dir": str(args.output_dir),
            "total_rows": total,
            "with_english": joined,
            "zh_only": total - joined - dropped,
            "dropped": dropped,
            "num_shards": shard_index,
        }
    )


def write_shard(rows: list[dict[str, Any]], output_dir: Path, shard_index: int, num_shards: int) -> None:
    from datasets import Dataset

    dataset = Dataset.from_list(rows, features=features())
    path = output_dir / f"train-{shard_index:05d}-of-{num_shards:05d}.parquet"
    dataset.to_parquet(str(path))
