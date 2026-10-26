from __future__ import annotations

import argparse
import math
from pathlib import Path

from tqdm.auto import tqdm

from .common import as_int, iter_jsonl, normalize_text


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory with original zh-captions train-*.parquet")
    parser.add_argument("--translations", type=Path, required=True,
                        help="JSONL from translate command")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for en-captions parquet")
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--limit", type=int, default=0)


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


def run(args: argparse.Namespace) -> None:
    from datasets import Dataset

    # Load translations into a dict
    print("Loading translations...")
    trans_map: dict[str, str] = {}
    for _, record in iter_jsonl(args.translations):
        sid = record.get("source_id")
        en_text = normalize_text(record.get("text_en"))
        if sid and en_text and record.get("ok"):
            trans_map[sid] = en_text
    print(f"  Loaded {len(trans_map)} translations")

    # Scan original shards
    shards = sorted(args.data_dir.glob("train-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No train-*.parquet files in {args.data_dir}")

    total_input = 0
    total_mapped = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Clean old output
    for stale in args.output_dir.glob("train-*.parquet"):
        stale.unlink()

    # Process shards and accumulate rows for output sharding
    all_rows: list[dict] = []
    for shard_path in tqdm(shards, desc="Scan shards", unit="shard"):
        ds = Dataset.from_parquet(str(shard_path))
        for row in ds:
            total_input += 1
            sid = row["source_id"]
            en_text = trans_map.get(sid)
            if en_text is None:
                continue
            all_rows.append({
                "image": row["image"],
                "text": en_text,
                "label": as_int(row.get("label")),
                "label_name": str(row.get("label_name") or ""),
                "source_id": sid,
            })
            total_mapped += 1
            if args.limit > 0 and total_mapped >= args.limit:
                break
        if args.limit > 0 and total_mapped >= args.limit:
            break

    print(f"  Input rows: {total_input}, Mapped: {total_mapped}")

    if not all_rows:
        raise ValueError("No rows mapped")

    # Write output shards
    shard_size = max(1, args.shard_size)
    num_shards = math.ceil(len(all_rows) / shard_size)
    written = 0

    for shard_idx in tqdm(range(num_shards), desc="Write shards", unit="shard"):
        start = shard_idx * shard_size
        end = min(start + shard_size, len(all_rows))
        batch = all_rows[start:end]
        ds = Dataset.from_list(batch, features=features())
        path = args.output_dir / f"train-{shard_idx:05d}-of-{num_shards:05d}.parquet"
        ds.to_parquet(str(path))
        written += len(batch)

    if written != len(all_rows):
        raise RuntimeError(f"Expected {len(all_rows)} rows, wrote {written}")

    print({"output_dir": str(args.output_dir), "num_rows": written, "num_shards": num_shards})
