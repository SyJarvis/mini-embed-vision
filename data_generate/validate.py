from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .common import iter_jsonl, normalize_text


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("captions", type=Path)
    parser.add_argument("--min-chars", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--require-unique", action="store_true")


def run(args: argparse.Namespace) -> None:
    total = valid = 0
    ids: list[str] = []
    duplicate_count = 0
    errors: list[str] = []
    for line_no, record in iter_jsonl(args.captions):
        total += 1
        sample_id = record.get("id")
        text = normalize_text(record.get("text"))
        row_errors = []
        if not isinstance(sample_id, str) or not sample_id:
            row_errors.append("missing id")
        if not args.min_chars <= len(text) <= args.max_chars:
            row_errors.append(f"text length {len(text)}")
        if row_errors:
            errors.append(f"{line_no}: {'; '.join(row_errors)}")
        else:
            valid += 1
            if isinstance(sample_id, str):
                ids.append(sample_id)
    if args.require_unique:
        counts = Counter(ids)
        duplicate_count = sum(count - 1 for count in counts.values() if count > 1)
    print(
        {
            "file": str(args.captions),
            "total": total,
            "valid": valid,
            "invalid": total - valid,
            "duplicate_ids": duplicate_count,
            "first_errors": errors[:10],
        }
    )
    if total - valid or duplicate_count:
        raise SystemExit(1)
