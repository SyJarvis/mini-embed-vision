"""Extract text-image pairs from MMEB-train subsets and emit a deduped to-translate workload.

Outputs:
  1. pairs parquet: {source_id, text_en, image_path, subset} — one row per unique
     (image, text) pair across selected subsets (instruction templates stripped)
  2. to-translate parquet: {source_id: mmeb/text/<sha1-12>, text} — unique texts only,
     ready for `python -m data_generate translate --direction en2zh`
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pyarrow.parquet as pq
from tqdm.auto import tqdm

# Per-subset harvesting rules, verified against actual data on 2026-10-26:
#   kind=i2t        → text = pos_text (clean caption), image = qry_image_path
#   kind=t2i        → text = qry minus PREFIX,          image = pos_image_path
#   kind=classify   → text = pos_text (label),          image = qry_image_path
SUBSET_RULES: dict[str, dict] = {
    "MSCOCO_i2t":      {"kind": "i2t"},
    "MSCOCO_t2i":      {"kind": "t2i", "prefix": "Find me an everyday image that matches the given caption: "},
    "VisualNews_i2t":  {"kind": "i2t"},
    "VisualNews_t2i":  {"kind": "t2i", "prefix": "Retrieve an image of this news caption. "},
    "SUN397":          {"kind": "classify"},
    "ImageNet_1K":     {"kind": "classify"},
    "VOC2007":         {"kind": "classify"},
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mmeb-dir", type=Path, required=True,
                        help="MMEB-train root (contains <subset>/train-*.parquet)")
    parser.add_argument("--output-pairs", type=Path, required=True,
                        help="Output pairs parquet (text-image metadata)")
    parser.add_argument("--output-translate", type=Path, required=True,
                        help="Output to-translate parquet (unique texts, translate.py compatible)")


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def run(args: argparse.Namespace) -> None:
    pairs: dict[tuple[str, str], dict[str, str]] = {}  # (image_path, text) -> row
    unmatched_prefix = 0

    for subset, rule in SUBSET_RULES.items():
        shard = args.mmeb_dir / subset / "train-00000-of-00001.parquet"
        if not shard.exists():
            raise FileNotFoundError(shard)
        table = pq.read_table(shard, columns=["qry", "qry_image_path", "pos_text", "pos_image_path"])
        kind = rule["kind"]
        prefix = rule.get("prefix", "")
        added = 0
        for r in tqdm(table.to_pylist(), desc=subset, unit="row"):
            if kind == "i2t":
                text, image = r["pos_text"], r["qry_image_path"]
            elif kind == "t2i":
                qry = r["qry"] or ""
                if prefix and not qry.startswith(prefix):
                    unmatched_prefix += 1
                    text = qry.split(":", 1)[-1].strip() if ":" in qry else qry
                else:
                    text = qry[len(prefix):] if prefix else qry
                image = r["pos_image_path"]
            else:  # classify
                text, image = r["pos_text"], r["qry_image_path"]
            text = (text or "").strip()
            image = (image or "").strip()
            if not text or not image:
                continue
            key = (image, text)
            if key not in pairs:
                pairs[key] = {
                    "source_id": f"mmeb/{subset}/{Path(image).stem}",
                    "text_en": text,
                    "image_path": image,
                    "subset": subset,
                }
                added += 1
        print({"subset": subset, "rows": table.num_rows, "unique_pairs_added": added})

    # pairs parquet
    import datasets as hfds
    pairs_rows = list(pairs.values())
    hfds.Dataset.from_list(pairs_rows).to_parquet(str(args.output_pairs))

    # unique-text translate workload
    seen: dict[str, str] = {}  # text -> key
    for row in pairs_rows:
        if row["text_en"] not in seen:
            seen[row["text_en"]] = text_key(row["text_en"])
    translate_rows = [{"source_id": f"mmeb/text/{k}", "text": t} for t, k in seen.items()]
    args.output_translate.parent.mkdir(parents=True, exist_ok=True)
    hfds.Dataset.from_list(translate_rows).to_parquet(str(args.output_translate))

    print({
        "pairs_total": len(pairs_rows),
        "unique_texts": len(translate_rows),
        "unmatched_prefix_rows": unmatched_prefix,
        "output_pairs": str(args.output_pairs),
        "output_translate": str(args.output_translate),
    })
