#!/usr/bin/env python3
"""
Translate English benchmark evaluation data to Chinese.

Tasks:
1. ImageNet-1k class names (1,000 English -> Chinese)
2. Flickr30k T2I queries (5,000 English -> Chinese)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data_generate.common import JSONLRecord, append_jsonl, load_done_ids, load_dotenv, normalize_text


def extract_zh(text: str) -> str:
    """Extract the Chinese translation from model reasoning - find the best candidate."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return ""

    candidates = []

    for line in lines:
        skip_words = ["\u6211\u4eec", "\u9700\u8981", "\u6ce8\u610f", "\u8bf4\u660e", "\u6807\u51c6", "\u7528\u6237"]
        if any(line.startswith(w) for w in skip_words):
            continue

        chinese = sum(1 for c in line if "\u4e00" <= c <= "\u9fff")
        if chinese < 2:
            continue

        # Check for quoted translation
        quotes = re.findall(r'"([^"]*[\u4e00-\u9fff][^"]*)"', line)
        for q in quotes:
            q_clean = q.strip()
            q_chinese = sum(1 for c in q_clean if "\u4e00" <= c <= "\u9fff")
            if q_chinese >= 2:
                candidates.append((q_clean, q_chinese, len(q_clean)))

        # Check after known markers
        for marker in ["Translation: ", "\u7ffb\u8bd1\uff1a", "translation: ", "Possible translation: "]:
            idx = line.find(marker)
            if idx >= 0:
                after = line[idx + len(marker):].strip().strip("'\"\u201c\u201d\u2018\u2019")
                after_chinese = sum(1 for c in after if "\u4e00" <= c <= "\u9fff")
                if after_chinese >= 2:
                    candidates.append((after, after_chinese, len(after)))

        # The raw line itself
        cleaned = line.strip("'\"\u201c\u201d\u2018\u2019\uff0c\u3002\uff01\uff1f\uff1b\uff1a\u3001 ")
        cleaned_chinese = sum(1 for c in cleaned if "\u4e00" <= c <= "\u9fff")
        if cleaned_chinese >= 2 and cleaned_chinese / max(len(cleaned), 1) > 0.2:
            candidates.append((cleaned, cleaned_chinese, len(cleaned)))

    if not candidates:
        return ""

    candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return candidates[0][0]
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", choices=["flickr", "imagenet", "all"], default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--model", default="DeepSeek-V4-Flash-0731")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")


def build_client(args: argparse.Namespace):
    from openai import OpenAI

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise ValueError(f"{args.api_key_env} is not set")
    if not args.base_url:
        raise ValueError("--base-url is required or set OPENAI_BASE_URL")
    return OpenAI(base_url=args.base_url, api_key=api_key, timeout=args.timeout)


def request_translate(client, args: argparse.Namespace, en_text: str) -> str:
    last_error: Exception | None = None
    for attempt in range(args.retries + 1):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": en_text},
                ],
                temperature=0,
                max_tokens=args.max_tokens,
            )
            msg = response.choices[0].message

            # Try content first, then reasoning
            zh = extract_zh(msg.content or "") or extract_zh(
                getattr(msg, "reasoning", "") or ""
            )
            if zh:
                return zh

            raise ValueError(
                f"Empty response: content={repr(msg.content[:100] if msg.content else None)}"
            )
        except Exception as exc:
            last_error = exc
            if attempt >= args.retries:
                break
            time.sleep(min(30.0, 2.0 * (2**attempt)))
    raise RuntimeError(str(last_error) if last_error else "Unknown error")


def process_one(item_id: str, en_text: str, client, args) -> JSONLRecord:
    record = {"id": item_id}
    try:
        if args.dry_run:
            record.update({"ok": True, "en": en_text, "zh": ""})
            return record
        zh = request_translate(client, args, en_text)
        record.update({"ok": True, "en": en_text, "zh": zh})
        return record
    except Exception as exc:
        record.update({"ok": False, "en": en_text, "error": str(exc)})
        return record


def translate_items(items: list[tuple[str, str]], args: argparse.Namespace, desc: str) -> dict[str, str]:
    output_path = args.output_dir / f"translations_{desc}.jsonl"
    error_path = output_path.with_name(output_path.name + ".errors.jsonl")
    done_ids = load_done_ids(output_path)
    client = None if args.dry_run else build_client(args)

    pending = [(iid, txt) for iid, txt in items if iid not in done_ids]
    if not pending:
        print(f"[{desc}] All {len(items)} items already translated")
        from data_generate.common import read_jsonl
        return {rec["id"]: rec["zh"] for rec in read_jsonl(output_path) if rec.get("ok")}

    print(f"[{desc}] {len(pending)}/{len(items)} items to translate")
    success = failed = 0
    consecutive_failures = 0
    stop = False
    write_lock = threading.Lock()

    progress = tqdm(total=len(pending), desc=f"Translate {desc}", unit="req")
    max_in_flight = max(1, args.concurrency) * 4

    def handle(future: Future[JSONLRecord]) -> bool:
        nonlocal success, failed, consecutive_failures
        record = future.result()
        progress.update(1)
        if record.get("ok"):
            with write_lock:
                append_jsonl(output_path, record)
            success += 1
            consecutive_failures = 0
        else:
            with write_lock:
                append_jsonl(error_path, record)
            failed += 1
            consecutive_failures += 1
        progress.set_postfix(ok=success, fail=failed)
        return consecutive_failures >= 10

    in_flight: list[Future[JSONLRecord]] = []
    idx = 0
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        while idx < len(pending) and not stop:
            while len(in_flight) < max_in_flight and idx < len(pending):
                iid, txt = pending[idx]
                in_flight.append(pool.submit(process_one, iid, txt, client, args))
                idx += 1
            for future in as_completed(in_flight):
                in_flight.remove(future)
                if handle(future):
                    stop = True
                    break
        for future in as_completed(in_flight):
            if handle(future):
                stop = True
                break
        if stop:
            for future in in_flight:
                future.cancel()
    progress.close()

    from data_generate.common import read_jsonl
    result = {rec["id"]: rec["zh"] for rec in read_jsonl(output_path) if rec.get("ok")}
    print(f"[{desc}] Done: {success} ok, {failed} failed. Total: {len(result)}")
    return result


def translate_imagenet(args: argparse.Namespace):
    from datasets import Dataset

    data_dir = Path("/mnt/data3/datasets/imagenet-1k")
    shards = sorted(data_dir.glob("imagenet-1k-validation-*.arrow"))
    if not shards:
        raise FileNotFoundError(f"No ImageNet validation shards in {data_dir}")
    ds = Dataset.from_file(str(shards[0]))
    class_names = ds.features["label"].names
    print(f"[ImageNet] {len(class_names)} class names")

    items = [(str(i), name) for i, name in enumerate(class_names)]
    translations = translate_items(items, args, "imagenet")

    out_dir = args.output_dir / "imagenet-1k-zh"
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = {}
    for i, name in enumerate(class_names):
        zh = translations.get(str(i), "")
        mapping[str(i)] = {"en": name, "zh": zh}

    json_path = out_dir / "class_names_zh.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    jsonl_path = out_dir / "class_names_zh.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for i, name in enumerate(class_names):
            f.write(json.dumps({"index": i, "en": name, "zh": translations.get(str(i), "")}, ensure_ascii=False) + "\n")

    print(f"[ImageNet] Output: {json_path}")
    for i in range(5):
        print(f"  EN: {class_names[i]}")
        print(f"  ZH: {mapping[str(i)]['zh']}")
        print()


def translate_flickr(args: argparse.Namespace):
    from datasets import Dataset

    cache_dir = Path.home() / ".cache/huggingface/datasets/mteb___flickr30kt2i"
    revision_dirs = list((cache_dir / "query" / "0.0.0").iterdir())
    if not revision_dirs:
        ds = Dataset.from_dataset("mteb/flickr30kt2i", "query", trust_remote_code=True)
    else:
        arrows = list(revision_dirs[0].glob("*.arrow"))
        assert arrows, "No arrow files for Flickr queries"
        ds = Dataset.from_file(str(arrows[0]))

    print(f"[Flickr] {len(ds)} queries loaded")
    items = [(row["id"], row["text"]) for row in ds]
    translations = translate_items(items, args, "flickr")

    out_dir = args.output_dir / "flickr30k-zh"
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "queries_zh.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in ds:
            f.write(json.dumps({
                "id": rec["id"], "en": rec["text"], "zh": translations.get(rec["id"], "")
            }, ensure_ascii=False) + "\n")

    zh_texts = [translations.get(row["id"], "") for row in ds]
    zh_ds = ds.add_column("text_zh", zh_texts)
    parquet_path = out_dir / "queries_zh.parquet"
    zh_ds.to_parquet(str(parquet_path))

    with (out_dir / "id_to_zh.json").open("w", encoding="utf-8") as f:
        json.dump({row["id"]: translations.get(row["id"], "") for row in ds}, f, ensure_ascii=False, indent=2)

    print(f"[Flickr] Output: {jsonl_path}")
    print(f"[Flickr] Parquet: {parquet_path}")
    for s in ds[:3]:
        print(f"  EN: {s['text'][:80]}")
        print(f"  ZH: {translations.get(s['id'], '')[:80]}")
        print()


def run(args: argparse.Namespace) -> None:
    load_dotenv()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.task in ("imagenet", "all"):
        translate_imagenet(args)
    if args.task in ("flickr", "all"):
        translate_flickr(args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate benchmark evaluation data to Chinese")
    add_arguments(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
