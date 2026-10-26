from __future__ import annotations

import argparse
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator

from tqdm.auto import tqdm

from .common import JSONLRecord, append_jsonl, load_done_ids, load_dotenv, normalize_text


TRANSLATE_PROMPT = "Translate the following Chinese image caption to English. Keep the original objective, factual style describing only visible content. Output only the translation, no explanations, no quotes, no extra text.\n\n"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing train-*.parquet shards")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSONL path for translations")
    parser.add_argument("--error-output", type=Path)
    parser.add_argument("--model", default="DeepSeek-V4-Flash-0731")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-wait", type=float, default=2.0)
    parser.add_argument("--fail-fast", type=int, default=20,
                        help="Stop after N consecutive failures; 0 disables")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N items; 0 means all")
    parser.add_argument("--dry-run", action="store_true",
                        help="Scan inputs without calling the API")


def build_client(args: argparse.Namespace):
    from openai import OpenAI

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise ValueError(f"{args.api_key_env} is not set")
    if not args.base_url:
        raise ValueError("--base-url is required or set OPENAI_BASE_URL")
    return OpenAI(base_url=args.base_url, api_key=api_key, timeout=args.timeout)


def extract_translation(content: str | None, reasoning: str | None) -> str:
    raw = normalize_text(content) if content else ""

    if raw:
        return raw

    raw_reasoning = normalize_text(reasoning) if reasoning else ""
    if not raw_reasoning:
        raise ValueError("Empty response from model")

    # Strategy 1: Find line starting with "翻译："
    for line in raw_reasoning.split("\n"):
        line = line.strip()
        idx = line.find("翻译：")
        if idx >= 0:
            candidate = line[idx + 3:].strip().strip('"\'„"”“')
            for sep in [" 或者 ", " 或 "]:
                if sep in candidate:
                    candidate = candidate.split(sep)[0]
            candidate = normalize_text(candidate)
            if candidate and len(candidate) > 5:
                return candidate

    # Strategy 2: Last English-like sentence
    lines = [l.strip() for l in raw_reasoning.split("\n") if l.strip()]
    for line in reversed(lines):
        if (
            line
            and line[0].isascii()
            and line[0].isupper()
            and " " in line
            and not any("\u4e00" <= c <= "\u9fff" for c in line)
        ):
            candidate = normalize_text(line)
            if candidate and len(candidate) > 5:
                return candidate

    # Strategy 3: Last non-thought line
    skip_words = ["我们被", "注意", "输出", "翻译", "为了", "可以", "这里", "让", "但", "所以", "最终"]
    for line in reversed(lines):
        if not any(c in line for c in skip_words):
            candidate = normalize_text(line)
            if candidate and len(candidate) > 10:
                return candidate

    raise ValueError(f"Could not extract translation from reasoning:\n{raw_reasoning[:200]}")


def request_translation(client, args: argparse.Namespace, zh_text: str) -> str:
    last_error: Exception | None = None
    for attempt in range(args.retries + 1):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "user", "content": TRANSLATE_PROMPT + zh_text},
                ],
                temperature=0,
                max_tokens=args.max_tokens,
            )
            msg = response.choices[0].message
            en_text = extract_translation(msg.content, getattr(msg, "reasoning", None))
            if not en_text:
                raise ValueError("Empty translation after extraction")
            return en_text
        except Exception as exc:
            last_error = exc
            if attempt >= args.retries:
                break
            time.sleep(min(30.0, args.retry_wait * (2**attempt)))
    raise RuntimeError(str(last_error) if last_error else "Unknown translation error")


def process_one(source_id: str, zh_text: str, args: argparse.Namespace,
                 client) -> JSONLRecord:
    record: JSONLRecord = {"source_id": source_id}
    try:
        if args.dry_run:
            record.update({"ok": True, "dry_run": True, "text_en": "", "text_zh": zh_text})
            return record
        en_text = request_translation(client, args, zh_text)
        record.update({"ok": True, "text_zh": zh_text, "text_en": en_text})
        return record
    except Exception as exc:
        record.update({"ok": False, "text_zh": zh_text, "error": str(exc)})
        return record


def load_shards(data_dir: Path) -> list[Path]:
    shards = sorted(data_dir.glob("train-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No train-*.parquet files in {data_dir}")
    return shards


def iter_unprocessed(data_dir: Path, done_ids: set[str],
                     limit: int) -> Iterator[tuple[str, str]]:
    """Yield (source_id, text) pairs, one shard at a time to avoid OOM."""
    from datasets import Dataset

    shards = load_shards(data_dir)
    count = 0
    for shard_path in shards:
        ds = Dataset.from_parquet(str(shard_path))
        for row in ds:
            sid = row["source_id"]
            text = row.get("text", "")
            if not text or sid in done_ids:
                continue
            yield sid, text
            count += 1
            if limit > 0 and count >= limit:
                return


def run(args: argparse.Namespace) -> None:
    load_dotenv()
    output = args.output
    errors = args.error_output or output.with_name(output.name + ".errors.jsonl")
    done_ids = load_done_ids(output)
    client = None if args.dry_run else build_client(args)

    # Build progress with a quick scan for total count
    from datasets import Dataset
    shards = load_shards(args.data_dir)
    total_available = 0
    for shard_path in shards:
        ds = Dataset.from_parquet(str(shard_path))
        total_available += len(ds)
    if args.limit > 0:
        total_pending = min(args.limit, total_available - len(done_ids))
    else:
        total_pending = total_available - len(done_ids)

    if total_pending <= 0:
        print({"output": str(output), "status": "all_done", "skipped": len(done_ids)})
        return

    success = failed = skipped = 0
    consecutive_failures = 0
    stop = False
    write_lock = threading.Lock()

    progress = tqdm(total=total_pending, desc="Translate", unit="req")
    max_in_flight = max(1, args.concurrency) * 4

    def handle(future: Future[JSONLRecord]) -> bool:
        nonlocal success, failed, consecutive_failures
        record = future.result()
        progress.update(1)
        if record.get("ok"):
            with write_lock:
                append_jsonl(output, record)
            success += 1
            consecutive_failures = 0
        else:
            with write_lock:
                append_jsonl(errors, record)
            failed += 1
            consecutive_failures += 1
        progress.set_postfix(ok=success, fail=failed)
        return args.fail_fast > 0 and consecutive_failures >= args.fail_fast

    in_flight: list[Future[JSONLRecord]] = []
    item_iter = iter_unprocessed(args.data_dir, done_ids, args.limit)

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        # Prime the pipeline
        for _ in range(max_in_flight):
            try:
                sid, zh_text = next(item_iter)
                in_flight.append(pool.submit(process_one, sid, zh_text, args, client))
            except StopIteration:
                break

        while in_flight and not stop:
            for future in as_completed(in_flight):
                in_flight.remove(future)
                if handle(future):
                    stop = True
                    break
                # Replenish
                try:
                    sid, zh_text = next(item_iter)
                    in_flight.append(pool.submit(process_one, sid, zh_text, args, client))
                except StopIteration:
                    pass
                break  # yield back to outer loop to refresh in_flight
        if stop:
            for future in in_flight:
                future.cancel()

    progress.close()
    print(
        {
            "output": str(output),
            "success": success,
            "failed": failed,
            "skipped_done": skipped,
            "stopped_by_fail_fast": stop,
        }
    )
