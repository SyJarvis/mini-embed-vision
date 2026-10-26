from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from .common import JSONLRecord, append_jsonl, encode_image_b64, load_done_ids, load_dotenv, normalize_text
from .sources import iter_images


DEFAULT_PROMPT = (
    "请用一句中文客观描述这张图片中可见的主体、关键细节和场景。"
    "只写能从画面直接看到的内容，不要猜测意图或品牌。"
    "不要使用列表、标题、引号或额外解释，只输出一句完整描述。"
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", choices=["manifest", "imagenet-arrow"], default="manifest")
    parser.add_argument("--source-dir", type=Path, default=Path("."), help="Image root or ImageNet Arrow directory")
    parser.add_argument("--manifest", type=Path, help="JSONL manifest with id and image_path fields")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--error-output", type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-wait", type=float, default=2.0)
    parser.add_argument("--fail-fast", type=int, default=5, help="Stop after N consecutive failures; 0 disables")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--min-chars", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=0, help="Process at most N pending images; 0 means all")
    parser.add_argument("--dry-run", action="store_true", help="Validate and encode inputs without calling the API")


def build_client(args: argparse.Namespace):
    import os

    from openai import OpenAI

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise ValueError(f"{args.api_key_env} is not set")
    return OpenAI(base_url=args.base_url, api_key=api_key, timeout=args.timeout)


def request_caption(client, args: argparse.Namespace, image_b64: str) -> tuple[str, dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(args.retries + 1):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": args.prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                        ],
                    }
                ],
                temperature=0,
                max_tokens=args.max_tokens,
            )
            message = response.choices[0].message
            caption = normalize_text(message.content) or normalize_text(getattr(message, "reasoning", None))
            if not caption or not args.min_chars <= len(caption) <= args.max_chars:
                raise ValueError(f"Invalid caption length: {len(caption)}")
            usage: dict[str, Any] = {}
            if response.usage is not None:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            return caption, usage
        except Exception as exc:
            last_error = exc
            if attempt >= args.retries:
                break
            time.sleep(min(30.0, args.retry_wait * (2**attempt)))
    raise RuntimeError(str(last_error) if last_error else "Unknown caption error")


def process_one(sample: JSONLRecord, args: argparse.Namespace, client) -> JSONLRecord:
    record = {key: value for key, value in sample.items() if key != "image"}
    try:
        image_b64 = encode_image_b64(sample["image"], args.max_side, args.jpeg_quality)
        if args.dry_run:
            record.update({"ok": True, "dry_run": True, "text": "", "encoded_b64_chars": len(image_b64)})
            return record
        caption, usage = request_caption(client, args, image_b64)
        record.update({"ok": True, "text": caption})
        if usage:
            record["usage"] = usage
        return record
    except Exception as exc:
        record.update({"ok": False, "error": str(exc)})
        return record


def run(args: argparse.Namespace) -> None:
    load_dotenv()
    if args.prompt_file:
        args.prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    output = args.output
    errors = args.error_output or output.with_name(output.name + ".errors.jsonl")
    done_ids = load_done_ids(output)
    client = None if args.dry_run else build_client(args)

    success = failed = skipped = 0
    consecutive_failures = 0
    stop = False
    write_lock = threading.Lock()
    pending = 0
    prepared: list[JSONLRecord] = []
    for sample in iter_images(args.source, args.source_dir, args.manifest, args.split):
        if sample["id"] in done_ids:
            skipped += 1
            continue
        prepared.append(sample)
        pending += 1
        if args.limit > 0 and pending >= args.limit:
            break

    progress = tqdm(total=pending, desc="Caption", unit="img")
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
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        for sample in prepared:
            if stop:
                break
            in_flight.append(pool.submit(process_one, sample, args, client))
            if len(in_flight) >= max_in_flight:
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
    print(
        {
            "output": str(output),
            "success": success,
            "failed": failed,
            "skipped_done": skipped,
            "dry_run": args.dry_run,
            "stopped_by_fail_fast": stop,
        }
    )
