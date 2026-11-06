#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate Chinese captions for ImageNet-1k Arrow shards via an OpenAI-compatible vision API."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Iterator

from openai import OpenAI
from PIL import Image
from tqdm.auto import tqdm

DEFAULT_DATASET_DIR = "/mnt/data3/datasets/imagenet-1k"
DEFAULT_SPLIT = "train"
DEFAULT_MODEL = "/root/models/Qwen3.8-27B-NVFP4"
DEFAULT_BASE_URL = "http://47.107.128.214:10008/v1"
DEFAULT_LIMIT = 50
DEFAULT_MAX_SIDE = 768
DEFAULT_JPEG_QUALITY = 85
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TIMEOUT = 60.0
DEFAULT_RETRIES = 3

CAPTION_PROMPT = (
    "请用一句中文客观描述这张图片中可见的主体、关键细节和场景。"
    "只写能从画面直接看到的内容，不要猜测拍摄意图、品牌或类别名称。"
    "不要使用列表、标题、引号或额外解释，只输出一句完整的中文描述。"
)


class CaptionError(RuntimeError):
    """Raised when a caption request fails after retries."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Chinese captions for ImageNet-1k images.")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR, help="Hugging Face ImageNet-1k directory")
    parser.add_argument("--split", default=DEFAULT_SPLIT, choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Number of images to process. Use 0 for all.")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N samples in the selected split")
    parser.add_argument(
        "--output",
        default=None,
        help="Success JSONL path. Default: data/imagenet-1k-captions/{split}_captions.jsonl",
    )
    parser.add_argument(
        "--error-output",
        default=None,
        help="Failure JSONL path. Default: data/imagenet-1k-captions/{split}_captions.errors.jsonl",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--fail-fast", type=int, default=5, help="Stop after N consecutive failures. Use 0 to disable.")
    parser.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE, help="Resize longest image side before encode")
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--concurrency", type=int, default=16, help="Number of parallel caption requests")
    parser.add_argument("--dry-run", action="store_true", help="Read and encode images without calling the API")
    return parser.parse_args()


def default_output_paths(split: str) -> tuple[Path, Path]:
    out_dir = Path("data/imagenet-1k-captions")
    return out_dir / f"{split}_captions.jsonl", out_dir / f"{split}_captions.errors.jsonl"


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load KEY=VALUE pairs from a local .env without overriding existing env vars."""
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
        if not key or key in os.environ:
            continue
        value = value.strip().strip("'").strip('"')
        os.environ[key] = value


def require_api_key() -> str:
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set. Export it or add it to .env before running caption generation.")
    return api_key


def list_arrow_shards(dataset_dir: Path, split: str) -> list[Path]:
    pattern = re.compile(rf"^imagenet-1k-{re.escape(split)}-\d+-of-\d+\.arrow$")
    shards = sorted(path for path in dataset_dir.iterdir() if path.is_file() and pattern.match(path.name))
    if not shards:
        raise FileNotFoundError(f"No Arrow shards found for split={split} in {dataset_dir}")
    return shards


def load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.is_file():
        return done
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
            sample_id = record.get("id")
            if isinstance(sample_id, str) and sample_id:
                done.add(sample_id)
    return done


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def resize_image(image: Image.Image, max_side: int) -> Image.Image:
    rgb = image.convert("RGB")
    width, height = rgb.size
    longest = max(width, height)
    if longest <= max_side:
        return rgb
    scale = max_side / float(longest)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return rgb.resize(new_size, Image.Resampling.LANCZOS)


def encode_image_jpeg_b64(image: Image.Image, max_side: int, jpeg_quality: int) -> str:
    resized = resize_image(image, max_side=max_side)
    buffer = io.BytesIO()
    resized.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def clean_caption(text: str | None) -> str:
    if not text:
        return ""
    caption = text.strip()
    caption = caption.replace("```", "").strip()
    caption = caption.strip("\"'“”‘’")
    caption = re.sub(r"\s+", " ", caption)
    return caption


def request_caption(
    client: OpenAI,
    model: str,
    image_b64: str,
    max_tokens: int,
    retries: int,
) -> tuple[str, dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": CAPTION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                            },
                        ],
                    }
                ],
                temperature=0,
                max_tokens=max_tokens,
            )
            message = response.choices[0].message
            caption = clean_caption(message.content)
            if not caption:
                caption = clean_caption(getattr(message, "reasoning", None))
            if not caption:
                raise CaptionError("Model returned an empty caption")
            usage = {}
            if response.usage is not None:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            return caption, usage
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(8.0, 2 ** attempt))
    raise CaptionError(str(last_error) if last_error is not None else "Unknown caption error")


def count_split_samples(shards: list[Path]) -> int:
    from datasets import Dataset

    return sum(len(Dataset.from_file(str(shard_path))) for shard_path in shards)


def iter_samples(
    shards: list[Path],
    split: str,
    offset: int,
    limit: int,
    done_ids: set[str] | None = None,
) -> Iterator[dict[str, Any]]:
    from datasets import Dataset

    seen = 0
    emitted = 0
    remaining = None if limit <= 0 else limit
    done_ids = done_ids or set()

    for shard_path in shards:
        dataset = Dataset.from_file(str(shard_path))
        label_feature = dataset.features.get("label")
        for row_idx in range(len(dataset)):
            if seen < offset:
                seen += 1
                continue
            if remaining is not None and emitted >= remaining:
                return

            sample_id = f"{split}/{shard_path.name}/{row_idx}"
            if sample_id in done_ids:
                yield {
                    "id": sample_id,
                    "split": split,
                    "shard": shard_path.name,
                    "row": row_idx,
                    "skipped": True,
                }
                seen += 1
                emitted += 1
                continue

            row = dataset[row_idx]
            label = row.get("label")
            label_name = None
            if label_feature is not None and hasattr(label_feature, "int2str") and label is not None:
                try:
                    label_name = label_feature.int2str(label)
                except Exception:
                    label_name = None

            yield {
                "id": sample_id,
                "split": split,
                "shard": shard_path.name,
                "row": row_idx,
                "label": label,
                "label_name": label_name,
                "image": row["image"],
            }
            seen += 1
            emitted += 1


def build_client(base_url: str, api_key: str, timeout: float) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


def _process_one(sample: dict[str, Any], args: argparse.Namespace, client) -> dict[str, Any]:
    """处理单个样本：返回带结果字段的 record（成功带 ok=True，失败带 error）。线程内执行。"""
    record = {
        "id": sample["id"],
        "split": sample["split"],
        "shard": sample["shard"],
        "row": sample["row"],
        "label": sample["label"],
        "label_name": sample["label_name"],
    }
    try:
        image_b64 = encode_image_jpeg_b64(
            sample["image"],
            max_side=args.max_side,
            jpeg_quality=args.jpeg_quality,
        )
        record["encoded"] = True
        if args.dry_run:
            record["text"] = ""
            record["dry_run"] = True
            record["image_bytes_b64"] = len(image_b64)
            record["ok"] = True
            return record
        caption, usage = request_caption(
            client=client,
            model=args.model,
            image_b64=image_b64,
            max_tokens=args.max_tokens,
            retries=args.retries,
        )
        record["text"] = caption
        if usage:
            record["usage"] = usage
        record["ok"] = True
        return record
    except Exception as exc:
        record["error"] = str(exc)
        record["ok"] = False
        return record


def process_samples(args: argparse.Namespace) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {dataset_dir}")

    output_path, error_path = default_output_paths(args.split)
    if args.output:
        output_path = Path(args.output)
    if args.error_output:
        error_path = Path(args.error_output)

    shards = list_arrow_shards(dataset_dir, args.split)
    done_ids = load_done_ids(output_path)
    client = None if args.dry_run else build_client(args.base_url, require_api_key(), args.timeout)

    started = time.time()

    total = count_split_samples(shards)
    if args.offset:
        total = max(0, total - args.offset)
    if args.limit > 0:
        total = min(total, args.limit)

    samples = iter_samples(shards, args.split, args.offset, args.limit, done_ids=done_ids)

    success = 0
    skipped = 0
    failed = 0
    encoded = 0
    consecutive_failures = 0
    stop = False
    write_lock = threading.Lock()

    concurrency = max(1, args.concurrency)
    progress = tqdm(total=total, desc=f"Caption {args.split}", leave=True)
    progress.write(f"concurrency={concurrency}, dry_run={args.dry_run}")

    def handle_result(record: dict[str, Any]) -> bool:
        """落盘并更新计数；返回是否应触发 fail-fast 停止。"""
        nonlocal success, failed, encoded, consecutive_failures
        progress.update(1)
        if record.get("encoded"):
            encoded += 1
        if record.get("ok"):
            with write_lock:
                append_jsonl(output_path, record)
                done_ids.add(record["id"])
            success += 1
            consecutive_failures = 0
        else:
            with write_lock:
                append_jsonl(error_path, record)
            failed += 1
            consecutive_failures += 1
            if args.fail_fast > 0 and consecutive_failures >= args.fail_fast:
                progress.write(
                    f"Stopping after {consecutive_failures} consecutive failures. "
                    f"Last error: {record.get('error')}"
                )
                return True
        progress.set_postfix({"ok": success, "skip": skipped, "fail": failed})
        return False

    # 背压：在途 future 数量上限，避免把百万样本一次性全提交进队列。
    in_flight = []
    max_in_flight = concurrency * 8

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for sample in samples:
            if stop:
                break
            if sample.get("skipped"):
                skipped += 1
                progress.update(1)
                progress.set_postfix({"ok": success, "skip": skipped, "fail": failed})
                continue

            in_flight.append(pool.submit(_process_one, sample, args, client))

            # 背压：在途过多时，先收一批再继续提交
            if len(in_flight) >= max_in_flight:
                while in_flight:
                    done = next(as_completed(in_flight))
                    in_flight.remove(done)
                    if handle_result(done.result()):
                        stop = True
                        break

        # 收尾：处理剩余在途任务
        for done in as_completed(in_flight):
            if handle_result(done.result()):
                stop = True
                break
        # 若 fail-fast 触发，取消尚未开始的 future
        if stop:
            for fut in in_flight:
                fut.cancel()

    progress.close()
    elapsed = time.time() - started
    print(
        json.dumps(
            {
                "split": args.split,
                "limit": args.limit,
                "offset": args.offset,
                "output": str(output_path),
                "error_output": str(error_path),
                "success": success,
                "skipped": skipped,
                "failed": failed,
                "encoded": encoded,
                "elapsed_sec": round(elapsed, 3),
                "dry_run": args.dry_run,
                "concurrency": concurrency,
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    load_dotenv()
    args = parse_args()
    process_samples(args)


if __name__ == "__main__":
    main()
