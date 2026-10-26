from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Iterator

from PIL import Image


JSONLRecord = dict[str, Any]


def load_dotenv(path: Path = Path(".env")) -> None:
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
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip("'").strip('"')


def iter_jsonl(path: Path) -> Iterator[tuple[int, JSONLRecord]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc


def read_jsonl(path: Path) -> list[JSONLRecord]:
    return [record for _, record in iter_jsonl(path)]


def append_jsonl(path: Path, record: JSONLRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    for _, record in iter_jsonl(path):
        sample_id = record.get("id")
        if isinstance(sample_id, str) and sample_id:
            done.add(sample_id)
    return done


def load_image(reference: str | Path) -> Image.Image:
    path = Path(reference)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def resize_image(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def encode_image_b64(
    image: Image.Image,
    max_side: int = 768,
    jpeg_quality: int = 85,
) -> str:
    rgb = image.convert("RGB")
    resized = resize_image(rgb, max_side)
    buffer = io.BytesIO()
    resized.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    cleaned = text.strip().replace("```", "").strip().strip("\"'“”‘’")
    return " ".join(cleaned.split())


def as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
