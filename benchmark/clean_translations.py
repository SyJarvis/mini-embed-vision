#!/usr/bin/env python3
"""
Clean Chinese translation data containing LLM reasoning artifacts.

Extracts pure Chinese translations from dirty entries. Handles:
- Short class names (1+ Chinese chars, incl. CJK Extension A)
- LLM chain-of-thought leakage (English reasoning after translation)
- Multiple translation candidates (picks best/last complete one)
"""

import json
import re
from pathlib import Path


def is_chinese_char(c: str) -> bool:
    cp = ord(c)
    return (
        0x4e00 <= cp <= 0x9fff       # CJK Unified Ideographs
        or 0x3400 <= cp <= 0x4dbf    # CJK Extension A
        or 0xf900 <= cp <= 0xfaff    # CJK Compatibility Ideographs
    )


def count_chinese(text: str) -> int:
    return sum(1 for c in text if is_chinese_char(c))


def strip_leading_noise(text: str) -> str:
    text = text.strip()
    text = re.sub(
        r"^(Alternatively[,:\s]*|A more natural.*?[:\s]*|we can .*?[:\s]*"
        r"|I'll go with[:\s]*|So[:\s]*|But to be precise.*?[:\s]*"
        r"|output[:\s]*|more concise[:\s]*|The most common.*?[:\s]*"
        r"|Also,\s*|Note[:\s]*|Given\s.*?[:\s]*|might be\s*"
        r"|is often\s*|is less\s.*?[:\s]*|is also\s*)",
        "", text, flags=re.IGNORECASE,
    )
    text = text.strip().strip('"\u201c\u201d\u2018\u2019\'- ,.;:!?\u3000')
    for i, c in enumerate(text):
        if is_chinese_char(c):
            if i > 0:
                leading = text[:i]
                if all(ord(ch) < 0x2000 or ch.isspace() for ch in leading):
                    text = text[i:]
            break
    return text.strip()


def cut_at_restart(text: str) -> str:
    """Cut at the first English/Chinese reasoning restart point."""
    restart_patterns = [
        r'\s+But\s', r'\s+Or\s', r'\s+However', r'\s+Alternatively',
        r"\s+I'll\s", r"\s+I\s+think\s", r'\s+Actually', r'\s+But\s+that',
        r'\s+But\s+the\s+user', r'\s+But\s+note', r'\s+But\s+for',
        r'\s+But\s+since', r'\s+But\s+if', r'\s+But\s+to\s',
        r'\s+So\s+we\s', r'\s+The\s+instruction', r'\s+The\s+user',
        r'\s+The\s+phrase', r'\s+The\s+word', r'\s+In\s+context',
        r'\s+In\s+Chinese', r'\s+This\s+is', r'\s+This\s+could',
        r'\s+If\s+they', r'\s+If\s+the', r'\s+Perhaps', r'\s+Probably',
        r'\s+Note[:\s]', r'\s+Since\s', r'\s+Given\s', r'\s+should\s+we',
        r'\s+we\s+should', r'\s+we\s+could', r'\s+we\s+can\s',
        r'\s+Since\s+they', r'\s+Alaska\s',
        r'或者更简洁[：:]?', r'或者更', r'但注意', r'但为了', r'不过',
        r'，所以', r'，但', r'，不过',
    ]
    earliest = len(text)
    for pattern in restart_patterns:
        m = re.search(pattern, text)
        if m and m.start() < earliest:
            earliest = m.start()
    if earliest < len(text):
        text = text[:earliest].strip()
    return text


def cut_english_after_chinese(text: str) -> str:
    """Remove English reasoning that appears after or mixed with Chinese text."""
    if not text.strip():
        return text

    # Step 1: Remove parenthetical English/pinyin notes after Chinese
    # e.g. "蠵龟 (xī guī) or 红海龟" -> "蠵龟 or 红海龟" -> then handle "or"
    # e.g. "泥狗 (but that's not standard...)" -> "泥狗"
    # Remove (English) and （English） groups that contain mostly ASCII
    def remove_en_parens(m):
        inner = m.group(1)
        en_count = sum(1 for c in inner if c.isascii() and c.isalpha())
        zh_count = sum(1 for c in inner if is_chinese_char(c))
        if en_count >= 2 and en_count > zh_count:
            return ""  # remove English parenthetical
        return m.group(0)  # keep if mostly Chinese

    text = re.sub(r"\(([^()]{1,80})\)", remove_en_parens, text)
    text = re.sub(r"\uff08([^\uff09]{1,80})\uff09", remove_en_parens, text)

    # Clean up double spaces and leading/trailing punctuation left behind
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"[，,]\s*[，,]", "，", text)

    # Step 2: Find continuous Chinese segments and keep only those
    # Split into segments: Chinese-heavy vs English-heavy
    # We want to keep Chinese segments, drop English-only segments
    segments = re.split(r"(\s+)", text)
    result_parts = []
    prev_was_space = False
    for seg in segments:
        if not seg:
            continue
        if seg.isspace():
            if result_parts and not prev_was_space:
                result_parts.append(" ")
                prev_was_space = True
            continue
        zh = sum(1 for c in seg if is_chinese_char(c))
        en = sum(1 for c in seg if c.isascii() and c.isalpha())
        # Keep if has Chinese, or if it's short and between Chinese (punctuation/numbers)
        if zh > 0:
            result_parts.append(seg)
            prev_was_space = False
        elif en > 0 and len(seg) <= 2 and result_parts:
            # Short English between Chinese (rare) - keep
            result_parts.append(seg)
            prev_was_space = False
        elif en >= 3:
            # English word/phrase - this is reasoning, drop it and everything after
            break
        else:
            # Punctuation, numbers, etc - keep if between Chinese
            if result_parts:
                result_parts.append(seg)
                prev_was_space = False

    text = "".join(result_parts).strip()

    # Step 3: Remove trailing Latin scientific names
    # e.g. "大灰鸮，乌林鸮，Strix nebulosa" -> "大灰鸮，乌林鸮"
    # e.g. "黑天鹅，Cygnus atratus" -> "黑天鹅"
    # Latin name: Capitalized word optionally followed by lowercase words
    text = re.sub(r"[，,]\s*[A-Z][a-z]+(?:\s+[a-z]+)*\s*$", "", text)
    text = re.sub(r"\s+[A-Z][a-z]+(?:\s+[a-z]+)*\s*$", lambda m: m.group(0) if not any(is_chinese_char(c) for c in text[:m.start()]) else "", text)
    # Also handle genus names like "Cavia cobaya", "Porphyrio porphyrio"
    text = re.sub(r"[，,]\s*[A-Z][a-z]+\s+[a-z]+\s*$", "", text)

    # Step 4: Remove trailing "). So" and similar fragments
    text = re.sub(r"\)\.?\s*So.*$", "", text)
    text = re.sub(r"\)\.?\s*But.*$", "", text)
    text = re.sub(r"\)\.?\s*However.*$", "", text)
    text = re.sub(r"\.\s*So\s+the\s+translation.*$", "", text)

    # Step 5: Clean trailing punctuation/orphaned connectors
    text = re.sub(r"[，,]\s*$", "", text)
    text = re.sub(r"\s+or\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+or\s+[\u4e00-\u9fff]", "", text)  # "鸭嘴 or as a" -> "鸭嘴"
    text = re.sub(r"\s+Or\s+[\u4e00-\u9fff]", "", text)
    text = re.sub(r"\s+as\s+a\s*$", "", text, flags=re.IGNORECASE)

    # Step 6: Remove trailing English words/phrases (3+ ascii letters) at the very end
    text = re.sub(r"\s+[a-zA-Z]{3,}\s*$", "", text)

    return text.strip().strip('，,、；;()（）')


def cut_at_trailing_quote(text: str) -> str:
    for q in ['\u201d', '\u2019', '"']:
        idx = text.find(q)
        if 0 < idx < len(text) - 2:
            after = text[idx + 1:].strip()
            if after:
                text = text[:idx].strip()
                break
    return text.strip().strip('"\u201c\u201d\u2018\u2019\'')


def is_truncated(text: str) -> bool:
    text = text.rstrip('，、,；;')
    if not text:
        return True
    truncation_endings = [
        "穿着", "身穿", "一个", "一名", "一位", "一些", "一群", "那",
        "和", "与", "及", "或", "被", "把", "手持", "正在", "拿着",
        "戴着", "拿着碗", "身穿蓝", "浸在水中", "密集", "户外野",
        "一根棍", "教练手持", "一名身穿", "一群穿着", "一位身穿",
        "拿着碗筷",
    ]
    for ending in truncation_endings:
        if text.endswith(ending):
            return True
    if text.endswith(("、", "，", ",", "：", ":", "(", "（", "或")):
        return True
    return False


def is_complete(text: str, min_zh: int = 1, allow_short: bool = False) -> bool:
    text = text.strip()
    zh = count_chinese(text)
    if zh < min_zh:
        return False
    if "..." in text or "\u2026" in text:
        return False
    if allow_short:
        return True
    if is_truncated(text):
        return False
    return True


def extract_translation(zh: str, en: str, min_zh: int = 1, allow_short: bool = False) -> tuple[str, str]:
    original = zh.strip()
    if not original:
        return "", "failed"

    zh_chars = count_chinese(original)
    reasoning_markers = [
        "Alternatively", "I'll go with", "I think", "but that",
        "but the user", "But to be precise", "But since",
        "But note:", "But if", "However,", "So we", "So translation",
        "we translate", "we can output", "we should", "we could",
        "为了准确", "为了简洁", "更自然", "更准确", "更简洁",
        "所以中文翻译", "也可以说", "或者更", "翻译为",
        " But ", " Or ", " However", " Actually", " Note:",
        " Since ", " Given ", " Perhaps", " Probably",
        "might be", "is often", "is also", "is less",
        "But note", "But to be", "But since",
        "但注意", "但为了", "不过",
        ", and the", ", but ",
    ]
    starts_dirty = original.startswith(("-", " ", '"', "'", "I", "But", "However", "Alternatively", "A more", "could be", ".", "The "))
    has_reasoning = any(m in original for m in reasoning_markers)

    # Already clean
    if zh_chars >= min_zh and not starts_dirty and not has_reasoning:
        if allow_short:
            # For short names: accept if mostly Chinese or pure Chinese
            if zh_chars / max(len(original), 1) > 0.3 or zh_chars == len(original.replace(" ", "")):
                return original, "ok"
        else:
            if zh_chars / max(len(original), 1) > 0.3:
                return original, "ok"

    candidates = []

    # Strategy 1: Quoted Chinese strings
    quoted_patterns = [
        r'"([^"]*[\u4e00-\u9fff\u3400-\u4dbf][^"]*)"',
        r'\u201c([^\u201d]*[\u4e00-\u9fff\u3400-\u4dbf][^\u201d]*)\u201d',
    ]
    for pattern in quoted_patterns:
        for match in re.finditer(pattern, original):
            q = match.group(1).strip()
            q = cut_at_restart(q)
            if is_complete(q, min_zh, allow_short):
                candidates.append((q, count_chinese(q), len(q), 0))

    # Strategy 2: After markers
    markers = [
        "翻译为：", "翻译:", "翻译：", "翻译为",
        "I'll go with: ", "I'll go with ",
        "Alternatively: ", "Alternatively, ",
        "A more natural phrasing might be: ",
        "A more natural phrasing might be ",
        "So: ", "So translation: ",
        "So the translation is: ",
        "But to be precise, we can output: ",
        "we can output: ", "Output: ",
        "more concise: ",
    ]
    for marker in markers:
        idx = original.find(marker)
        if idx >= 0:
            after = original[idx + len(marker):].strip()
            after = strip_leading_noise(after)
            after = cut_at_restart(after)
            after = cut_at_trailing_quote(after)
            after = strip_leading_noise(after)
            if is_complete(after, min_zh, allow_short):
                candidates.append((after, count_chinese(after), len(after), 1))

    # Strategy 3: Split by "或者" / " Or "
    if not any(c[3] <= 1 for c in candidates):
        or_splits = re.split(r'或者更简洁[：:]?|或者更|或者|\s+Or\s+', original)
        for seg in or_splits:
            seg = strip_leading_noise(seg)
            seg = cut_at_restart(seg)
            if is_complete(seg, min_zh, allow_short):
                candidates.append((seg, count_chinese(seg), len(seg), 2))

    # Strategy 4: Longest predominantly-Chinese segment
    if not candidates:
        segments = re.split(r'\n|\. [A-Z]| But | Or | However| Alternatively', original)
        for seg in segments:
            seg = strip_leading_noise(seg)
            seg = cut_at_restart(seg)
            if is_complete(seg, min_zh, allow_short):
                candidates.append((seg, count_chinese(seg), len(seg), 3))

    if not candidates:
        return original, "failed"

    candidates.sort(key=lambda x: (x[3], -x[1], -x[2]))
    best = candidates[0][0]

    # Final cleanup
    best = best.strip().strip('"\u201c\u201d\u2018\u2019\'').strip()
    best = cut_at_restart(best)
    best = cut_english_after_chinese(best)
    best = cut_at_trailing_quote(best)
    best = best.strip().strip('"\u201c\u201d\u2018\u2019\'').strip()
    best = cut_english_after_chinese(best)

    if count_chinese(best) < min_zh:
        return original, "failed"
    if not allow_short and is_truncated(best):
        return original, "failed"

    if best == original:
        return original, "failed"

    return best, "cleaned"


def clean_file(input_path: Path, output_path: Path, min_zh: int = 1, allow_short: bool = False):
    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    stats = {"ok": 0, "cleaned": 0, "failed": 0, "empty": 0}
    failed_records = []

    for rec in records:
        zh = rec.get("zh", "").strip()
        en = rec.get("en", "")
        if not zh:
            stats["empty"] += 1
            rec["clean_status"] = "empty"
            failed_records.append(rec)
            continue

        cleaned, status = extract_translation(zh, en, min_zh=min_zh, allow_short=allow_short)
        rec["zh_original"] = zh if status == "cleaned" else ""
        rec["zh"] = cleaned
        rec["clean_status"] = status

        if status == "ok":
            stats["ok"] += 1
        elif status == "cleaned":
            stats["cleaned"] += 1
        else:
            stats["failed"] += 1
            failed_records.append(rec)

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    failed_path = output_path.with_suffix(".failed.jsonl")
    with open(failed_path, "w", encoding="utf-8") as f:
        for rec in failed_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return stats, len(records)


def main():
    base = Path("data")

    flickr_in = base / "flickr30k-zh" / "queries_zh.jsonl"
    flickr_out = base / "flickr30k-zh" / "queries_zh_clean.jsonl"
    if flickr_in.exists():
        print(f"\n{'='*60}")
        print(f"Cleaning: {flickr_in}")
        stats, total = clean_file(flickr_in, flickr_out, min_zh=4, allow_short=False)
        print(f"  Total: {total}")
        print(f"  Already clean: {stats['ok']}")
        print(f"  Cleaned:       {stats['cleaned']}")
        print(f"  Failed:        {stats['failed']}")
        print(f"  Empty:         {stats['empty']}")
        print(f"  Output: {flickr_out}")

    imagenet_in = base / "imagenet-1k-zh" / "class_names_zh.jsonl"
    imagenet_out = base / "imagenet-1k-zh" / "class_names_zh_clean.jsonl"
    if imagenet_in.exists():
        print(f"\n{'='*60}")
        print(f"Cleaning: {imagenet_in}")
        stats, total = clean_file(imagenet_in, imagenet_out, min_zh=1, allow_short=True)
        print(f"  Total: {total}")
        print(f"  Already clean: {stats['ok']}")
        print(f"  Cleaned:       {stats['cleaned']}")
        print(f"  Failed:        {stats['failed']}")
        print(f"  Empty:         {stats['empty']}")
        print(f"  Output: {imagenet_out}")

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
