from __future__ import annotations

import argparse

from . import caption, pack, translate, validate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build multimodal caption training datasets")
    subparsers = parser.add_subparsers(dest="command", required=True)
    caption.add_arguments(subparsers.add_parser("caption", help="Generate captions from images"))
    pack.add_arguments(subparsers.add_parser("pack", help="Pack captions and images into training parquet"))
    translate.add_arguments(subparsers.add_parser("translate", help="Translate Chinese captions to English"))
    validate.add_arguments(subparsers.add_parser("validate", help="Validate generated caption JSONL"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "caption":
        caption.run(args)
    elif args.command == "translate":
        translate.run(args)
    elif args.command == "pack":
        pack.run(args)
    else:
        validate.run(args)
