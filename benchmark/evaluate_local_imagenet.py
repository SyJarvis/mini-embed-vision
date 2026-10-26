#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from benchmark.model_wrapper import MiniEmbedVisionMTEBWrapper



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mini-embed-vision-v1.5")
    parser.add_argument("--data_dir", default="/mnt/data3/datasets/imagenet-1k")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="benchmark/results/v15_imagenet_val/results.json")
    parser.add_argument("--model_backend", choices=["mini", "nomic"], default="mini")
    parser.add_argument(
        "--language", choices=["en", "zh"], default="zh",
        help="Language for class-name prompts (default: zh, uses translated class names)",
    )
    parser.add_argument("--text-model", default="BAAI/bge-base-zh-v1.5")
    parser.add_argument("--vision-backend", choices=["transformers", "timm"], default="transformers")
    parser.add_argument("--timm-model-name", default="eva02_base_patch14_224")
    parser.add_argument("--text-pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--query-prefix", default=None, help="e.g. 'query: ' for e5 models")
    parser.add_argument("--model-revision", default="v1.5")
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    shards = sorted(Path(args.data_dir).glob("imagenet-1k-validation-*.arrow"))
    if not shards:
        raise FileNotFoundError(f"No ImageNet validation shards under {args.data_dir}")

    class_names_en = Dataset.from_file(str(shards[0])).features["label"].names

    if args.language == "zh":
        zh_jsonl = Path(__file__).resolve().parents[1] / "data" / "imagenet-1k-zh" / "class_names_zh_clean.jsonl"
        if not zh_jsonl.exists():
            raise FileNotFoundError(
                f"Chinese class names not found: {zh_jsonl}. "
                "Run benchmark/clean_translations.py first, or use --language en."
            )
        zh_map = {}
        with open(zh_jsonl, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                zh_map[rec["index"]] = rec["zh"].strip()
        class_names = [zh_map.get(i, class_names_en[i]) for i in range(len(class_names_en))]
        missing = sum(1 for i in range(len(class_names_en)) if i not in zh_map)
        if missing:
            print(f"[Warning] {missing} class names missing Chinese translation, falling back to English")
        print(f"[Config] Language: zh (Chinese class names)")
    else:
        class_names = list(class_names_en)
        print(f"[Config] Language: en (English class names)")
    if args.model_backend == "nomic":
        from benchmark.nomic_wrapper import LocalNomicEmbedVision

        wrapper = LocalNomicEmbedVision(
            vision_path=args.model,
            text_path="nomic-embed-text-v1.5",
            device=args.device,
            batch_size=args.batch_size,
        )
        class_embeddings = wrapper.encode_texts(class_names)
        model_name = "nomic-ai/nomic-embed-vision-v1.5"
        model_revision = "official-v1.5"
    else:
        wrapper = MiniEmbedVisionMTEBWrapper(
            args.model,
            args.device,
            batch_size=args.batch_size,
            max_length=512,
            text_model_name=args.text_model,
            vision_backend=args.vision_backend,
            timm_model_name=args.timm_model_name,
            text_pooling=args.text_pooling,
            query_prefix=args.query_prefix,
        )
        class_embeddings = wrapper._encode_texts(class_names).cpu()
        model_name = "teamix/mini-embed-vision"
        model_revision = args.model_revision

    correct = 0
    total = 0
    top_5_correct = 0
    started = time.time()
    for shard_index, shard in enumerate(shards):
        dataset = Dataset.from_file(str(shard))
        for start in range(0, len(dataset), args.batch_size):
            end = min(start + args.batch_size, len(dataset))
            batch = dataset[start:end]
            images = [image.convert("RGB") if image.mode != "RGB" else image for image in batch["image"]]
            image_embeddings = (
                wrapper.encode_images(images)
                if args.model_backend == "nomic"
                else wrapper._encode_images(images).cpu()
            )
            scores = image_embeddings @ class_embeddings.t()
            predicted = scores.argmax(dim=1)
            labels = torch.tensor(batch["label"], dtype=torch.long)
            top_5 = scores.topk(k=5, dim=1).indices
            correct += int((predicted == labels).sum())
            top_5_correct += int((top_5 == labels[:, None]).any(dim=1).sum())
            total += len(labels)
            if total % 5000 < args.batch_size:
                elapsed = time.time() - started
                print(
                    f"[Progress] {total} images | "
                    f"top1={correct / total:.4f} top5={top_5_correct / total:.4f} | "
                    f"{elapsed:.1f}s",
                    flush=True,
                )
            if args.limit is not None and total >= args.limit:
                break
        if args.limit is not None and total >= args.limit:
            break

    accuracy = correct / total
    top_5_accuracy = top_5_correct / total
    output = {
        "model_name": model_name,
        "model_revision": model_revision,
        "task_name": "ImageNet1kValZeroShotClassification",
        "protocol": "Full 50,000-image validation set; zero-shot classification",
        "class_prompt": "{class_name}",
        "language": args.language,
        "image_count": total,
        "class_count": len(class_names),
        "scores": {
            "validation": {
                "accuracy": accuracy,
                "top_5_accuracy": top_5_accuracy,
                "main_score": accuracy,
            }
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
    print(json.dumps(output["scores"]["validation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
