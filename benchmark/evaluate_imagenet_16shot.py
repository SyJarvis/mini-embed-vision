#!/usr/bin/env python3
"""
ImageNet-1k 16-shot linear classification benchmark.

Reproduces the MTEB Imagenet1kClassification protocol:
- 5 experiments, each sampling 16 images/class from the train split
- Encode train + test images with the model's image encoder
- Train sklearn LogisticRegression on train embeddings
- Evaluate accuracy on test split

Uses local ImageNet arrow shards (no network required).

Usage:
    CUDA_VISIBLE_DEVICES=0 python benchmark/evaluate_imagenet_16shot.py \
        --model mini-embed-vision-v1.5 \
        --data_dir /mnt/data3/datasets/imagenet-1k \
        --device cuda:0 \
        --batch_size 128 \
        --output benchmark/results/v15_imagenet_16shot/results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from benchmark.model_wrapper import MiniEmbedVisionMTEBWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="ImageNet-1k 16-shot linear classification")
    parser.add_argument("--model", default="mini-embed-vision-v1.5")
    parser.add_argument("--data_dir", default="/mnt/data3/datasets/imagenet-1k")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_experiments", type=int, default=5)
    parser.add_argument("--samples_per_label", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_shards", type=int, default=None,
                        help="Limit number of train shards to scan (None=all)")
    parser.add_argument("--test_split", choices=["test", "validation"], default="validation",
                        help="Which split to evaluate on (default: validation)")
    parser.add_argument("--output", default="benchmark/results/v15_imagenet_16shot/results.json")
    parser.add_argument("--model_backend", choices=["mini", "nomic"], default="mini")
    return parser.parse_args()


def load_train_labels(data_dir: Path, max_shards: int | None = None) -> dict[int, list[int]]:
    """Load train split and build label -> [global_indices] mapping.

    Returns dict mapping label_id to list of dataset row indices within
    the concatenated train shards.
    """
    shards = sorted(data_dir.glob("imagenet-1k-train-*.arrow"))
    if max_shards:
        shards = shards[:max_shards]

    label_to_indices: dict[int, list[int]] = {}
    global_idx = 0
    for shard in shards:
        ds = Dataset.from_file(str(shard))
        labels = ds["label"]
        for label in labels:
            label_to_indices.setdefault(label, []).append(global_idx)
            global_idx += 1
        del ds
    return label_to_indices


def load_train_images(data_dir: Path, selected_indices: set[int],
                       index_to_shard: list[tuple[int, int]]) -> list:
    """Load images for the selected global indices.

    index_to_shard: list of (shard_idx, local_idx) for each global index.
    """
    # Group by shard
    shard_to_local: dict[int, list[tuple[int, int]]] = {}
    for gi in sorted(selected_indices):
        shard_idx, local_idx = index_to_shard[gi]
        shard_to_local.setdefault(shard_idx, []).append((local_idx, gi))

    images = {}
    shards = sorted(data_dir.glob("imagenet-1k-train-*.arrow"))
    for shard_idx, reqs in shard_to_local.items():
        ds = Dataset.from_file(str(shards[shard_idx]))
        for local_idx, gi in reqs:
            images[gi] = ds[local_idx]["image"]
        del ds
    return images


def build_index_map(data_dir: Path, max_shards: int | None = None) -> list[tuple[int, int]]:
    """Build mapping from global index to (shard_idx, local_idx)."""
    shards = sorted(data_dir.glob("imagenet-1k-train-*.arrow"))
    if max_shards:
        shards = shards[:max_shards]

    index_map = []
    for shard_idx, shard in enumerate(shards):
        ds = Dataset.from_file(str(shard))
        n = len(ds)
        for local_idx in range(n):
            index_map.append((shard_idx, local_idx))
        del ds
    return index_map


@torch.inference_mode()
def encode_images(wrapper, images: list, batch_size: int, device: str) -> torch.Tensor:
    """Encode a list of PIL images, return [N, dim] tensor on CPU."""
    all_embeddings = []
    from PIL import Image as PILImage

    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        batch = images[start:end]
        rgb_images = []
        for img in batch:
            if not isinstance(img, PILImage.Image):
                img = PILImage.open(img).convert("RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")
            rgb_images.append(img)

        if hasattr(wrapper, "_encode_images"):
            emb = wrapper._encode_images(rgb_images).cpu()
        else:
            emb = wrapper.encode_images(rgb_images).cpu()
        all_embeddings.append(emb)
        if start % (batch_size * 10) == 0:
            print(f"  Encoding: {start}/{len(images)}", flush=True)

    return torch.cat(all_embeddings, dim=0)


def sample_train_indices(label_to_indices: dict[int, list[int]],
                          samples_per_label: int, n_experiments: int, seed: int):
    """Sample train indices for each experiment.

    Returns list of lists: experiments[exp_idx] = list of global indices.
    Also returns union of all indices for batch encoding.
    """
    rng = np.random.RandomState(seed)
    all_selected = []
    for exp in range(n_experiments):
        exp_indices = []
        for label in sorted(label_to_indices.keys()):
            candidates = label_to_indices[label]
            if len(candidates) <= samples_per_label:
                chosen = candidates
            else:
                chosen = rng.choice(candidates, size=samples_per_label, replace=False).tolist()
            exp_indices.extend(chosen)
        all_selected.append(exp_indices)
    return all_selected


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    # Determine test split
    test_pattern = f"imagenet-1k-{args.test_split}-*.arrow"
    test_shards = sorted(data_dir.glob(test_pattern))
    if not test_shards:
        raise FileNotFoundError(f"No {args.test_split} shards under {data_dir}")

    print(f"[Config] Model: {args.model}")
    print(f"[Config] Data: {data_dir}")
    print(f"[Config] Test split: {args.test_split} ({len(test_shards)} shards)")
    print(f"[Config] n_experiments: {args.n_experiments}, samples_per_label: {args.samples_per_label}")
    print(f"[Config] Device: {args.device}")

    # Load model wrapper
    if args.model_backend == "nomic":
        from benchmark.nomic_wrapper import LocalNomicEmbedVision
        wrapper = LocalNomicEmbedVision(
            vision_path=args.model,
            text_path="nomic-embed-text-v1.5",
            device=args.device,
            batch_size=args.batch_size,
        )
        model_name = "nomic-ai/nomic-embed-vision-v1.5"
        model_revision = "official-v1.5"
    else:
        wrapper = MiniEmbedVisionMTEBWrapper(
            args.model,
            args.device,
            batch_size=args.batch_size,
            max_length=512,
        )
        model_name = "teamix/mini-embed-vision"
        model_revision = "v1.5"

    # Build label -> indices map from train split
    print("\n[1/4] Scanning train split for label distribution...")
    max_shards = args.max_train_shards
    label_to_indices = load_train_labels(data_dir, max_shards)
    n_classes = len(label_to_indices)
    print(f"  Found {n_classes} classes in train split")

    # Build global index -> (shard, local) map
    index_map = build_index_map(data_dir, max_shards)
    print(f"  Total train samples: {len(index_map)}")

    # Sample train indices for all experiments
    print("\n[2/4] Sampling train indices...")
    experiments = sample_train_indices(
        label_to_indices, args.samples_per_label, args.n_experiments, args.seed
    )
    union_indices = sorted(set().union(*[set(exp) for exp in experiments]))
    print(f"  {args.n_experiments} experiments x {args.samples_per_label} samples/class x {n_classes} classes")
    print(f"  Union of unique train samples to encode: {len(union_indices)}")

    # Load and encode train images (union)
    print(f"\n[3/4] Loading and encoding {len(union_indices)} train images...")
    started = time.time()
    train_images = load_train_images(data_dir, set(union_indices), index_map)
    # Reorder to match union_indices
    train_image_list = [train_images[gi] for gi in union_indices]
    train_embeddings = encode_images(wrapper, train_image_list, args.batch_size, args.device)
    elapsed = time.time() - started
    print(f"  Train encoding done in {elapsed:.1f}s, shape: {train_embeddings.shape}")

    # Index lookup: global_idx -> position in union
    idx_to_pos = {gi: pos for pos, gi in enumerate(union_indices)}

    # Load and encode test images
    print(f"\n[4/4] Loading and encoding {args.test_split} images...")
    started = time.time()
    test_images = []
    test_labels = []
    for shard in test_shards:
        ds = Dataset.from_file(str(shard))
        for i in range(len(ds)):
            sample = ds[i]
            img = sample["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            test_images.append(img)
            test_labels.append(sample["label"])
        del ds

    print(f"  Total {args.test_split} images: {len(test_images)}")
    test_embeddings = encode_images(wrapper, test_images, args.batch_size, args.device)
    test_labels = np.array(test_labels)
    elapsed = time.time() - started
    print(f"  Test encoding done in {elapsed:.1f}s, shape: {test_embeddings.shape}")

    # Run experiments
    print(f"\nRunning {args.n_experiments} experiments...")
    all_scores = []
    for exp_idx in range(args.n_experiments):
        selected = experiments[exp_idx]
        train_pos = [idx_to_pos[gi] for gi in selected]
        X_train = train_embeddings[train_pos].numpy()
        y_train = np.array([index_map[gi][0] * 0 + gi for gi in selected])  # placeholder

        # Get actual labels
        y_train = np.array([
            [label for label, indices in label_to_indices.items() if gi in indices][0]
            for gi in selected
        ])
        # Simpler: use index_map to get shard, then lookup label — but we stored labels separately
        # Actually, let's just rebuild labels from the sampling
        # We know each experiment has samples_per_label per class in sorted class order
        y_train = np.repeat(np.arange(n_classes), args.samples_per_label)

        X_test = test_embeddings.numpy()
        y_test = test_labels

        clf = LogisticRegression(
            max_iter=1000,
            random_state=args.seed + exp_idx,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        all_scores.append(acc)
        print(f"  Experiment {exp_idx + 1}/{args.n_experiments}: accuracy = {acc:.4f}")

    mean_acc = float(np.mean(all_scores))
    std_acc = float(np.std(all_scores))

    print(f"\n{'='*50}")
    print(f"  Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"{'='*50}")

    output = {
        "model_name": model_name,
        "model_revision": model_revision,
        "task_name": "Imagenet1kClassification",
        "protocol": f"{args.n_experiments} experiments, {args.samples_per_label}-shot linear probing (LogisticRegression)",
        "test_split": args.test_split,
        "n_experiments": args.n_experiments,
        "samples_per_label": args.samples_per_label,
        "n_classes": n_classes,
        "test_image_count": len(test_images),
        "scores": {
            "validation": {
                "accuracy": mean_acc,
                "accuracy_std": std_acc,
                "per_experiment": [float(s) for s in all_scores],
                "main_score": mean_acc,
            }
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[Output] Results saved to: {output_path}")
    print(json.dumps(output["scores"]["validation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
