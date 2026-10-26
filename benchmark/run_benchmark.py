#!/usr/bin/env python3
"""
mini-embed-vision Benchmark Runner (MTEB v2).

Usage:
    # ImageNet classification
    python benchmark/run_benchmark.py --checkpoint checkpoints/best.pth --tasks imagenet

    # Flickr30k text→image retrieval
    python benchmark/run_benchmark.py --checkpoint checkpoints/best.pth --tasks flickr

    # Quick image tasks (ImageNet + Flickr)
    python benchmark/run_benchmark.py --checkpoint checkpoints/best.pth --tasks quick_image

    # Chinese text tasks
    python benchmark/run_benchmark.py --checkpoint checkpoints/best.pth --tasks quick

    # List available task presets
    python benchmark/run_benchmark.py --list_tasks
"""

import argparse
import json
import os
import sys
from datetime import datetime

# Ensure project root is on sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import mteb
from benchmark.model_wrapper import MiniEmbedVisionMTEBWrapper
from benchmark.task_configs import PRESETS, get_tasks, list_all_tasks


def parse_args():
    parser = argparse.ArgumentParser(
        description="mini-embed-vision MTEB Benchmark Runner"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to .pth checkpoint file (e.g. checkpoints/best.pth)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="quick_image",
        help=f"Task preset ({'/'.join(PRESETS.keys())}) or comma-separated task names",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Encoding batch size (default: 64)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Max text sequence length (default: 128)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu, default: auto)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: benchmark/results/<timestamp>)",
    )
    parser.add_argument(
        "--list_tasks",
        action="store_true",
        help="List available tasks / presets and exit",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Filter tasks by language code(s), e.g. cmn eng-Latn. "
            "Default: no filter (all languages)."
        ),
    )
    return parser.parse_args()


def print_results_summary(model_result) -> None:
    """Print evaluation results summary.

    Handles MTEB v2 result structures (BenchmarkResults → model_results → task_results).
    """
    print("\n" + "=" * 70)
    print("  MTEB Evaluation Results Summary")
    print("=" * 70)

    # Navigate to the actual TaskResult list
    if hasattr(model_result, "model_results"):
        # BenchmarkResults
        all_task_results = []
        for mr in model_result.model_results:
            all_task_results.extend(getattr(mr, "task_results", []))
    else:
        all_task_results = getattr(model_result, "task_results", [])

    rows = []
    modalities_seen = set()
    for tr in all_task_results:
        task_name = getattr(tr, "task_name", getattr(tr, "task", str(tr)))
        scores = getattr(tr, "scores", None) or {}

        if isinstance(scores, dict):
            for subset, split_or_results in scores.items():
                if isinstance(split_or_results, dict):
                    # plain scores dict
                    ms = split_or_results.get("main_score", None)
                    rows.append((task_name, subset, "test", ms))
                elif isinstance(split_or_results, list):
                    # list of experiment score dicts → take first entry's main_score
                    for entry in split_or_results:
                        if isinstance(entry, dict) and "main_score" in entry:
                            rows.append((task_name, subset, "test", entry["main_score"]))
                            break
                    else:
                        rows.append((task_name, subset, "test", None))

        # Collect modality info
        task_obj = getattr(tr, "task", None)
        if task_obj and hasattr(task_obj, "metadata"):
            for m in getattr(task_obj.metadata, "modalities", []):
                modalities_seen.add(m)

    if not rows:
        print("  (No results to display)")
        return

    # Print modality tags
    if modalities_seen:
        print(f"  Modalities: {', '.join(sorted(modalities_seen))}")

    print(f"  {'Task':<35} {'Split':<12} {'Score':>10}")
    print(f"  {'-'*35} {'-'*12} {'-'*10}")
    for task_name, subset, split_name, score in rows:
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"  {task_name:<35} {split_name:<12} {score_str:>10}")

    valid_scores = [s for _, _, _, s in rows if s is not None]
    if valid_scores:
        avg = sum(valid_scores) / len(valid_scores)
        print(f"  {'-'*35} {'-'*12} {'-'*10}")
        print(f"  {'AVERAGE':<35} {'':12} {avg:>10.4f}")

    print("=" * 70)


def main():
    args = parse_args()

    if args.list_tasks:
        list_all_tasks()
        return

    # Resolve task names
    task_names = get_tasks(args.tasks)
    print(f"\n[Config] Tasks: {args.tasks} -> {len(task_names)} tasks")
    for t in task_names:
        print(f"    - {t}")
    print(f"[Config] Checkpoint: {args.checkpoint}")
    print(f"[Config] Batch size: {args.batch_size}")
    print(f"[Config] Max length: {args.max_length}")
    print(f"[Config] Device: {args.device or 'auto'}")
    if args.languages:
        print(f"[Config] Language filter: {args.languages}")

    # Load MTEB tasks
    tasks = mteb.get_tasks(tasks=task_names, languages=args.languages)
    print(f"[Config] Loaded {len(tasks)} tasks from MTEB")

    if len(tasks) == 0:
        print(
            "[Error] No tasks loaded. Check that task names are correct and "
            "language filter (if any) matches task languages."
        )
        sys.exit(1)

    # Load model wrapper
    model = MiniEmbedVisionMTEBWrapper(
        checkpoint_path=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(
            _project_root, "benchmark", "results", timestamp
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # Run evaluation
    print(f"\n{'='*70}")
    print("  Starting MTEB Evaluation...")
    print(f"{'='*70}\n")

    results = mteb.evaluate(
        model,
        tasks,
        encode_kwargs={
            "batch_size": args.batch_size,
        },
        overwrite_strategy="always",
        show_progress_bar=True,
    )

    # Print results summary
    print_results_summary(results)

    # Save results
    result_file = os.path.join(args.output_dir, "results.json")
    try:
        if hasattr(results, "model_dump"):
            data = results.model_dump()
        else:
            data = str(results)

        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n[Output] Results saved to: {result_file}")
    except Exception as e:
        print(f"\n[Warning] Failed to save results JSON: {e}")
        # Fallback: try saving model_results separately
        if hasattr(results, "model_results"):
            try:
                fallback_file = os.path.join(args.output_dir, "results_fallback.json")
                simple = []
                for mr in results.model_results:
                    task_data = []
                    for tr in getattr(mr, "task_results", []):
                        task_data.append({
                            "task_name": getattr(tr, "task_name", str(tr)),
                            "scores": getattr(tr, "scores", {}),
                        })
                    simple.append({"model_name": getattr(mr, "model_name", "?"), "tasks": task_data})
                with open(fallback_file, "w", encoding="utf-8") as f:
                    json.dump(simple, f, ensure_ascii=False, indent=2, default=str)
                print(f"[Output] Fallback results saved to: {fallback_file}")
            except Exception as e2:
                print(f"[Warning] Fallback save also failed: {e2}")


if __name__ == "__main__":
    main()
