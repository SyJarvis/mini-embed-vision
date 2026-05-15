#!/usr/bin/env python3
"""
C-MTEB Benchmark Runner for MiniEmbedVision.

Usage:
    # 快速评测（5个核心任务）
    python benchmark/run_benchmark.py --model_path ./mini-embed-vision-v1.0 --tasks quick

    # 全量评测
    python benchmark/run_benchmark.py --model_path ./mini-embed-vision-v1.0 --tasks full

    # 指定任务
    python benchmark/run_benchmark.py --model_path ./mini-embed-vision-v1.0 --tasks "T2Retrieval,ATEC,TNews"

    # 查看可用任务
    python benchmark/run_benchmark.py --list_tasks
"""

import argparse
import json
import sys
import os
from datetime import datetime

# 确保项目根目录在 sys.path 中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import mteb
from benchmark.model_wrapper import MiniEmbedVisionWrapper
from benchmark.task_configs import get_tasks, list_all_tasks, PRESETS


def parse_args():
    parser = argparse.ArgumentParser(description="C-MTEB Benchmark for MiniEmbedVision")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./mini-embed-vision-v1.0",
        help="模型路径（HuggingFace 格式目录）",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="quick",
        help=f"任务预设 ({'/'.join(PRESETS.keys())}) 或逗号分隔的任务名",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="编码批大小 (default: 32)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="最大序列长度 (default: 512)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备 (cuda/cpu, default: auto)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="结果输出目录 (default: benchmark/results/<timestamp>)",
    )
    parser.add_argument(
        "--list_tasks",
        action="store_true",
        help="列出所有可用任务并退出",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["cmn"],
        help="评测语言 (default: cmn 即中文)",
    )
    return parser.parse_args()


def print_results_summary(model_result) -> None:
    """打印评测结果摘要表。"""
    print("\n" + "=" * 70)
    print("  C-MTEB Evaluation Results Summary")
    print("=" * 70)

    task_results = model_result.task_results

    # 按 main_score 降序排列输出
    rows = []
    for tr in task_results:
        if hasattr(tr, "scores") and tr.scores:
            # scores 是 dict[str, dict[str, Any]]，key 是 hf_subset
            # 取第一个 subset 的分数
            for subset, split_scores in tr.scores.items():
                for split_name, metrics in split_scores.items():
                    main_score = metrics.get("main_score", None)
                    rows.append((tr.task_name, subset, split_name, main_score))

    if not rows:
        print("  (No results to display)")
        return

    # 格式化输出
    print(f"  {'Task':<30} {'Split':<12} {'Score':>10}")
    print(f"  {'-'*30} {'-'*12} {'-'*10}")
    for task_name, subset, split_name, score in rows:
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"  {task_name:<30} {split_name:<12} {score_str:>10}")

    # 计算平均分
    valid_scores = [s for _, _, _, s in rows if s is not None]
    if valid_scores:
        avg = sum(valid_scores) / len(valid_scores)
        print(f"  {'-'*30} {'-'*12} {'-'*10}")
        print(f"  {'AVERAGE':<30} {'':12} {avg:>10.4f}")

    print("=" * 70)


def main():
    args = parse_args()

    if args.list_tasks:
        list_all_tasks()
        return

    # 获取任务列表
    task_names = get_tasks(args.tasks)
    print(f"\n[Config] Tasks: {args.tasks} -> {len(task_names)} tasks")
    print(f"[Config] Model: {args.model_path}")
    print(f"[Config] Batch size: {args.batch_size}")
    print(f"[Config] Languages: {args.languages}")

    # 加载 MTEB 任务
    tasks = mteb.get_tasks(tasks=task_names, languages=args.languages)
    print(f"[Config] Loaded {len(tasks)} tasks from MTEB")

    # 加载模型包装器
    model = MiniEmbedVisionWrapper(
        model_path=args.model_path,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    # 设置输出目录
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(_project_root, "benchmark", "results", timestamp)
    os.makedirs(args.output_dir, exist_ok=True)

    # 运行评测
    print(f"\n{'='*70}")
    print("  Starting MTEB Evaluation...")
    print(f"{'='*70}\n")

    results = mteb.evaluate(
        model,
        tasks,
        encode_kwargs={"batch_size": args.batch_size},
        overwrite_strategy="always",
        show_progress_bar=True,
    )

    # 打印结果摘要
    print_results_summary(results)

    # 保存结果
    result_file = os.path.join(args.output_dir, "results.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(
            results.model_dump() if hasattr(results, "model_dump") else str(results),
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    print(f"\n[Output] Results saved to: {result_file}")


if __name__ == "__main__":
    main()
