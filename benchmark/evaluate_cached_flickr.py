#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from pytrec_eval import RelevanceEvaluator

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from benchmark.model_wrapper import MiniEmbedVisionMTEBWrapper



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mini-embed-vision-v1.5")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output", default="benchmark/results/v15_flickr_t2i/results.json")
    parser.add_argument("--model_backend", choices=["mini", "nomic"], default="mini")
    parser.add_argument(
        "--language", choices=["en", "zh"], default="zh",
        help="Language for text queries (default: zh, uses translated queries)",
    )
    return parser.parse_args()


def load_cached_split(cache_dir: Path, config: str):
    revision_dir = next((cache_dir / config / "0.0.0").iterdir())
    arrow_files = [
        f for f in revision_dir.glob("*.arrow") if not f.name.startswith("cache-")
    ]
    if not arrow_files:
        arrow_files = list(revision_dir.glob("*.arrow"))
    if not arrow_files:
        raise FileNotFoundError(f"No cached Arrow file for config={config}")
    return Dataset.from_file(str(arrow_files[0]))


@torch.inference_mode()
def encode(model_wrapper, texts, images):
    text_embeddings = []
    for start in range(0, len(texts), model_wrapper.batch_size):
        batch = texts[start : start + model_wrapper.batch_size]
        text_embeddings.append(model_wrapper._encode_texts(batch).cpu())

    image_embeddings = []
    for start in range(0, len(images), model_wrapper.batch_size):
        batch = images[start : start + model_wrapper.batch_size]
        image_embeddings.append(model_wrapper._encode_images(batch).cpu())

    return torch.cat(text_embeddings), torch.cat(image_embeddings)


def metrics_at_k(ranking, relevant_ids, k_values):
    result = {}
    relevant = set(relevant_ids)
    for k_value in k_values:
        top = ranking[:k_value]
        hits = sum(document_id in relevant for document_id in top)
        result[f"recall_at_{k_value}"] = hits / len(relevant)
        result[f"precision_at_{k_value}"] = hits / k_value
        reciprocal = next(
            (1.0 / (rank + 1) for rank, document_id in enumerate(top) if document_id in relevant),
            0.0,
        )
        result[f"mrr_at_{k_value}"] = reciprocal
    return result


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir or Path.home() / ".cache/huggingface/datasets/mteb___flickr30kt2i")
    queries = load_cached_split(cache_dir, "query")
    corpus = load_cached_split(cache_dir, "corpus")
    qrels = load_cached_split(cache_dir, "qrels")

    if args.language == "zh":
        zh_jsonl = Path(__file__).resolve().parents[1] / "data" / "flickr30k-zh" / "queries_zh_clean.jsonl"
        if not zh_jsonl.exists():
            raise FileNotFoundError(
                f"Chinese queries not found: {zh_jsonl}. "
                "Run benchmark/clean_translations.py first, or use --language en."
            )
        zh_map = {}
        with open(zh_jsonl, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                zh_map[rec["id"]] = rec["zh"].strip()
        query_texts = queries["text"]
        zh_texts = [zh_map.get(qid, query_texts[i]) for i, qid in enumerate(queries["id"])]
        missing = sum(1 for qid in queries["id"] if qid not in zh_map)
        if missing:
            print(f"[Warning] {missing} queries missing Chinese translation, falling back to English")
        queries = queries.remove_columns(["text"]).add_column("text", zh_texts)
        print(f"[Config] Language: zh (Chinese queries)")
    else:
        print(f"[Config] Language: en (English queries)")

    if args.model_backend == "nomic":
        from benchmark.nomic_wrapper import LocalNomicEmbedVision

        wrapper = LocalNomicEmbedVision(
            vision_path=args.model,
            text_path="nomic-embed-text-v1.5",
            device=args.device,
            batch_size=args.batch_size,
        )
        query_embeddings = wrapper.encode_texts(queries["text"])
        image_embeddings = wrapper.encode_images(corpus["image"])
        model_name = "nomic-ai/nomic-embed-vision-v1.5"
        model_revision = "official-v1.5"
    else:
        wrapper = MiniEmbedVisionMTEBWrapper(
            args.model,
            args.device,
            batch_size=args.batch_size,
            max_length=512,
        )
        query_embeddings, image_embeddings = encode(wrapper, queries["text"], corpus["image"])
        model_name = "teamix/mini-embed-vision"
        model_revision = "v1.5"
    scores = query_embeddings @ image_embeddings.t()
    rankings = scores.argsort(dim=1, descending=True).tolist()

    query_relevance = {}
    for row in qrels:
        query_relevance.setdefault(row["query-id"], {})[row["corpus-id"]] = row["score"]
    corpus_ids = corpus["id"]
    rankings_with_ids = [[corpus_ids[index] for index in ranking] for ranking in rankings]

    k_values = [1, 3, 5, 10, 20, 100, 1000]
    aggregate = {}
    query_count = len(queries)
    for ranking, (query_id, relevant_map) in zip(rankings_with_ids, query_relevance.items()):
        metrics = metrics_at_k(ranking, relevant_map, k_values)
        for metric, value in metrics.items():
            aggregate[metric] = aggregate.get(metric, 0.0) + value
    aggregate = {metric: value / query_count for metric, value in aggregate.items()}

    run = {
        query_id: {
            corpus_id: float(scores[query_index, corpus_index])
            for corpus_index, corpus_id in enumerate(corpus_ids)
        }
        for query_index, query_id in enumerate(queries["id"])
    }
    evaluator = RelevanceEvaluator(query_relevance, {f"ndcg_cut.{k_value}" for k_value in k_values})
    evaluated = evaluator.evaluate(run)
    for k_value in k_values:
        aggregate[f"ndcg_at_{k_value}"] = 0.0
    for values in evaluated.values():
        for metric, value in values.items():
            key = metric.replace("ndcg_cut_", "ndcg_at_")
            aggregate[key] += value
    for k_value in k_values:
        aggregate[f"ndcg_at_{k_value}"] /= query_count
    aggregate["main_score"] = aggregate["ndcg_at_10"]

    rerank_aggregate = {}
    if args.model_backend != "nomic":
        rerank_evaluator = RelevanceEvaluator(
            query_relevance,
            {f"ndcg_cut.{k_value}" for k_value in k_values if k_value <= 100},
        )
        rerank_run = {}
        rerank_k = min(100, len(corpus_ids))
        for query_index, query_id in enumerate(queries["id"]):
            candidate_indices = rankings[query_index][:rerank_k]
            candidate_images = image_embeddings[candidate_indices].to(wrapper.device)
            query_embedding = query_embeddings[query_index].unsqueeze(0).to(wrapper.device)
            rerank_scores = wrapper.model.score_rerank(
                query_embedding,
                candidate_images.unsqueeze(0),
            )[0]
            rerank_run[query_id] = {
                corpus_ids[corpus_index]: float(score)
                for corpus_index, score in zip(candidate_indices, rerank_scores.cpu())
            }
        rerank_evaluated = rerank_evaluator.evaluate(rerank_run)
        rerank_aggregate = {
            f"ndcg_at_{k_value}": 0.0 for k_value in k_values if k_value <= 100
        }
        for values in rerank_evaluated.values():
            for metric, value in values.items():
                key = metric.replace("ndcg_cut_", "ndcg_at_")
                rerank_aggregate[key] += value
        rerank_count = len(rerank_evaluated)
        rerank_aggregate = {
            metric: value / rerank_count for metric, value in rerank_aggregate.items()
        }
        rerank_aggregate["main_score"] = rerank_aggregate["ndcg_at_10"]
        aggregate["embedding_ndcg_at_10"] = aggregate["ndcg_at_10"]
        aggregate["rerank_ndcg_at_10"] = rerank_aggregate["ndcg_at_10"]

    output = {
        "model_name": model_name,
        "model_revision": model_revision,
        "task_name": "Flickr30kT2IRetrieval",
        "language": args.language,
        "dataset_revision": "e819702b287bfbe084e129a61f308a802b7c108e",
        "query_count": query_count,
        "corpus_count": len(corpus),
        "scores": {
            "test": aggregate,
            **({"test_rerank_top_100": rerank_aggregate} if rerank_aggregate else {}),
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
