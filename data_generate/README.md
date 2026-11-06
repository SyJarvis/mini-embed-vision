# Data Generation Tool

This package turns an image source into caption JSONL and packs it into the
parquet format consumed by `MultimodalParquetDataset`.

## Workflow

```text
images / ImageNet Arrow --> caption JSONL --> validation --> training parquet
```

## 1. Prepare a manifest

Each JSONL row must contain a globally unique `id` and an `image_path`.
Relative paths resolve against `--source-dir`. Optional metadata is preserved.

```jsonl
{"id":"myset/000001","image_path":"images/000001.jpg","label":12,"label_name":"cat"}
```

## 2. Generate captions

`--dry-run` validates and encodes images without an API call.

```bash
python -m data_generate caption \
  --source manifest \
  --manifest data_generate/examples/manifest.jsonl \
  --source-dir . \
  --output data/my-captions/captions.jsonl \
  --model "$VISION_MODEL" \
  --base-url "$OPENAI_BASE_URL" \
  --concurrency 8 \
  --dry-run
```

The command is resumable: existing success `id`s are skipped, successes and
errors are written to separate JSONL files, and `--fail-fast` stops sustained
API failures. Replace `--prompt` or use `--prompt-file` to generate another
language or style.

ImageNet Arrow remains supported:

```bash
python -m data_generate caption \
  --source imagenet-arrow \
  --source-dir /mnt/data3/datasets/imagenet-1k \
  --split train \
  --output data/my-captions/captions.jsonl \
  --model "$VISION_MODEL" \
  --base-url "$OPENAI_BASE_URL"
```

## 3. Validate

```bash
python -m data_generate validate data/my-captions/captions.jsonl --require-unique
```

## 4. Pack

```bash
python -m data_generate pack \
  --captions data/my-captions/captions.jsonl \
  --source manifest \
  --source-dir . \
  --output-dir data/my-caption-parquet \
  --shard-size 2000
```

For ImageNet Arrow, use `--source imagenet-arrow` and the same Arrow
`--source-dir` used during generation. The output schema is:

| Field | Type |
|---|---|
| `image` | decoded JPEG image |
| `text` | caption |
| `label` | int64, `-1` when unavailable |
| `label_name` | string, empty when unavailable |
| `source_id` | deterministic sample ID |

Train with:

```bash
python train.py --data_dir data/my-caption-parquet
```
