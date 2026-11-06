# mini-embed-vision v1.5 训练与导出指南

> 2026-08-21 | EVA02-Base-P14 + BGE-base-zh-v1.5

## 1. 训练目标

在不显著增加可训练参数量级的前提下，显著提升视觉表征能力，把文本输入长度从 128 tokens 提升到 BGE 的 512 tokens 上限，并提供轻量图文重排序能力。

### 核心目标

- **视觉升级**：`CLIP ViT-B/32` → `EVA02-Base-P14-224 (mim_in22k)`，patch token 从 49 提升到 257。
- **文本保留中文基线**：继续使用 `BAAI/bge-base-zh-v1.5`，冻结文本塔。
- **文本池化修正**：按 BGE 官方语义使用 CLS pooling，同时保留 mean pooling 配置用于 ablation。
- **文本长度提升**：训练与 benchmark 默认 `max_length=512`。
- **保持轻量**：总参数约 190.4M，可训练参数约 88.1M。
- **统一部署格式**：最终导出单文件 `model.safetensors`，模型内部包含 BGE、EVA02、投影层和全部训练后权重。
- **精排能力**：可选轻量 Cross-Modal Reranker，与双塔 embedding 共同组成检索 pipeline。

### 非目标

- 本阶段不引入 Qwen3-Embedding。
- 不切换 LLM 统一架构。
- 不启用 MRL、指令感知和 reranker。

## 2. 模型结构

```text
Text
  → BGE-base-zh-v1.5 (frozen)
  → CLS pooling
  → Text Projection MLP
  → L2 normalize
  → text embedding [B, 768]

Image
  → Resize 224 + CenterCrop 224 + CLIP normalization
  → EVA02-Base-P14-224 (trainable)
  → global average pooling
  → Vision Projection MLP
  → L2 normalize
  → image embedding [B, 768]

Training
  → bidirectional InfoNCE(text, image)

Optional reranker
  → [text_emb, image_emb, product, abs-difference]
  → MLP
  → relevance score
```

### 组件参数

| 组件 | 结构 | 参数量 | 状态 |
|---|---:|---:|---|
| Text encoder | BGE-base-zh-v1.5 / BERT-base | ~102M | frozen |
| Vision encoder | EVA02-Base-P14 / SwiGLU | ~85.8M | trainable |
| Text projection | Linear 768→768 + GELU + Linear + LN | ~1.18M | trainable |
| Vision projection | Linear 768→768 + GELU + Linear + LN | ~1.18M | trainable |
| Temperature | scalar | 1 | trainable |
| Reranker | 4×768 → 256 → 1 MLP | ~0.79M | optional / trainable |
| **Total** |  | **~190.4M** | **~88.1M trainable** |

启用 reranker 后总参数约 **191.2M**，可训练参数约 **88.9M**。

### 关键配置

| 配置 | 值 |
|---|---:|
| `embed_dim` | 768 |
| `vision_backend` | `timm` |
| `timm_model_name` | `eva02_base_patch14_224` |
| `text_pooling` | `cls` |
| `enable_reranker` | `true`（v1.5 rerank 版本） |
| `reranker_hidden_dim` | 256 |
| `reranker_loss_weight` | 0.2 |
| `max_text_len` | 512 |
| image input | 224×224 |
| patch size | 14 |
| vision tokens | 256 patches + CLS = 257 |

## 3. 数据与预处理

当前数据仍使用 ImageNet 中文描述 parquet 数据集。

### 文本

- Tokenizer：BGE tokenizer
- Padding：batch 内动态 padding
- Truncation：512 tokens
- 对齐文本不需要额外 prompt
- 后续查询式检索可按 BGE 推荐增加查询指令

### 图像

训练与推理必须使用同一套 EVA02 预处理：

```python
from torchvision import transforms

image_transform = transforms.Compose(
    [
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ]
)
```

### 数据划分

- `train/val` 按 `source_id` 确定性哈希切分
- 默认验证集比例：2%
- 同一 `source_id` 不会同时出现在训练集和验证集

## 4. 训练流程

### 4.1 前向计算

1. 读取 `(image, caption)`。
2. caption 编码为 `input_ids` 和 `attention_mask`，最长 512。
3. 图像变换为 `[3,224,224]`。
4. BGE 输出 `last_hidden_state[:, 0]`。
5. EVA02 输出 `[B,768]` pooled feature。
6. 双塔分别过投影 MLP 并 L2 normalize。

### 4.2 对比损失

采用双向 InfoNCE：

```text
L_contrastive = (CE(text→image, labels) + CE(image→text, labels)) / 2

L_rerank = BCE(reranker_score(text, image), positive/negative label)

L = L_contrastive + 0.2 * L_rerank
```

### 4.3 DDP 负例

DDP 环境使用可反传的 all-gather：

1. 收集所有 rank 的 text/image embeddings。
2. 正样本标签为全局对角位置。
3. 本地 loss 对全局负例计算。
4. backward 时仅回传本地分片梯度。

### 4.4 重排序训练

Reranker 与双塔共享同一个 batch：

- 对角线 `(text_i, image_i)` 为正样本，label=1
- 非对角 `(text_i, image_j)` 为负样本，label=0
- DDP 下使用全局 image embeddings，负例数随 GPU 数线性增加
- reranker 输入为归一化后的 text/image embedding
- 特征包括 `[text, image, text*image, |text-image|]`

这样 reranker 学到的是与主检索空间一致的语义相关性，可以直接用于 embedding 召回后的精排。

### 4.5 优化器与验证

| 参数 | 默认值 |
|---|---:|
| optimizer | AdamW |
| learning rate | 1e-5 |
| weight decay | 0.01 |
| warmup ratio | 5% |
| scheduler | linear decay |
| batch size per GPU | 32 |
| autocast | bf16（可用时） |
| grad clip | 1.0 |
| metric | Val Recall@1/5/10 |
| checkpoint | 按 R@1 保存 `best.pth` |

### 4.6 检索+重排序流程

```text
1. query text → text embedding
2. image corpus → image embeddings
3. cosine 检索 top rerank_k（默认 100）
4. reranker 对 top rerank_k 重新打分
5. 返回最终 top_k
```

## 5. 环境准备

建议使用本项目已有环境；`timm` 已加入 `requirements.txt`。

```bash
cd /mnt/data3/runke.zhong/workspace/mini-embed-vision
python -m pip install -r requirements.txt
```

### 必需模型文件

```bash
export EVA02_DIR=/home/runke.zhong/workspace/eva02_base_patch14_224.mim_in22k
```

该目录必须包含：

```text
model.safetensors
```

BGE 可以从本地缓存读取；如需显式使用本地目录，传入 `--text-model <bge_dir>`。

## 6. 训练命令

### 6.1 单卡训练

```bash
export WANDB_RUN_ID=v15-eva02-rerank-001

python train.py \
  --data-dir data/imagenet-1k-zh-captions \
  --text-model BAAI/bge-base-zh-v1.5 \
  --vision-model "$EVA02_DIR" \
  --vision-backend timm \
  --timm-model-name eva02_base_patch14_224 \
  --embed-dim 768 \
  --max-text-len 512 \
  --batch-size 32 \
  --epochs 30 \
  --lr 1e-5 \
  --save-dir checkpoints_v15 \
  --enable-reranker \
  --wandb \
  --wandb-project mini-embed-vision \
  --wandb-run-name v15-eva02-bge-rerank \
  --wandb-run-id "$WANDB_RUN_ID" \
  --num-workers 8
```

### 6.2 多卡训练

以下示例使用 3 卡：

```bash
torchrun --nproc_per_node=3 train.py \
  --data-dir data/imagenet-1k-zh-captions \
  --text-model BAAI/bge-base-zh-v1.5 \
  --vision-model "$EVA02_DIR" \
  --vision-backend timm \
  --timm-model-name eva02_base_patch14_224 \
  --embed-dim 768 \
  --max-text-len 512 \
  --batch-size 32 \
  --epochs 30 \
  --lr 1e-5 \
  --save-dir checkpoints_v15 \
  --enable-reranker \
  --wandb \
  --wandb-project mini-embed-vision \
  --wandb-run-name v15-eva02-bge-rerank \
  --wandb-run-id "$WANDB_RUN_ID" \
  --num-workers 8
```

如果显存不足，优先调整：

```text
batch-size 32 → 16
grad-accum 1 → 2
num-workers 8 → 4
```

### 6.3 冒烟测试

```bash
python train.py \
  --data-dir data/imagenet-1k-zh-captions \
  --vision-model "$EVA02_DIR" \
  --vision-backend timm \
  --max-text-len 512 \
  --batch-size 8 \
  --epochs 1 \
  --lr 1e-5 \
  --save-dir /tmp/checkpoints_v15_smoke \
  --enable-reranker \
  --num-workers 2 \
  --smoke
```

冒烟测试只验证链路，不用于判断模型效果。

### 6.4 恢复训练

```bash
python train.py \
  --data-dir data/imagenet-1k-zh-captions \
  --vision-model "$EVA02_DIR" \
  --vision-backend timm \
  --max-text-len 512 \
  --batch-size 32 \
  --epochs 30 \
  --lr 1e-5 \
  --save-dir checkpoints_v15 \
  --enable-reranker \
  --wandb \
  --wandb-run-id "$WANDB_RUN_ID" \
  --resume checkpoints_v15/model_epoch30.pth
```

### 6.5 W&B 使用说明

首次使用：

```bash
wandb login
```

只想本地记录、不上传：

```bash
export WANDB_MODE=offline
```

恢复同一个 W&B run：

```bash
export WANDB_RUN_ID=v15-eva02-rerank-001
python train.py ... --resume checkpoints_v15/model_epoch30.pth \
  --wandb --wandb-run-id "$WANDB_RUN_ID"
```

记录指标：

```text
train/loss
train/rerank_loss
train/learning_rate
val/R@1
val/R@5
val/R@10
best/R@1
best/epoch
```

多卡训练时只有 rank 0 上传日志，其他 rank 不初始化 W&B。

## 7. Ablation 建议

训练前建议固定随机种子和数据划分，至少对比：

| 实验 | 视觉骨干 | 文本池化 | 文本长度 | 目的 |
|---|---|---:|---:|---|
| v1.0 baseline | CLIP B/32 | mean | 128 | 旧基线 |
| v1.0 + 512 | CLIP B/32 | mean | 512 | 单独测文本长度 |
| v1.5 | EVA02 B/14 | CLS | 512 | 官方 BGE 语义 |
| v1.5 mean | EVA02 B/14 | mean | 512 | 验证池化影响 |

在 `model.py` 中把 `text_pooling="cls"` 改为 `"mean"`，或在构建 config 时显式传入：

```python
MiniEmbedVisionConfig(
    vision_backend="timm",
    text_pooling="mean",
)
```

## 8. 导出统一模型

v1.5 的导出目标是 self-contained 模型目录，不依赖原始 EVA02/BGE 目录。

### 8.1 导出内容

```text
mini-embed-vision-v1.5/
├── config.json               # 含 BGE text_config，不指向外部模型
├── model.safetensors         # BGE + EVA02 + projection + reranker + logit_scale
├── model.py                  # remote code
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── vocab.txt
└── preprocessor_config.json  # 记录 EVA02 预处理参数
```

### 8.2 导出命令

当前 `export_hf_model.py` 的入口仍需补充 CLI；可先在 Python 中调用：

```bash
python - <<'PY'
import os
from export_hf_model import export_model

eva02_dir = os.environ["EVA02_DIR"]

export_model(
    save_dir="mini-embed-vision-v1.5",
    checkpoint_path="checkpoints_v15/best.pth",
    text_model_name="BAAI/bge-base-zh-v1.5",
    vision_model_name=eva02_dir,
    vision_backend="timm",
    timm_model_name="eva02_base_patch14_224",
    enable_reranker=True,
)
PY
```

### 8.3 离线加载验证

```bash
python - <<'PY'
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "mini-embed-vision-v1.5",
    trust_remote_code=True,
    local_files_only=True,
)
model.eval()

input_ids = torch.tensor([[101, 100, 102]])
attention_mask = torch.ones_like(input_ids)
pixel_values = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    text_emb = model.encode_text(input_ids, attention_mask)
    image_emb = model.encode_image(pixel_values)

print(text_emb.shape, image_emb.shape)
print(text_emb.norm(dim=-1), image_emb.norm(dim=-1))
PY
```

期望输出：

```text
torch.Size([1, 768]) torch.Size([1, 768])
tensor([1.]) tensor([1.])
```

## 9. Benchmark

训练完成并导出/加载 `.pth` checkpoint 后，可运行 benchmark。

### 9.1 v1.5 benchmark

```bash
python benchmark/run_benchmark.py \
  --checkpoint checkpoints_v15/best.pth \
  --vision_backend timm \
  --timm_model_name eva02_base_patch14_224 \
  --vision_model_path "$EVA02_DIR" \
  --max_length 512 \
  --batch_size 32 \
  --enable_reranker \
  --tasks quick_image
```

### 9.2 文本长度对照

```bash
python benchmark/run_benchmark.py \
  --checkpoint checkpoints_v15/best.pth \
  --vision_backend timm \
  --vision_model_path "$EVA02_DIR" \
  --max_length 128 \
  --batch_size 32 \
  --enable_reranker \
  --tasks quick_image
```

注意：MTEB embedding wrapper 目前只执行 `encode_text()` / `encode_image()`；MTEB 任务分数衡量的是第一-stage embedding 质量。`--enable_reranker` 的作用是确保 checkpoint 能正确加载 reranker 权重。

## 9.5 使用检索+重排序 Pipeline

```python
import torch
from model import MiniEmbedVisionModel
from pipeline import RetrievalRerankPipeline

model = MiniEmbedVisionModel.from_pretrained(
    "mini-embed-vision-v1.5",
    trust_remote_code=True,
)
pipeline = RetrievalRerankPipeline(model)

query_embeddings = model.encode_text(input_ids, attention_mask)
image_embeddings = model.encode_image(pixel_values)

final_indices, final_scores, candidates, retrieval_scores = pipeline(
    query_embeddings,
    image_embeddings,
    top_k=20,
    rerank_k=100,
)
```

返回值：

| 返回值 | 形状 | 含义 |
|---|---:|---|
| `final_indices` | `[Q,K]` | rerank 后的 top-k 图像 ID |
| `final_scores` | `[Q,K]` | reranker 分数 |
| `candidates` | `[Q,R]` | 第一-stage 召回候选 |
| `retrieval_scores` | `[Q,N]` | 全库 cosine 分数 |

同一 checkpoint 分别用 128 和 512 跑，可以量化长文本输入对具体任务的收益。

## 10. 验收清单

- [ ] 冒烟训练完整跑通 1 epoch。
- [ ] 训练 loss 正常下降，无 NaN。
- [ ] Val Recall@1/5/10 正常输出。
- [ ] `best.pth` 可恢复。
- [ ] 导出目录包含单个 `model.safetensors`。
- [ ] 删除/移动原始 EVA02 和 BGE 目录后，`AutoModel.from_pretrained(..., local_files_only=True)` 仍可加载。
- [ ] 导出模型文本/图像输出均为 `[B,768]` 且 L2 范数为 1。
- [ ] benchmark 使用 512 文本长度完成至少 quick preset。
