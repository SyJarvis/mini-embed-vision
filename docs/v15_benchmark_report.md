# mini-embed-vision v1.5 Benchmark 报告

> 2026-08-21 | MTEB Flickr30k T2I | 5,000 queries / 1,000 images

## 目标基准分数

> 本节汇总各参考模型与设计预期，作为 v1.5 中文 benchmark 的对标基准。

### 参考模型基准

| 模型 | 架构 | 参数量 | Flickr30k NDCG@10 | ImageNet Top-1 | 数据来源 |
|------|------|--------|-------------------|----------------|----------|
| Nomic Embed Vision v1.5 | CLIP 双塔 (英文) | ~400M | 0.6781 | 0.6772 | 本地评测 |
| Nomic 官方报告 | CLIP 双塔 (英文) | ~400M | — | 0.710 | 官方 MMTEB |
| Qwen3-VL-Embedding-2B | LLM 统一架构 | 2B | — | — | MMTEB 63.9 (综合) |
| Qwen3-VL-Embedding-8B | LLM 统一架构 | 8B | — | — | MMTEB 67.9 (综合) |
| CLIP-ViT-B/32 (原始) | CLIP 双塔 (英文) | ~150M | — | 0.633 | OpenAI 原始 |

### v1.5 设计预期（来自 benchmark_design.md）

| 任务 | 预期指标 | 预期值 | 依据 |
|------|---------|--------|------|
| ImageNet-1k Classification (16-shot) | accuracy | 50-65% | CLIP-ViT-B/32 zero-shot ~63.3%，中文数据微调后可能接近或略降 |
| Flickr30k T2I Retrieval | Recall@1 | 30-50% | 训练集验证 R@1 ~62%，但 Flickr30k 是英文且图片域不同 |
| Flickr30k T2I Retrieval | NDCG@10 | — | 无显式预期，验证标准为 R@1 > 0.20 |

> **注意**：设计预期基于英文评测。实际中文评测结果见下文「中文 Benchmark 结果」章节。

### v1.5 实际结果 vs 目标

| 任务 | 指标 | 设计预期 | 实际 (中文) | 实际 (英文) | 达标 |
|------|------|---------|-------------|-------------|------|
| Flickr30k T2I | Recall@1 | >0.20 | **0.1188** | 0.0398 | ❌ (中文未达预期，英文远低于预期) |
| Flickr30k T2I | NDCG@10 | — | 0.2368 | 0.1046 | — |
| ImageNet zero-shot | Top-1 | 50-65% (16-shot) | 0.0593 | 0.0154 | ❌ (zero-shot 协议，非 16-shot，不可直接对比) |

> **说明**：设计预期中的 ImageNet 指标针对的是 MTEB 16-shot 线性探测协议（`Imagenet1kClassification`），而实际跑的是全量 50k zero-shot 分类（`ImageNet1kValZeroShotClassification`），两者协议不同，不可直接比较。16-shot 线性探测预期 50-65%，zero-shot 无显式预期。

### v1.0 → v1.5 → Nomic 对比（Flickr30k T2I，英文）

| 模型 | NDCG@10 | Recall@1 | Recall@10 |
|------|---------|----------|-----------|
| v1.0 (英文) | 0.0961 | 0.0350 | 0.1808 |
| v1.5 (英文) | 0.1046 | 0.0398 | 0.1946 |
| v1.5 (中文) | **0.2368** | **0.1188** | **0.3814** |
| Nomic v1.5 (英文) | 0.6781 | 0.5110 | 0.8546 |

### 差距分析

- **v1.5 中文 vs Nomic**：Flickr30k NDCG@10 差 44.1pp，ImageNet Top-1 差 61.8pp。差距虽大但较英文评测缩小。
- **v1.5 中文 vs 设计预期**：Flickr30k R@1 未达 0.20 预期（实际 0.1188），说明跨模态对齐质量仍不足。
- **核心瓶颈**：视觉-语言对齐质量，而非文本语言。中文评测已将文本编码器优势发挥出来，进一步提升需更大规模图文对齐训练数据或更强视觉编码器。

## 结论

- v1.5 embedding 在 Flickr30k text→image 检索上取得 **NDCG@10 = 0.1046**（英文 query）。
- 对比 v1.0 的 0.09615，提升 **+0.85 个百分点（相对提升 8.8%）**。
- 对 embedding top-100 使用 reranker 后，NDCG@10 提升到 **0.1073**，额外提升 **+0.27 个百分点**。
- v1.5 在该英文跨模态任务上有小幅但稳定的能力提升；reranker 有正向收益，但收益仍偏小，说明还需要 hard negative 训练。
- **中文 benchmark**：切换中文 query/类名后性能大幅提升，Flickr30k NDCG@10 从 0.1046 → **0.2368**（+126%），ImageNet Top-1 从 0.0154 → **0.0593**（+285%），验证了中文文本编码器的实际能力此前被英文评测低估。

## Nomic 相同协议对比

对比模型：`nomic-ai/nomic-embed-vision-v1.5 + nomic-embed-text-v1.5`。
Nomic 文本输入使用官方要求的 `search_query: ` 前缀；图像输入使用官方 CLIPImageProcessor 参数。

### Flickr30k T2I

| 模型 | NDCG@1 | NDCG@5 | NDCG@10 | NDCG@100 | Recall@10 | Recall@100 |
|---|---:|---:|---:|---:|---:|---:|
| mini-embed-vision v1.5 | 0.0398 | 0.0817 | 0.1046 | 0.1813 | 0.1946 | 0.5830 |
| Nomic Embed Vision v1.5 | **0.5110** | **0.6509** | **0.6781** | **0.7056** | **0.8546** | **0.9792** |

### ImageNet-1k 全量 validation zero-shot

| 模型 | Top-1 | Top-5 |
|---|---:|---:|
| mini-embed-vision v1.5 | 0.0154 | 0.0576 |
| Nomic Embed Vision v1.5 | **0.6772** | **0.8570** |

Nomic 官方报告 ImageNet zero-shot 为 71.0%。本实验同一实现下得到 67.7%，差异主要来自类名 prompt、tokenizer/processor 与官方评测实现细节，但不影响与 v1.5 的相对结论。

### 差距判断

- v1.5 与 Nomic 的差距非常大，不是小幅调参问题：
  - Flickr30k T2I NDCG@10 差约 **57.4 个百分点**。
  - ImageNet zero-shot Top1 差约 **66.2 个百分点**。
- 主要原因不是模型规模，而是训练目标和数据覆盖：
  - Nomic vision encoder 使用大规模图文对齐训练，并显式优化 ImageNet zero-shot / Datacomp。
  - v1.5 主要在自有 caption 数据上训练，以冻结 BGE 文本空间对齐 EVA02 特征。
  - v1.5 的文本塔是中文 BGE，对英文 Flickr/ImageNet 类名不占优势。
  - EVA02 MIM backbone 提供视觉表征，但没有经历同等规模跨模态对齐，因此视觉-语言空间对齐质量不足。

## 结果对比

| 模型 / Pipeline | NDCG@1 | NDCG@5 | NDCG@10 | NDCG@100 | Recall@10 | Recall@100 |
|---|---:|---:|---:|---:|---:|---:|
| v1.0 embedding | 0.0350 | 0.0747 | 0.0962 | 0.1747 | 0.1808 | 0.5778 |
| v1.5 embedding | 0.0398 | 0.0817 | **0.1046** | 0.1813 | 0.1946 | 0.5830 |
| v1.5 embedding + top100 rerank | 0.0408 | 0.0838 | **0.1073** | 0.1829 | - | - |

## ImageNet-1k 全量验证

本地数据：`/mnt/data3/datasets/imagenet-1k`

| 协议 | 样本数 | Top-1 | Top-5 |
|---|---:|---:|---:|
| 全量 validation zero-shot 分类 | 50,000 | 0.0154 | 0.0576 |

实现方式：

1. 使用 v1.5 文本塔编码 1,000 个 ImageNet 类名。
2. 使用 v1.5 图像塔编码 50,000 张 validation 图像。
3. 以 cosine similarity 最大的类名作为预测类别。

注意事项：

- v1.5 导出权重没有 ImageNet 分类头，因此不能做 1000-way linear head 推理。
- 本协议评估的是文本-图像共享空间的 zero-shot 分类能力。
- MTEB `Imagenet1k` 官方任务是 5 次实验、每类 16 张的 16-shot 线性探测协议，与本实验不同，两者不能直接比较。
- 当前 zero-shot 分类能力明显弱于检索能力；如果目标包含通用物体分类，需要引入分类监督或更强的图文对齐数据。

相对 v1.0：

- NDCG@1：+0.48pp
- NDCG@5：+0.70pp
- NDCG@10：+0.85pp
- Recall@10：+1.38pp
- Recall@100：+0.52pp

## 中文 Benchmark 结果

> 2026-08-24 | Flickr30k 中文 query / ImageNet-1k 中文类名 | 对比英文基线

v1.5 模型的文本编码器为 `bge-base-zh-v1.5`（中文模型），此前 benchmark 使用英文 query 和英文类名，严重低估了模型实际能力。本节补充中文评测结果，使用翻译后的中文数据替换英文输入。

### 数据准备

- **Flickr30k 中文 query**：将 5,000 条英文 query 翻译为中文，清洗 LLM 推理痕迹后 5,000/5,000 全部可用（100%）。
- **ImageNet-1k 中文类名**：将 1,000 个英文类名翻译为中文，清洗后 1,000/1,000 全部可用（100%）。
- 清洗脚本：`benchmark/clean_translations.py`
- 清洗后数据：`data/flickr30k-zh/queries_zh_clean.jsonl`、`data/imagenet-1k-zh/class_names_zh_clean.jsonl`
- 评测脚本已新增 `--language zh` 参数（默认中文），`--language en` 可回退英文。

### Flickr30k T2I 检索：中文 vs 英文

| 指标 | 英文 query (en) | 中文 query (zh) | 相对提升 |
|---|---:|---:|---:|
| Recall@1 | 0.0398 | **0.1188** | +199% |
| Recall@5 | 0.1236 | **0.2912** | +136% |
| Recall@10 | 0.1946 | **0.3814** | +96% |
| Recall@100 | 0.5830 | **0.7700** | +32% |
| NDCG@10 | 0.1046 | **0.2368** | +126% |
| NDCG@10 (rerank) | 0.1073 | 0.2329 | +117% |

中文 query 下 embedding NDCG@10 从 0.1046 提升到 **0.2368**，翻了一倍多。Reranker 在中文场景下未带来正向收益（0.2368 → 0.2329），可能需要针对中文数据重训 reranker。

### ImageNet-1k Zero-Shot 分类：中文 vs 英文

| 指标 | 英文类名 (en) | 中文类名 (zh) | 相对提升 |
|---|---:|---:|---:|
| Top-1 准确率 | 0.0154 | **0.0593** | +285% |
| Top-5 准确率 | 0.0576 | **0.1816** | +215% |

中文类名下 Top-1 从 1.54% 提升到 **5.93%**，Top-5 从 5.76% 提升到 **18.16%**。绝对值仍偏低，主要受限于视觉编码器的跨模态对齐训练规模。

### 与 Nomic 的差距（中文视角）

| 指标 | mini-embed-vision v1.5 (zh) | Nomic v1.5 (en) |
|---|---:|---:|
| Flickr30k NDCG@10 | 0.2368 | 0.6781 |
| ImageNet Top-1 | 0.0593 | 0.6772 |

切换到中文后差距有所缩小，但 Nomic 仍大幅领先。Nomic 使用英文文本编码器和大规模图文对齐训练，v1.5 的核心瓶颈在于视觉-语言对齐质量，而非文本语言。

### 结果文件

- `benchmark/results/v15_flickr_t2i_zh/results.json`
- `benchmark/results/v15_imagenet_val_zh/results.json`

### 复现命令

```bash
cd /mnt/data3/runke.zhong/workspace/mini-embed-vision

# Flickr30k 中文检索
CUDA_VISIBLE_DEVICES=0 python benchmark/evaluate_cached_flickr.py   --model mini-embed-vision-v1.5   --device cuda:0   --batch_size 128   --language zh   --output benchmark/results/v15_flickr_t2i_zh/results.json

# ImageNet-1k 中文 zero-shot 分类
CUDA_VISIBLE_DEVICES=0 python benchmark/evaluate_local_imagenet.py   --model mini-embed-vision-v1.5   --data_dir /mnt/data3/datasets/imagenet-1k   --device cuda:0   --batch_size 128   --language zh   --output benchmark/results/v15_imagenet_val_zh/results.json
```

## 评估配置

- 模型：`mini-embed-vision-v1.5`
- 任务：`Flickr30kT2IRetrieval`
- 数据：5,000 条英文文本 query，1,000 张图像 corpus，每个 query 一个正例图像。
- 文本长度：512 tokens。
- 图像输入：224×224，EVA02 bicubic preprocessing。
- Embedding 相似度：cosine similarity。
- Rerank：cosine 召回 top-100，再使用 `CrossModalReranker` 排序。
- 指标：MTEB 兼容的 `pytrec_eval` NDCG、Recall、MRR。

## 复现命令

```bash
cd /mnt/data3/runke.zhong/workspace/mini-embed-vision

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 \
python benchmark/evaluate_cached_flickr.py \
  --model mini-embed-vision-v1.5 \
  --device cuda:0 \
  --batch_size 64 \
  --output benchmark/results/v15_flickr_t2i/results.json
```

结果文件：

- `benchmark/results/v15_flickr_t2i/results.json`
- v1.0 基线：`benchmark/results/flickr_final/results.json`
- 运行日志：`logs/benchmark_v15_flickr_t2i_cached_rerank.log`
- ImageNet 全量结果：`benchmark/results/v15_imagenet_val/results.json`
- ImageNet 运行日志：`logs/benchmark_v15_imagenet_val.log`
- Nomic Flickr 结果：`benchmark/results/nomic_flickr_t2i/results.json`
- Nomic ImageNet 结果：`benchmark/results/nomic_imagenet_val/results.json`

## 当前 benchmark 覆盖与限制

已覆盖：

- 英文跨模态 text→image 检索。
- embedding 召回质量。
- embedding + reranker pipeline 质量。

尚未覆盖：

- MTEB 官方 ImageNet 16-shot 线性探测：MTEB cache 不完整；本地已完成 50k 全量 zero-shot 分类评估。
- T2Retrieval / MMarcoRetrieval：cache 存在但离线加载 `qrels` 时同样存在配置识别问题，且完整检索任务计算量大。
- TNews / ATEC / LCQMC：本地 cache 不完整，当前离线无法加载。
- Hard negative 场景 benchmark：待构造数据。

已覆盖（中文）：

- 中文跨模态 text→image 检索（Flickr30k 中文 query）。
- 中文 zero-shot 图像分类（ImageNet-1k 中文类名）。

## 下一步建议

1. 构造 Flickr30k hard negative 子集：每 query 1 个正例 + 99 个难负例，重点观察 reranker 增益。
2. 在完整验证集和业务数据上评估 reranker，而不只是简单 corpus。
3. 联网或补齐 C-MTEB / ImageNet 数据 cache 后运行官方 `run_benchmark.py`。
4. 将 `benchmark/evaluate_cached_flickr.py` 扩展成通用 cached retrieval evaluator。
5. 若 reranker 在 hard negative 上仍无显著收益，应提高 reranker loss 权重或重写 reranker 训练采样策略。
6. 针对中文数据重训 reranker：当前 reranker 在中文 Flickr30k 上无正向收益（0.2368 → 0.2329），需使用中文 hard negative 构造训练数据。
7. 扩展中文 benchmark 覆盖：补充中文 hard negative 检索子集和更多中文图文检索数据集。
