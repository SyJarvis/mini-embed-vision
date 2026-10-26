# mini-embed-vision 多模态嵌入 Benchmark 设计文档

> 2026-08-21 | 基于 MTEB v2.19.5 + ImageNet-zh 训练模型

---

## 1. 目标

在 MTEB v2 框架下评估 mini-embed-vision 模型在多模态任务上的表现，首批覆盖：

| 任务 | MTEB 任务名 | 评测维度 | 首批 |
|------|------------|---------|------|
| 图像分类 | `Imagenet1kClassification` | 图像嵌入 → skearn LogisticRegression | ✅ |
| 跨模态检索 | `Flickr30kT2IRetrieval` | 以文搜图，Recall@K | ✅ |
| 图像聚类 | (后续扩展) | | 待定 |
| 重排序 | (后续扩展, 由检索任务转换) | | 待定 |

---

## 2. 架构概览

```
benchmark/
  model_wrapper.py    ← 重写：MiniEmbedVisionMTEBWrapper (AbsEncoder)
  run_benchmark.py    ← 更新：支持 checkpoint 路径 + 图像任务
  task_configs.py     ← 扩展：加入图像任务配置
  __init__.py
```

调用链路：

```
mteb.evaluate(model, tasks)
  └─ task.evaluate(model)
       └─ task._evaluate_subset(model, data_split)
            ├─ 分类：encode(train_split) + SklearnEvaluator → accuracy
            └─ 检索：SearchEncoderWrapper(model) → encode(corpus) + encode(queries) → RetrievalEvaluator → R@K
                   └─ model.encode(inputs, task_metadata, hf_split, hf_subset, prompt_type)
                        ├─ inputs.dataset.features 含 "image" → model.encode_image(pixel_values)
                        ├─ inputs.dataset.features 含 "text"  → model.encode_text(input_ids, attention_mask)
                        └─ 两者皆有 → 编码并返回各自嵌入
```

---

## 3. Model Wrapper 设计

### 3.1 类：`MiniEmbedVisionMTEBWrapper(AbsEncoder)`

```python
class MiniEmbedVisionMTEBWrapper(AbsEncoder):
    def __init__(self, checkpoint_path, device=None):
        # 1. 从 .pth 加载模型
        #    mode = MiniEmbedVisionModel(MiniEmbedVisionConfig(...))
        #    state = torch.load(ckpt)["model"]; mode.load_state_dict(state, strict=False)
        #
        # 2. 初始化 tokenizer + image processor
        #    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-zh-v1.5")
        #    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        #
        # 3. 设置 mteb_model_meta
```

### 3.2 `encode()` 方法

核心方法，MTEB 通过 DataLoader 逐 batch 传入 `BatchedInput`：

```python
def encode(self, inputs, *, task_metadata, hf_split, hf_subset, prompt_type=None, **kwargs):
    """
    输入: DataLoader[BatchedInput], 每批是一个 TypedDict
    输出: np.ndarray [N, 768] L2 归一化嵌入
    """
    # 判断输入中有哪些模态
    has_text = "text" in inputs.dataset.features
    has_image = "image" in inputs.dataset.features

    if has_text and not has_image:
        # 纯文本 → encode_text (检索查询)
        for batch in inputs:
            texts = batch["text"]  # list[str]
            tok = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
            emb = model.encode_text(tok.input_ids, tok.attention_mask).cpu()
            all_embs.append(emb)

    elif has_image and not has_text:
        # 纯图像 → encode_image (候选库, 分类输入)
        for batch in inputs:
            images = batch["image"]  # list[PIL.Image]
            pix = image_processor(images=images, return_tensors="pt")["pixel_values"].to(device)
            emb = model.encode_image(pix).cpu()
            all_embs.append(emb)

    else:
        # 多模态（同时含 text + image）
        # ALIGN 参考: 编码后拼接 text_emb + image_emb
        # 但我们的 text 和 image 共享同一语义空间 → 只需返回其一
        # 根据 task_metadata.type 决定:
        #   "ImageClassification" → 返回 image_emb
        #   "Retrieval" - query → text_emb, corpus → image_emb (由 prompt_type 区分)
        ...

    return torch.cat(all_embs, dim=0).numpy()
```

### 3.3 关键参考

- **`mteb/models/model_implementations/align_models.py`**: ALIGNModel 展示了双模态 encode 的标准实现模式——检查 `inputs.dataset.features` 确定模态，分派到各自编码器。
- **`example/retrieval_demo.py`**: 加载 checkpoint 的正确模式——`ckpt["model"]` 含 399 个键，`strict=False`。
- **`train.py:build_model()`**: 构建模型的标准方式，`MiniEmbedVisionConfig` 参数与训练完全一致。

---

## 4. 首批任务

### 4.1 Imagenet1kClassification

| 属性 | 值 |
|------|-----|
| MTEB 类名 | `Imagenet1kClassification` |
| 数据集 | `mteb/wds_imagenet1k` (50k 验证图, 1000 类) |
| 输入列 | `jpg` → 映射为 `image` 模态 |
| 评估方式 | SklearnEvaluator: 从训练集抽 `samples_per_label=16` 张图/类编码 → 训练 LogisticRegression → 预测 test 集 → accuracy |
| 实验次数 | `n_experiments=5` |
| 主指标 | `accuracy` |
| 预期 | ~50-65% (CLIP-ViT-B/32 zero-shot ImageNet ~63.3%, 中文数据微调后可能接近或略降) |

### 4.2 Flickr30kT2IRetrieval

| 属性 | 值 |
|------|-----|
| MTEB 类名 | `Flickr30kT2IRetrieval` |
| 数据集 | `mteb/flickr30kt2i` |
| 模态 | text → query, image → corpus |
| 评估方式 | 31783 张候选图, 1000 文本查询 |
| 主指标 | `Recall@1`, `Recall@5`, `Recall@10`, `NDCG@10` |
| 预期 | R@1 ~30-50% (训练集验证 R@1 ~62%, 但 Flickr30k 是英文, 且图片域不同) |

---

## 5. SentenceTransformerEncoderWrapper 路线 vs 自定义 AbsEncoder 路线

调研后发现 MTEB v2 有两种接入方式：

### 路线 A: 自定义 AbsEncoder (推荐 ✅)

直接继承 `AbsEncoder`，实现自己的 `encode()`。

**优点**：
- 完全控制模型加载逻辑（从原始 .pth 加载，不依赖 HF 导出目录）
- 可以精确处理双模态分派
- 与现有 `example/retrieval_demo.py` 的模式一致
- ALIGNModel 有现成参考

**需要实现**：
- `encode()` 方法：处理 text/image/multimodal 三种输入
- `mteb_model_meta`：模型元数据注册

### 路线 B: SentenceTransformerEncoderWrapper (暂不采用)

MTEB v2 的 `get_model()` 大部分模型走的是 SentenceTransformer，但：
- 需要模型是 HuggingFace 格式 + `trust_remote_code`
- 且 `encode_image()` 需要自定义，无法通过 SentenceTransformer 标准化接口处理

**结论**：不适用于双模态嵌入模型。

---

## 6. 文件改动说明

### `benchmark/model_wrapper.py` — 完全重写

```python
class MiniEmbedVisionMTEBWrapper(AbsEncoder):
    """MTEB wrapper for MiniEmbedVision multimodal model.

    Supports:
    - Text inputs: encodes via BGE-base-zh-v1.5 + mean pooling + L2 norm
    - Image inputs: encodes via CLIP-ViT-B/32 + CLS token + L2 norm
    - Both modalities map to the same 768-d embedding space
    """
    def __init__(self, checkpoint_path, device=None):
        ...

    def encode(self, inputs, *, task_metadata, hf_split, hf_subset,
               prompt_type=None, **kwargs) -> Array:
        ...

    @property
    def dimension(self) -> int: return 768
```

### `benchmark/run_benchmark.py` — 更新

- 新增 `--checkpoint` 参数（替代旧的 `--model_path`）
- image_processor 初始化
- 结果输出包含 task 类别和模态信息
- 支持 `--image-tasks` 和 `--text-tasks` 两组预设

### `benchmark/task_configs.py` — 扩展

新增图像任务预设：

```python
IMAGE_RETRIEVAL_TASKS = [
    "Flickr30kT2IRetrieval",
    "Flickr30kI2TRetrieval",
    "MSCOCOT2IRetrieval",
    "MSCOCOI2TRetrieval",
]

IMAGE_CLASSIFICATION_TASKS = [
    "Imagenet1kClassification",
    "CIFAR10Classification",
    # ...
]

IMAGENET_ONLY = ["Imagenet1kClassification"]
QUICK_IMAGE = ["Imagenet1kClassification", "Flickr30kT2IRetrieval"]
```

---

## 7. 执行计划

```
Phase 1: 重写 model_wrapper.py
  - 实现 MiniEmbedVisionMTEBWrapper.__init__ (加载 checkpoint + tokenizer + image processor)
  - 实现 text_only encode 分支
  - 实现 image_only encode 分支
  - 实现 mteb_model_meta 注册

Phase 2: 更新 run_benchmark.py
  - checkpoint 参数
  - 图像任务预设
  - 结果输出增强

Phase 3: 扩展 task_configs.py
  - 图像任务列表 + 预设

Phase 4: 冒烟验证（ImageNet 分类）
  - HF_ENDPOINT=https://hf-mirror.com
  - python benchmark/run_benchmark.py --checkpoint checkpoints_183k/best.pth --tasks imagenet
  - 验证数据集下载、编码、分类评估全部跑通

Phase 5: 检索验证（Flickr30k）
  - python benchmark/run_benchmark.py --checkpoint checkpoints_183k/best.pth --tasks flickr
  - 验证 R@K 指标合理
```

---

## 8. 验证标准

| 检查点 | 预期结果 |
|--------|---------|
| `import mteb` + `get_tasks()` | 正常，MTEB v2.19.5 |
| `MiniEmbedVisionMTEBWrapper` 加载 | checkpoint 加载成功，无遗漏 key |
| `encode(text)` 输出 shape | [N, 768], float32, L2 归一化 |
| `encode(image)` 输出 shape | [N, 768], float32, L2 归一化 |
| Imagenet1kClassification | 完成 5 次实验, accuracy > 0.30 |
| Flickr30kT2IRetrieval | 完成评估, R@1 > 0.20 |

---

## 9. 风险与注意事项

1. **数据集下载**：MTEB 图像任务数据集较大（ImageNet 50k 图, Flickr30k 31783 图），首次下载需要网络。
2. **sklearn 版本兼容性**：当前 sklearn 1.9.0，MTEB 默认 LogisticRegression 参数需确认兼容。
3. **内存**：Flickr30k 3 万图同时编码可能占显存 10-15GB（batch 适中，bf16 下够用）。
4. **Flickr30k 语言**：Flickr30k 是英文描述，我们的文本编码器是 BGE-base-zh（中英双语），英文描述应该能处理。
