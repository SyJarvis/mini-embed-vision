# 多模态嵌入模型对比分析

> 2026-08-21 | 对比 Chinese-CLIP / nomic-embed-vision / Qwen3-VL-Embedding / mini-embed-vision v1.0

---

## 1. 模型基本信息

| 属性 | Chinese-CLIP ViT-L | nomic-embed-vision-v1.5 | Qwen3-VL-Embedding-2B | **mini-embed-vision v1.0** |
|------|-------------------|------------------------|----------------------|---------------------------|
| **发布** | 2022.11 | 2025 | 2026.02 | 2026.08 |
| **架构范式** | CLIP 双塔 | CLIP 双塔（自定义 ViT） | LLM 统一架构 | CLIP 双塔 |
| **团队** | 阿里达摩院 | Nomic AI | 阿里 Qwen | 我们自己 |
| **开源** | ✅ ModelScope | ✅ Transformers | ✅ Transformers | ✅ |
| **论文** | [arXiv:2211.01335](https://arxiv.org/abs/2211.01335) | [arXiv:2406.18587](https://arxiv.org/abs/2406.18587) | [arXiv:2601.04720](https://arxiv.org/abs/2601.04720) | — |

---

## 2. 视觉编码器对比

| 属性 | Chinese-CLIP | nomic-embed-vision | Qwen3-VL-Embedding | **我们 v1.0** |
|------|-------------|-------------------|-------------------|-------------|
| **视觉骨干** | ViT-L/14 | **自定义 ViT-B/16** | Qwen3-VL 内置 ✅ | ViT-B/32 ❌ |
| **参数量** | ~300M ✅ | ~86M | ~1.5B（LLM 部分） | ~87M |
| **Patch 大小** | **14×14** ✅ | **16×16** ✅ | 14×14 ✅ | **32×32** ❌ |
| **输入分辨率** | 224 / 336 | 224 | 动态可变 ✅ | 224 |
| **MLP** | GELU | **SwiGLU** ✅ | Qwen3 内置 | GELU |
| **位置编码** | 绝对位置 | **RoPE** ✅ | RoPE ✅ | 绝对位置 |
| **Flash Attn** | ❌ | **✅** | ✅ | ❌ |
| **注意力头数** | 16 | 12 | — | 12 |
| **隐层维度** | 1024 | 768 | — | 768 |

> **关键差异**：nomic 虽然也是 CLIP 双塔，但视觉编码器不是标准 CLIP ViT。它用了 **SwiGLU MLP + RoPE + Flash Attention + Attention Pooling**，本质上是 EVA 风格的现代 ViT，比标准 CLIP ViT 强很多。虽然参数量只有 86M（和我们一样），但 patch_size=16 意味着同分辨率下处理 4× 的 patch 数（196 vs 49），视觉细节丰富度完全不同。

---

## 3. 文本编码器对比

| 属性 | Chinese-CLIP | nomic-embed-text | Qwen3-VL-Embedding | **我们 v1.0** |
|------|-------------|------------------|-------------------|-------------|
| **文本模型** | RoBERTa-wwm-ext-base-chinese | **nomic-bert-v1.5 (自定义)** | Qwen3-VL 内置 | BGE-base-zh-v1.5 |
| **基础架构** | BERT-base | BERT-base + 改进 | LLM | RoBERTa-base |
| **参数量** | 102M | 137M | 2B | 102M |
| **隐层维度** | 768 | 768 | 4096 | 768 |
| **层数** | 12 | 12 | 28 | 12 |
| **上下文长度** | 512 | 2048 | **32K** ✅ | **128** ❌ |
| **词汇量** | 21128 | 30528 | — | 21128 |
| **Pooling** | CLS token | **mean pooling** | last_token_pool | **mean pooling** ✅ |
| **语言** | 中文 | 英文 | 30+ 语言 | 中文 ❌ |
| **MTEB 文本分** | — | 62.28 | 63.87 | — |

> **关键差异**：nomic-embed-text 虽然是自定义 BERT（不是 LLM），但在 MTEB 上拿到 62+ 分，证明**好的 BERT 级文本编码器的天花板也在 62-63**。Qwen3-VL 的 32K 上下文是碾压式的，但代价是 2B 参数。

---

## 4. 训练数据与策略对比

| 属性 | Chinese-CLIP | nomic-embed-vision | Qwen3-VL-Embedding | **我们 v1.0** |
|------|-------------|-------------------|-------------------|-------------|
| **训练数据量** | **~2 亿图文对** ✅ | 未公开（估计千万级） | 未公开（估计千万级） | **18 万** ❌ |
| **数据来源** | 中文互联网（多域）✅ | — | — | ImageNet 中文描述（单域）❌ |
| **域多样性** | 电商/新闻/社交/百科 ✅ | — | — | **仅自然图像** ❌ |
| **训练方法** | 二阶段（先文本→再联合）✅ | **LiT** (锁文本训视觉) ✅ | LLM 预训练+对齐 | LiT 风格 ✅ |
| **冻结文本** | 第一阶段冻结 | ✅ 冻结 | — | ✅ 冻结 |
| **损失函数** | InfoNCE | **Cosine Similarity Loss** | — | InfoNCE ✅ |
| **训练时长** | 未公开 | 未公开 | 未公开 | **几分钟** ❌ |

> **关键差异**：2 亿 vs 18 万，差了 **1000 倍**。这是 Chinese-CLIP 能跑到 Flickr30K R@1 69% 的最大原因。nomic 也同样依赖大量数据（虽然未公开数据量）。

---

## 5. Benchmark 性能对比

### 5.1 ImageNet Zero-shot 分类

| 模型 | ImageNet Zero-shot | 备注 |
|------|------------------|------|
| CLIP ViT-B/32 (原始) | 63.2% | OpenAI 原始 |
| CLIP ViT-L/14 (原始) | 75.6% | OpenAI 原始 |
| **nomic-embed-vision-v1.5** | **71.0%** | 自定义 ViT-B/16 |
| **我们 v1.0** | **未测（预计 ~30-40%）** | ViT-B/32 + 单域数据 |
| Qwen3-VL-Embedding-2B | — | 无此项 |

### 5.2 中文图文检索

| 模型 | MUGE R@1 | Flickr30K-CN R@1 | COCO-CN R@1 |
|------|---------|-----------------|------------|
| **Chinese-CLIP ViT-L** | **59.0** | **69.0** | **64.7** |
| Chinese-CLIP ViT-L-336 | **65.3** (finetune) | **84.4** (finetune) | **80.1** (finetune) |
| Wukong ViT-L | 42.7 | 51.7 | 53.4 |
| R2D2 ViT-L | 49.5 | 60.9 | 56.4 |
| **我们 v1.0** | **未测** | **未测** | **未测** |

### 5.3 MMTEB / MTEB 综合

| 模型 | MMTEB 平均 (Task) | 分类 | 检索 | 聚类 | STS |
|------|-----------------|------|------|------|-----|
| **Qwen3-VL-Embedding-2B** | **63.87** | 65.86 | 67.12 | 52.50 | 74.29 |
| **nomic-embed-text-v1.5** | 62.28 | — | — | — | — |
| nomic-embed-vision-v1.5 | MTEB 62.28 (text only) | — | — | — | — |
| **我们 v1.0** | **未测** | — | — | — | — |

---

## 6. 定位与差距雷达图

用五个维度定位四个模型：

```
                  Chinese-CLIP
                   nomic-vision-v1.5
                   Qwen3-VL-Embed-2B
                   mini-embed-vision v1.0 (我们)
                       
视觉骨干强度  ═══════●═════●════●═════●═══════
                (我们)  (nomic)  (CLIP-L) (Qwen3)
                       
文本能力      ════●══════●═══════●══════════●══
                (我们) (nomic) (Qwen3) (Chinese)
                       
数据量        ═●══════════════●════●══════●═══
               (我们)            (nomic)     (2亿)
                       
任务覆盖      ═●══════●═════════════●═══════●═
               (我们) (nomic)        (CLIP)  (Qwen3)
                       
部署效率      ═══════════●══●═══════●════●═══
                        (我们)     (nomic)   (大模型)
```

---

## 7. 核心结论

### 7.1 我们是唯一「小而偏」的——既是优势也是劣势

| 维度 | 我们的状态 |
|------|-----------|
| **视觉骨干** | 四者中最弱（ViT-B/32，patch32 丢一半细节） |
| **数据量** | 差距最大（18 万 vs 2 亿，差 1000×） |
| **文本编码器** | 中文很强（BGE 最适合中文场景） |
| **部署效率** | 最优（87M 可训练参数，0.5GB） |
| **benchmark** | 为零（还没跑过任何公开标准） |

### 7.2 nomic 给了我们最重要的启示

nomic-embed-vision-v1.5 **和我们几乎一样的可训练参数量（86M vs 87M）**，用 CLIP 双塔架构做到了：
- ImageNet 0-shot **71%**（CLIP ViT-B/32 原始只有 63%）
- MTEB **62.28**（和 Qwen3-VL-2B 的 63.87 只差 1.5 分）

区别在哪？

```
nomic:  自定义 ViT-B/16 (SwiGLU+RoPE+Flash)  +  大量数据  →  71% ImageNet
我们:   标准 ViT-B/32 (GELU+AbsPos)           +  18万单域  →  62% ImageNet-zh val
                   ↑ 架构差距                   ↑ 数据差距
```

**架构和数据各贡献一半**。换 ViT-B/16 + EVA 风格改进本身就能拉近一大半差距。

### 7.3 Chinese-CLIP 告诉我们中文赛道还有空间

Chinese-CLIP 是 2022 年的模型，用 ViT-L/14 + RoBERTa + 2 亿数据就在中文检索上做到 SOTA。**这说明「CLIP 双塔 + 中文数据」这条路线三年了依然不过时**——问题不在架构，在数据。

如果我们要做中文多模态嵌入，Chinese-CLIP 才是最直接的参照系：

| 改进路径 | 预期提升 | 参考 |
|---------|---------|------|
| ViT-B/32 → ViT-B/16 或 EVA02 | +5-10pp | nomic 同样 86M 但 patch16 |
| 18 万 → 100 万+ 中文图文对 | +10-15pp | Chinese-CLIP 2 亿 |
| 添加 MUGE/Flickr30K-CN 评测 | — | 和 Chinese-CLIP 直接对标 |

### 7.4 Qwen3-VL 是下一代的标杆，但不需要现在追

Qwen3-VL-Embedding-2B 在 MMTEB 上 63.87，nomic 在 text-only 上 62.28——差距只有 1.5 分。说明：

- **CLIP 双塔还没到天花板**：好的双塔依然能接近 LLM 统一架构的表现
- **LLM 统一架构的真正优势不在分数**，而在长上下文、多图理解、视频支持这些扩展能力
- **如果要对标 Qwen3-VL，核心瓶颈也不是架构，而是数据**

---

## 8. 建议的演进路径

| 阶段 | 做什么 | 核心变化 | 预期 Flickr30K-CN R@1 | 周期 |
|------|--------|---------|---------------------|------|
| **Phase 0** | 跑完当前 v1.0 的 MMTEB 基线 | — | ~35（估计） | **现在** |
| **Phase 1** | 视觉升级到 ViT-L/14 或 EVA02-Base | 换视觉骨干 | ~45-50 | 1-2 周 |
| **Phase 2** | 数据扩充（添加公开中文图文对 + 继续生成 ImageNet 描述） | 数据量 + 域多样性 | ~55-60 | 1-2 月 |
| **Phase 3** | 训练策略优化（二阶段训练、难负例挖掘） | 训练质量 | ~60-65 | 1-2 月 |
| **Phase ∞** | 架构向 LLM 统一演进 | 换 Qwen3-VL 基座 | 65+ | 需要数据 + 算力 |

**现在最紧迫的**：先把 Phase 0 完成（本周内跑完 MMTEB baseline），让所有讨论有数据支撑。

---

## 9. 数据引用

- Chinese-CLIP: [arXiv:2211.01335](https://arxiv.org/abs/2211.01335), [ModelScope](https://modelscope.cn/models/damo/multi-modal_clip-vit-large-patch14_zh)
- nomic-embed-vision: [arXiv:2406.18587](https://arxiv.org/abs/2406.18587), [Blog](https://blog.nomic.ai/posts/nomic-embed-vision), [GitHub](https://github.com/nomic-ai/contrastors)
- nomic-embed-text: [arXiv:2402.01613](https://arxiv.org/abs/2402.01613)
- Qwen3-VL-Embedding: [arXiv:2601.04720](https://arxiv.org/abs/2601.04720)
- CLIP: [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
- LiT: [arXiv:2111.07991](https://arxiv.org/abs/2111.07991)
