# mini-embed-vision v1.5 数据集制作指南

> 2026-08-21 | EVA02 + BGE 对比学习与轻量图文重排序

## 1. 目标

v1.5 需要同时训练两类能力：

1. **跨模态 embedding**：文本和图像映射到同一个 768 维空间。
2. **图文重排序**：对 embedding 检索得到的 top-k 候选重新打分。

因此数据集要满足三类需求：

| 能力 | 数据需求 | 当前数据是否满足 |
|---|---|---|
| 双塔 InfoNCE | 正样本图文对 | ✅ 满足 |
| 第一版 in-batch rerank | batch 内随机负例 | ✅ 满足 |
| 高质量精排 | query 级 hard negatives | ❌ 需要补充 |

结论：

```text
第一轮训练：直接使用现有 ImageNet-zh captions。
第二轮增强：挖掘并构造显式 hard negative pairs。
```

## 2. 当前数据盘点

### 2.1 数据位置

```text
data/imagenet-1k-zh-captions/
├── dataset_infos.json
├── README.md
└── train-00000-of-00037.parquet
└── ...
```

### 2.2 当前 schema

| 字段 | 类型 | 示例 | 说明 |
|---|---|---|---|
| `image` | Image | JPEG/PIL image | 图片内容 |
| `text` | string | `画面中是一个放置在浅色平面上的工具...` | 中文描述 |
| `label` | int | `726` | ImageNet 类别 ID |
| `label_name` | string | `plane, carpenter's plane...` | 英文类别名 |
| `source_id` | string | `train/.../0` | 原始样本 ID |

当前规模：

```text
183,240 rows
37 parquet shards
约 15.4 GB
```

### 2.3 当前划分方式

代码使用 `source_id` 哈希切分：

```text
train: 98%
val:   2%
```

优点：

- 确定性
- 同一 `source_id` 不会跨 split
- 多次训练划分一致

## 3. 目标数据格式

推荐使用“图片表 + 关系表”两张表，避免重复存储图片。

### 3.1 images 表

文件：

```text
data/v15/images.parquet
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `image_id` | string | ✅ | 全局唯一图片 ID |
| `image` | binary/image | ✅ | 图片 |
| `source_id` | string | ✅ | 原始来源 ID，用于防泄漏 |
| `label` | int | 推荐 | ImageNet 类别 |
| `label_name` | string | 可选 | 类别名 |
| `width` | int | 推荐 | QA 用 |
| `height` | int | 推荐 | QA 用 |
| `hash` | string | 推荐 | 图片去重 |
| `split` | string | ✅ | train / val / test |

示例：

| image_id | source_id | label | split | hash |
|---|---|---:|---|---|
| `img_00000001` | `train/.../0` | 726 | train | `9f2a...` |

### 3.2 pairs 表

文件：

```text
data/v15/pairs.parquet
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `query_id` | string | ✅ | 查询 ID |
| `query` | string | ✅ | 中文查询/描述 |
| `image_id` | string | ✅ | 候选图片 ID |
| `label` | int | ✅ | 1=positive，0=negative |
| `negative_type` | string | 负例必填 | random / same_class / mined / llm_filter |
| `rank` | int | 推荐 | embedding 检索排名 |
| `score` | float | 推荐 | embedding cosine 分数 |
| `split` | string | ✅ | train / val / test |
| `is_hard` | bool | 推荐 | hard negative 标记 |

示例：

| query_id | query | image_id | label | negative_type | rank | score |
|---|---|---|---:|---|---:|---:|
| `q_000001` | `一只橘猫坐在沙发上` | `img_A` | 1 | — | 1 | 0.82 |
| `q_000001` | `一只橘猫坐在沙发上` | `img_B` | 0 | mined | 2 | 0.81 |
| `q_000001` | `一只橘猫坐在沙发上` | `img_C` | 0 | same_class | 12 | 0.74 |

### 3.3 为什么不用一张大表

如果每个 query 挂 8 个负例，图片会被重复保存 9 次：

```text
单表重复图片：约 9× 存储
双表方案：图片只保存 1 次，pair 只存 image_id
```

## 4. 数据集版本规划

### v15-base

用于第一轮训练，验证训练链路和 in-batch reranker。

```text
来源：现有 imagenet-1k-zh-captions
负例：batch 内随机负例
reranker：in-batch BCE
```

不需要额外构造数据。

### v15-hard-v1

用于第二轮训练，提升 reranker 精排能力。

```text
每个 query:
    1 positive
    4~8 mined hard negatives
    1~2 random negatives（防过拟合）
```

预计规模：

```text
180k queries
每 query 6~10 candidates
共 1.1M~1.8M pairs
```

### v15-hard-v2

在 v1.5 初版模型训练后重新挖掘。

```text
每 1~2 epoch 更新 hard negatives
每个 query 保留 top 100 candidates
训练时动态采样 4~8 个负例
```

### rerank-eval

固定验证集，不参与训练。

```text
每个 query:
    1 positive
    99 hard negatives
```

用于报告：

```text
Recall@100 before rerank
R@1 before rerank
R@1 after rerank
MRR@10
nDCG@10
```

## 5. 图片处理规范

### 5.1 基础清洗

处理规则：

1. 全部转 RGB。
2. 去除损坏图片。
3. 去除超小图，例如短边 `< 32px`。
4. 保留原图，不在数据集中提前 resize 到 224。
5. 保存图片哈希，用于识别重复图。

不建议提前 resize：

```text
训练预处理是 Resize(224) + CenterCrop(224)
提前 resize 会永久丢失裁剪信息
```

### 5.2 图片哈希

推荐两级去重：

```text
perceptual hash：粗筛相似图
MD5/BLAKE3：识别完全重复文件
```

示例：

```python
from PIL import Image
import hashlib

def image_fingerprint(image: Image.Image) -> str:
    image = image.convert("RGB")
    return hashlib.blake2b(image.tobytes(), digest_size=16).hexdigest()
```

### 5.3 保留训练必需元数据

必须保留：

```text
image_id
source_id
split
label（如有）
```

`source_id` 是防止数据泄漏的关键字段。

## 6. 文本处理规范

### 6.1 清洗规则

1. 去除 HTML 标签和控制字符。
2. 统一空白字符。
3. 去除空文本和超短文本。
4. 保留中文标点。
5. 不做自动翻译。
6. 不做大小写归一化以外的强改写。

建议过滤：

```text
len(text.strip()) < 5 → 丢弃或人工检查
text 只含占位符 → 丢弃
同一 text 对应过多不同类别 → 标记为弱查询
```

### 6.2 长度分布统计

必须统计：

```text
char length P50/P90/P95/P99/max
token length P50/P90/P95/P99/max
token > 512 的比例
```

BGE 最大长度是 512 tokens。若长文本比例较高，需要评估：

```text
截断到 512
或先做摘要/分段
```

### 6.3 Query 与 Caption 命名

建议明确区分：

| 字段 | 用途 | 例子 |
|---|---|---|
| `caption` | 描述图片本身 | 图片中有一只橘猫坐在沙发上 |
| `query` | 用户检索式表达 | 一只橘猫坐在沙发上 |
| `instruction` | 任务指令（v1.6） | 找到这张图片 |

当前数据只有 `text`，第一轮可作为 caption，也可直接作为 query。第二轮建议生成更自然的短查询。

## 7. 负例构造策略

### 7.1 随机负例

构造：

```text
query_i + image_j, i != j
```

优点：

- 简单
- 几乎无 false negative
- 适合 warm-up

缺点：

- 太简单
- reranker 学不到细粒度区分

### 7.2 同类负例

使用 ImageNet `label`：

```text
query_i 的正例类别是 A
选择另一张也属于 A 的图片作为负例
```

风险：

```text
如果 caption 描述很泛化，同类图片也可能是正例
```

因此必须区分：

| caption 类型 | 同类图可否当负例 |
|---|---|
| 描述具体动作/位置/颜色/数量 | 通常可以 |
| 描述整个物体的大类 | 不安全 |
| 只描述类别名 | 不安全 |

建议：

```text
same_class negative 必须经过文本属性过滤
不能仅凭 label 相同就标记为 0
```

### 7.3 Embedding 挖掘负例

这是 v1.5 重点。

流程：

```text
1. 用 baseline 模型编码所有 images
2. 用 baseline 模型编码所有 queries
3. 对每个 query 计算 cosine top 100
4. 去掉 positive image
5. 去掉同 source_id
6. 去掉 exact duplicate / perceptual duplicate
7. 从 top 10~100 采样 hard negatives
8. 记录 rank、score、negative_type
```

推荐采样：

```text
top 5~30：最难负例
top 31~100：中等难负例
再混少量随机负例
```

不建议只取 top 1~5：

```text
最 top 的候选中 false positive 比例更高
训练信号噪声大
```

### 7.4 属性约束负例

适合后续增强：

| 查询属性 | 可选负例 |
|---|---|
| 颜色 | 改变颜色但同类别 |
| 数量 | 改变数量 |
| 动作 | 改变动作 |
| 位置 | 改变对象位置 |
| 材质 | 改变材质 |

适合用生成式模型或人工构造，但必须有质量审核。

### 7.5 False negative 防护

必须执行的规则：

1. 正例图片不能作为负例。
2. 同 `source_id` 不能作为负例。
3. exact duplicate 不能作为负例。
4. perceptual duplicate 需要过滤。
5. 同类别图片需谨慎。
6. 挖掘结果 top 1~3 建议人工抽检。
7. LLM/规则过滤时保留原因字段。

建议 schema 扩展：

| 字段 | 说明 |
|---|---|
| `filter_reason` | 为什么允许作为负例 |
| `annotator` | auto / human / model |
| `needs_review` | 是否需要人工检查 |

## 8. Hard Negative 挖掘流程

### 8.1 生成图片索引

```text
Input: images.parquet
Output: image_embeddings.npy, image_ids.json
```

要求：

```text
batch encoding
float32
embedding L2 normalized
保存 image_id 与 row index 映射
```

### 8.2 生成 query 索引

```text
Input: pairs.parquet 中 label=1 的 query
Output: query_embeddings.npy, query_ids.json
```

### 8.3 检索候选

对每个 query：

```text
top_k = 100
去掉 positive image
保留 rank 和 score
```

### 8.4 负例采样

训练集采样：

```text
hard pool = rank 10~100
每 query 采样 4~8 个
另加 1~2 个随机负例
```

验证集：

```text
每 query 固定 99 个 hard negatives
保存完整 rank，不做随机采样
```

### 8.5 动态挖掘

第一版可以静态挖掘。训练稳定后升级：

```text
每 1~2 epoch 重新编码
更新 top 100 pool
下一次 epoch 从 pool 采样
```

动态挖掘要求：

- 保存每轮 embedding 版本
- 保存采样随机种子
- pair 中记录 `mined_epoch`
- 保证同一 epoch 内正例不重复入负例池

## 9. 训练集 / 验证集 / 测试集

### 9.1 Split 单位

必须按 `source_id` 切分，不能按 row 切分。

原因：

```text
同一 source_id 可能派生多个 query、图片增强或 caption
按 row 切分会造成泄漏
```

推荐：

| split | 比例 |
|---|---:|
| train | 98% |
| val | 1% |
| test | 1% |

如果数据量扩大后：

| split | 比例 |
|---|---:|
| train | 97% |
| val | 1.5% |
| test | 1.5% |

### 9.2 Rerank 验证集

不能只用随机负例。

建议：

```text
每 query 1 positive + 99 mined hard negatives
```

并额外准备：

```text
1% human reviewed queries
```

用于校准自动挖掘质量。

### 9.3 泄漏检查

训练前必须检查：

```text
1. source_id 不跨 train/val/test
2. image_id 不跨 split
3. query_id 不跨 split
4. positive image 不出现在同 query 负例中
5. exact duplicate 图片不跨 split
6. perceptual duplicate 不进入同一 query 的负例池
```

## 10. 质量统计与 QA

### 10.1 基础统计

图片：

```text
总数
损坏数
重复数
宽高分布
短边 < 32 数量
灰度图/RGB 分布
```

文本：

```text
平均长度
token P50/P90/P95/P99
空文本数量
重复 query 数量
每 query candidate 数
```

Pair：

```text
正负比例
hard negative 占比
每 query 负例数分布
candidate label 分布
embedding score 分布
positive score vs negative score 分离度
```

### 10.2 必跑 QA 检查

```text
[ ] image_id 唯一
[ ] query_id + image_id 唯一
[ ] 每个 query 至少 1 个 positive
[ ] 每个 query 有足够 negatives
[ ] label 只能是 0/1
[ ] negative_type 在允许枚举内
[ ] split 一致
[ ] source_id 不泄漏
[ ] 正例不在负例列表
[ ] embedding shape = [N,768]
[ ] embedding 全 finite
[ ] embedding L2 norm ≈ 1
```

### 10.3 抽样人工检查

每类至少抽查：

```text
10 个 mined top 1
10 个 mined top 10
10 个 same_class
10 个 random
```

重点判断：

```text
负例是否其实也满足 query
query 是否过于泛化
caption 是否包含图片不存在的信息
```

## 11. 目录结构建议

```text
data/v15/
├── images.parquet
├── pairs.parquet
├── metadata.json
├── stats.json
├── qa_report.json
├── indexes/
│   ├── image_ids.json
│   ├── query_ids.json
│   ├── image_embeddings.npy
│   └── query_embeddings.npy
└── mined/
    ├── candidates_top100.parquet
    └── negatives_v1.parquet
```

`metadata.json` 示例：

```json
{
  "version": "v15-hard-v1",
  "base_dataset": "imagenet-1k-zh-captions",
  "num_images": 183240,
  "num_queries": 183240,
  "num_pairs": 1282680,
  "embed_dim": 768,
  "max_text_len": 512,
  "negative_sampling": {
    "hard_per_query": 6,
    "random_per_query": 1,
    "hard_rank_range": [10, 100]
  },
  "split_by": "source_id"
}
```

## 12. 伪代码

### 12.1 图片表构建

```python
import hashlib
import uuid
from PIL import Image

rows = []

for row in source_dataset:
    image = row["image"]
    if image is None or image.width < 32 or image.height < 32:
        continue

    fingerprint = hashlib.blake2b(
        image.convert("RGB").tobytes(),
        digest_size=16,
    ).hexdigest()

    rows.append({
        "image_id": f"img_{len(rows):08d}",
        "image": image,
        "source_id": row["source_id"],
        "label": row["label"],
        "label_name": row["label_name"],
        "width": image.width,
        "height": image.height,
        "hash": fingerprint,
        "split": split_by_source_id(row["source_id"]),
    })
```

### 12.2 正样本 pair 构建

```python
pairs = []

for image_row in image_rows:
    pairs.append({
        "query_id": f"q_{len(pairs):08d}",
        "query": image_row["caption"],
        "image_id": image_row["image_id"],
        "label": 1,
        "negative_type": None,
        "rank": 1,
        "score": None,
        "split": image_row["split"],
    })
```

### 12.3 挖掘负例

```python
import numpy as np

image_matrix = np.load("indexes/image_embeddings.npy")
query_matrix = np.load("indexes/query_embeddings.npy")

scores = query_matrix @ image_matrix.T
top_indices = np.argsort(-scores, axis=1)[:, :100]

negative_rows = []

for query_index, candidate_indices in enumerate(top_indices):
    query_id = query_ids[query_index]
    positive_id = positives[query_id]
    source_id = query_source_ids[query_id]

    for rank, candidate_index in enumerate(candidate_indices, start=1):
        candidate_id = image_ids[candidate_index]
        if candidate_id == positive_id:
            continue
        if image_source_ids[candidate_id] == source_id:
            continue
        if image_hashes[candidate_id] == positive_hashes[query_id]:
            continue

        negative_rows.append({
            "query_id": query_id,
            "query": query_texts[query_id],
            "image_id": candidate_id,
            "label": 0,
            "negative_type": "mined",
            "rank": rank,
            "score": float(scores[query_index, candidate_index]),
            "split": query_splits[query_id],
        })
```

### 12.4 训练采样

```python
hard_pool = [
    row for row in negatives
    if row["query_id"] == query_id
    and 10 <= row["rank"] <= 100
]

selected = random.sample(
    hard_pool,
    k=min(6, len(hard_pool)),
)
```

## 13. 后续新增域建议

ImageNet 是自然图像单域。为了提升 benchmark 泛化，建议逐步加入：

| 域 | 价值 | 注意 |
|---|---|---|
| 电商商品图 | 属性、品牌、材质检索 | 中文电商词典 |
| 文档/截图 | UI、图表、文档检索 | OCR 依赖 |
| 新闻图片 | 事件和实体理解 | 时效与版权 |
| 表情包/梗图 | 非正式文化语义 | 标注复杂 |
| 中文地标 | 地理实体检索 | 类别不平衡 |
| 食物 | 中文食物细分类 | 细粒度高 |
| 人物动作 | 动作/场景检索 | 隐私风险 |

优先级建议：

```text
1. 电商
2. 文档/截图
3. 新闻
4. 食物/地标
```

## 14. 交付验收

数据集交付前必须提供：

```text
images.parquet
pairs.parquet
metadata.json
stats.json
qa_report.json
embedding index
随机种子
数据字典
泄漏检查结果
人工抽检记录
```

验收标准：

| 指标 | 要求 |
|---|---:|
| source_id 跨 split 数量 | 0 |
| image_id 重复数量 | 0 |
| pair 主键重复 | 0 |
| query 无 positive 数量 | 0 |
| embedding NaN/Inf | 0 |
| positive 误入负例 | 0 |
| rerank val hard negative 数/query | 99 |
| 人工抽检 false negative 率 | < 2%（建议） |

## 15. 推荐执行顺序

```text
Step 1  现有数据跑通 v15-base
Step 2  生成 images.parquet + pairs.parquet
Step 3  编码 image/query embeddings
Step 4  挖掘 top100 candidates
Step 5  构造 v15-hard-v1
Step 6  构造 1+99 rerank-eval
Step 7  QA 与人工抽检
Step 8  用 hard negatives 训练 reranker
Step 9  对比 before/after rerank
Step 10 动态挖掘升级 v15-hard-v2
```
