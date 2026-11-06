# mini-embed-vision v1.5 使用指南

> 2026-08-21 | EVA02-B/14 + BGE-base-zh + Cross-Modal Reranker

## 1. 模型目录

当前工作目录：

```bash
cd /mnt/data3/runke.zhong/workspace/mini-embed-vision
```

模型目录：

```text
mini-embed-vision-v1.5/
├── config.json               # 模型结构配置，内嵌 BGE text_config
├── model.py                  # 自定义模型代码
├── model.safetensors         # 统一权重：BGE + EVA02 + projection + reranker
├── preprocessor_config.json  # EVA02 图像预处理参数
├── tokenizer.json
└── tokenizer_config.json
```

该目录是 self-contained 权重目录：

- 不需要再下载 BGE
- 不需要再下载 EVA02
- 不需要额外加载原始 checkpoint
- 不需要 Hugging Face 网络

## 2. 环境要求

推荐 Python 3.10+。

最低核心依赖：

```text
torch
torchvision
transformers
safetensors
pillow
timm
numpy
```

已在当前训练环境验证过的一组版本：

```text
torch==2.11.0+cu130
torchvision==0.26.0+cu130
transformers==5.10.2
timm==1.0.28
safetensors==0.7.0
```

`requirements.txt` 当前锁定的是训练前环境版本；如果部署环境解析依赖冲突，优先保证上表组合可用。

如果使用当前项目环境：

```bash
cd /mnt/data3/runke.zhong/workspace/mini-embed-vision
python -m pip install -r requirements.txt
```

## 3. 快速加载

```python
import os
import torch
from transformers import AutoModel, AutoTokenizer

model_dir = os.path.abspath("mini-embed-vision-v1.5")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    local_files_only=True,
)

model = AutoModel.from_pretrained(
    model_dir,
    trust_remote_code=True,
    local_files_only=True,
).to(device)
model.eval()
```

`model_dir` 可以换成绝对路径，例如：

```python
model_dir = "/mnt/data3/runke.zhong/workspace/mini-embed-vision/mini-embed-vision-v1.5"
```

`/home/runke.zhong/workspace/mini-embed-vision/mini-embed-vision-v1.5` 和上述路径当前指向同一个目录。

如果从其他目录运行 `RetrievalRerankPipeline`，请先将仓库根目录加入 `PYTHONPATH`：

```bash
export PYTHONPATH=/mnt/data3/runke.zhong/workspace/mini-embed-vision:$PYTHONPATH
```

或直接复制项目根目录下的 `pipeline.py`。只做 embedding 编码时，模型目录本身已经自包含，不依赖仓库代码。

建议离线环境同时设置：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

## 4. 图像预处理

必须使用导出模型配套的 EVA02 预处理：

```python
from torchvision import transforms

image_transform = transforms.Compose(
    [
        transforms.Resize(
            224,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ]
)
```

不要使用 CLIP ViT-B/32 的 processor。

## 5. 文本嵌入

```python
texts = [
    "一只橘猫坐在沙发上",
    "一张包含表格和图表的截图",
]

encoded = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)
input_ids = encoded["input_ids"].to(device)
attention_mask = encoded["attention_mask"].to(device)

with torch.inference_mode():
    text_embeddings = model.encode_text(
        input_ids,
        attention_mask,
    )

print(text_embeddings.shape)  # torch.Size([2, 768])
print(text_embeddings.norm(dim=-1))  # 约等于 1
```

## 6. 图像嵌入

```python
from PIL import Image

image = Image.open("your_image.jpg").convert("RGB")
pixel_values = image_transform(image).unsqueeze(0).to(device)

with torch.inference_mode():
    image_embeddings = model.encode_image(pixel_values)

print(image_embeddings.shape)  # torch.Size([1, 768])
print(image_embeddings.norm(dim=-1))  # 约等于 1
```

批量处理：

```python
images = [
    Image.open("a.jpg").convert("RGB"),
    Image.open("b.jpg").convert("RGB"),
]
pixel_values = torch.stack([image_transform(img) for img in images]).to(device)

with torch.inference_mode():
    image_embeddings = model.encode_image(pixel_values)
```

## 7. 跨模态检索

文本查图像：

```python
retrieval_scores = text_embeddings @ image_embeddings.t()
top_k = torch.topk(retrieval_scores, k=10, dim=1)

print(top_k.indices)
print(top_k.values)
```

大规模图像库建议：

1. 离线预计算全部 image embeddings。
2. 保存为 `float32` 或 `float16`。
3. 使用 Faiss / Milvus / Elasticsearch dense vector 做第一-stage 召回。
4. 召回 top 100~1000。
5. 用 reranker 精排。

注意：`RetrievalRerankPipeline` 会返回全库 cosine 分数矩阵 `[Q, N]`，适合中小规模库。生产环境大规模图像库应优先使用 Faiss / Milvus / Elasticsearch dense vector 做第一-stage 召回。

## 8. 检索+重排序 Pipeline

项目已提供：

```python
from pipeline import RetrievalRerankPipeline
```

### 8.1 完整示例

```python
import os
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer
from pipeline import RetrievalRerankPipeline

model_dir = os.path.abspath("mini-embed-vision-v1.5")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    local_files_only=True,
)
model = AutoModel.from_pretrained(
    model_dir,
    trust_remote_code=True,
    local_files_only=True,
).to(device)
model.eval()

image_transform = transforms.Compose(
    [
        transforms.Resize(
            224,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ]
)

queries = ["一只橘猫坐在沙发上", "黑色行李箱"]
images = [
    Image.open("cat.jpg").convert("RGB"),
    Image.open("suitcase.jpg").convert("RGB"),
]

encoded = tokenizer(
    queries,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)
pixel_values = torch.stack([image_transform(img) for img in images])

with torch.inference_mode():
    query_embeddings = model.encode_text(
        encoded["input_ids"].to(device),
        encoded["attention_mask"].to(device),
    )
    image_embeddings = model.encode_image(pixel_values.to(device))

pipeline = RetrievalRerankPipeline(model, device=device)
final_indices, final_scores, candidates, retrieval_scores = pipeline(
    query_embeddings,
    image_embeddings,
    top_k=10,
    rerank_k=100,
)

print("最终图像 ID:", final_indices.cpu())
print("最终分数:", final_scores.cpu())
```

### 8.2 返回值说明

| 返回值 | 形状 | 含义 |
|---|---:|---|
| `final_indices` | `[Q, top_k]` | rerank 后的图像索引 |
| `final_scores` | `[Q, top_k]` | reranker 分数 |
| `candidates` | `[Q, rerank_k]` | 第一-stage 召回候选 |
| `retrieval_scores` | `[Q, N]` | 全库 cosine 分数 |

### 8.3 rerank_k 设置建议

| 场景 | 建议值 |
|---|---:|
| 小库测试 | 100 |
| 生产检索 | 200~1000 |
| 强召回需求 | 1000+ |

`rerank_k` 越大，召回覆盖率越高，但精排耗时也越高。

## 9. 只使用 Reranker

如果已经有第一-stage 候选的 embeddings，可以只调用 reranker：

```python
with torch.inference_mode():
    scores = model.score_rerank(
        query_embeddings,
        candidate_image_embeddings,
    )

print(scores.shape)
```

输入形状：

```text
query_embeddings:               [Q, 768]
candidate_image_embeddings:     [Q or 1, C, 768]
```

输出：

```text
 scores: [Q, C]
```

通常一个 query 时，candidate 形状用 `[1, C, 768]`。

## 10. 与 Faiss 集成示例

```python
import faiss
import numpy as np

image_matrix = image_embeddings.cpu().numpy().astype("float32")
faiss.normalize_L2(image_matrix)

index = faiss.IndexFlatIP(image_matrix.shape[1])
index.add(image_matrix)

query_matrix = query_embeddings.cpu().numpy().astype("float32")
faiss.normalize_L2(query_matrix)

retrieval_scores, candidates = index.search(query_matrix, 100)

candidate_images = torch.from_numpy(image_matrix[candidates]).to(device)
rerank_scores = model.score_rerank(
    query_embeddings,
    candidate_images,
)

relative_order = torch.argsort(
    rerank_scores,
    dim=1,
    descending=True,
)
final_indices = torch.gather(
    torch.from_numpy(candidates).to(device),
    1,
    relative_order,
)

说明：

- 需要额外安装 `faiss-cpu` 或 `faiss-gpu`。
- 示例中的 `rerank_scores` 是 reranker 的未校准分数，只用于同一 query 内部排序，不要当作概率解释。
- `IndexFlatIP` 是精确检索；百万级以上库建议使用 `IVF`、`HNSW` 或向量数据库索引。
```

## 11. Embedding 保存与读取

保存：

```python
image_embeddings = image_embeddings.float().cpu().numpy()
np.save("image_embeddings.npy", image_embeddings)
```

读取：

```python
image_embeddings = np.load("image_embeddings.npy")
image_embeddings = torch.from_numpy(image_embeddings).to(device)
```

建议：

- 检索库保存 `float32`
- 大规模库可保存 `float16`，检索前转 `float32`
- embedding 必须 L2 normalized

## 12. 性能建议

### 12.1 batch size

| GPU 显存 | 建议 batch size |
|---|---:|
| 16GB | 8~16 |
| 24GB | 16~32 |
| 40GB+ | 32~64 |

### 12.2 混合精度

GPU 推理建议：

```python
with torch.inference_mode(), torch.autocast(
    device_type="cuda",
    dtype=torch.bfloat16,
):
    embeddings = model.encode_image(pixel_values)
```

如需降低显存占用，可加载后转 `bfloat16`：

```python
model = AutoModel.from_pretrained(
    model_dir,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
).to(device).eval()
```

建议先用验证集确认指标变化。EVA02 RoPE 对数值精度较敏感，不建议在未验证的情况下使用 `float16`。

### 12.3 CPU 推理

CPU 可运行，但速度较慢。建议：

```text
batch size 1~8
优先只做小规模验证
```

## 13. 输入限制

| 输入 | 限制 |
|---|---:|
| 文本长度 | 512 tokens |
| 图像输入 | 3×224×224 |
| embedding 维度 | 768 |
| 图像格式 | RGB |

## 14. 常见问题

### 14.0 模型加载慢或提示执行自定义代码

这是 Hugging Face `trust_remote_code` 的正常提示。`mini-embed-vision-v1.5/model.py` 是必需的自定义模型代码，请确认传入 `trust_remote_code=True`。

### 14.1 是否需要加载 BGE 或 EVA02？

不需要。`model.safetensors` 已包含全部权重。

### 14.2 是否需要联网？

不需要。使用：

```python
local_files_only=True
```

或：

```bash
export HF_HUB_OFFLINE=1
```

### 14.3 能否只用文本或只用图像？

可以：

```python
model.encode_text(...)
model.encode_image(...)
```

两侧输出都在同一个 768 维空间。

### 14.4 Reranker 能否单独使用？

可以。只要有第一-stage 的 query/image embeddings：

```python
model.score_rerank(query_embeddings, candidate_image_embeddings)
```

### 14.5 为什么 retrieval 和 rerank 结果相同？

如果候选本身很简单，embedding top1 已经正确，rerank 无法再提升。请使用 hard negative 数据验证 rerank 收益。

### 14.6 rerank 分数和 cosine 分数可以混用吗？

不可以。cosine 分数用于双塔召回；reranker 分数只用于同一 query 的候选内部排序。两者尺度不同，不能直接相加或比较。

## 15. 训练指标

当前最佳 checkpoint：

| Metric | Value |
|---|---:|
| Best Epoch | 9 |
| Val R@1 | 0.7361 |
| Val R@5 | 0.9355 |
| Val R@10 | 0.9657 |

抽样前 256 条验证数据：

| Metric | Value |
|---|---:|
| Embedding R@1 | 0.9219 |
| Rerank R@1 | 0.9219 |
| Recall@100 | 1.0000 |

该子集偏简单，完整 benchmark 和 hard negative 评测仍需后续执行。

## 16. 最小可用示例

```python
import os
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer

model_dir = os.path.abspath("mini-embed-vision-v1.5")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
model = AutoModel.from_pretrained(
    model_dir,
    trust_remote_code=True,
    local_files_only=True,
).to(device).eval()

transform = transforms.Compose(
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

text = "一只橘猫坐在沙发上"
image = Image.open("cat.jpg").convert("RGB")

encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
pixel_values = transform(image).unsqueeze(0)

with torch.inference_mode():
    text_embedding = model.encode_text(
        encoded["input_ids"].to(device),
        encoded["attention_mask"].to(device),
    )
    image_embedding = model.encode_image(pixel_values.to(device))

score = (text_embedding @ image_embedding.t()).item()
print(score)
```
