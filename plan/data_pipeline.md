# mini-embed-vision v1.5 数据集处理 Pipeline 草案

## 1. Pipeline 总览

```
原始数据源 ──► 统一格式转换 ──► 质量过滤 ──► 去重 ──► 难负例挖掘 ──► 训练数据集
   │               │               │           │            │
   │               │               │           │            └─ 生成三元组 (anchor, pos, hard_neg)
   │               │               │           └─ 图像 pHash + 文本 MinHash
   │               │               └─ 图像质量 + 文本质量 + 语言检测
   │               └─ 转为统一 JSONL 格式
   └─ Wukong / COCO-CN / LAION-CN / 自有数据
```

统一数据格式（JSONL，每行一条）：
```json
{
  "id": "wukong_000001",
  "image_path": "images/wukong_000001.jpg",
  "text": "一只橘色的猫趴在沙发上睡觉",
  "source": "wukong",
  "metadata": {
    "width": 800,
    "height": 600,
    "text_length": 14,
    "language": "zh",
    "language_ratio": 0.95
  }
}
```

---

## 2. 模块设计

### 2.1 数据源接入层 (`sources/`)

每个数据源一个 loader，输出统一 JSONL。

```python
# sources/base.py
from abc import ABC, abstractmethod
from typing import Iterator, Dict

class BaseSource(ABC):
    """数据源基类"""

    @abstractmethod
    def iter_samples(self) -> Iterator[Dict]:
        """
        输出统一格式样本:
        {
            "id": str,
            "image_path": str,   # 原始路径（后续会被处理）
            "text": str,
            "source": str,
        }
        """
        ...

    @abstractmethod
    def estimate_total(self) -> int:
        """预估总样本数，用于进度条"""
        ...
```

```python
# sources/wukong.py
import json, os
from .base import BaseSource

class WukongSource(BaseSource):
    """
    Wukong 数据集加载器
    数据格式: 多个 tar 文件 + 对应的 JSON 标注
    下载: https://wukong-dataset.github.io/wukong-dataset/
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def iter_samples(self):
        for jsonl_file in sorted(glob(os.path.join(self.data_dir, "*.jsonl"))):
            with open(jsonl_file) as f:
                for line in f:
                    item = json.loads(line)
                    yield {
                        "id": f"wukong_{item['image_id']}",
                        "image_path": item["image_url"],  # 需要先下载
                        "text": item["caption"],
                        "source": "wukong",
                    }

    def estimate_total(self):
        # Wukong 约 100M
        return 100_000_000
```

```python
# sources/coco_cn.py
class COCOCNSource(BaseSource):
    """
    COCO-CN 数据集加载器
    下载: https://github.com/li-xirong/coco-cn
    """
    def __init__(self, annotation_file: str, image_dir: str):
        self.annotation_file = annotation_file
        self.image_dir = image_dir

    def iter_samples(self):
        with open(self.annotation_file) as f:
            data = json.load(f)
        for item in data:
            yield {
                "id": f"coco_cn_{item['image_id']}",
                "image_path": os.path.join(self.image_dir, item["file_name"]),
                "text": item["caption"],
                "source": "coco_cn",
            }

    def estimate_total(self):
        return 30_000
```

```python
# sources/laion_cn.py
class LAIONCNSource(BaseSource):
    """
    LAION-5B 中文子集
    通过 img2dataset 下载后加载 parquet 文件
    过滤条件: language == "zh" 且 language_probability > 0.7
    """
    def __init__(self, parquet_dir: str):
        self.parquet_dir = parquet_dir

    def iter_samples(self):
        import pyarrow.parquet as pq
        for pf in sorted(glob(os.path.join(self.parquet_dir, "*.parquet"))):
            table = pq.read_table(pf)
            for row in table.to_pydict():
                if row.get("language") != "zh":
                    continue
                if row.get("language_probability", 0) < 0.7:
                    continue
                yield {
                    "id": f"laion_{row['uid']}",
                    "image_path": row.get("image_path", ""),
                    "text": row.get("text", ""),
                    "source": "laion_cn",
                }

    def estimate_total(self):
        return 10_000_000  # 过滤后估计
```

```python
# sources/synthetic.py
class SyntheticSource(BaseSource):
    """
    合成数据源：用 VLM 给图像生成中文描述
    支持本地 VLM（Qwen2-VL / InternVL）或 API
    """
    def __init__(self, image_dir: str, vlm_model: str = "Qwen/Qwen2-VL-2B-Instruct"):
        self.image_dir = image_dir
        self.vlm_model = vlm_model

    def iter_samples(self):
        # 遍历图像目录，用 VLM 生成描述
        # 返回 {"id": ..., "image_path": ..., "text": vlm_generated_caption, "source": "synthetic"}
        ...
```

---

### 2.2 质量过滤模块 (`filters/`)

```python
# filters/image_filter.py
from PIL import Image
import imagehash
import numpy as np
from dataclasses import dataclass

@dataclass
class ImageFilterConfig:
    min_width: int = 200
    min_height: int = 200
    max_aspect_ratio: float = 5.0        # 长宽比上限
    min_file_size_kb: int = 5            # 最小文件大小
    max_file_size_mb: float = 10.0       # 最大文件大小
    blur_threshold: float = 50.0         # Laplacian 方差阈值（低于此值判定模糊）
    enable_blur_detect: bool = False     # 模糊检测较慢，大数据集默认关闭
    enable_phash_dedup: bool = True
    phash_threshold: int = 8             # pHash 汉明距离阈值

class ImageFilter:
    def __init__(self, config: ImageFilterConfig = None):
        self.config = config or ImageFilterConfig()
        self._phash_cache = {}  # hash -> sample_id

    def check(self, image_path: str) -> tuple[bool, str]:
        """返回 (通过, 原因)"""
        cfg = self.config

        # 1. 文件大小检查（快速失败）
        file_size = os.path.getsize(image_path)
        if file_size < cfg.min_file_size_kb * 1024:
            return False, "file_too_small"
        if file_size > cfg.max_file_size_mb * 1024 * 1024:
            return False, "file_too_large"

        try:
            img = Image.open(image_path)
            w, h = img.size
        except Exception:
            return False, "corrupt_image"

        # 2. 分辨率检查
        if w < cfg.min_width or h < cfg.min_height:
            return False, "resolution_too_low"

        # 3. 长宽比检查
        aspect = max(w, h) / min(w, h)
        if aspect > cfg.max_aspect_ratio:
            return False, "extreme_aspect_ratio"

        # 4. 模糊检测（可选，较慢）
        if cfg.enable_blur_detect:
            import cv2
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            if variance < cfg.blur_threshold:
                return False, "blurry"

        # 5. pHash 去重
        if cfg.enable_phash_dedup:
            h = imagehash.phash(img)
            for existing_hash, existing_id in self._phash_cache.items():
                if h - existing_hash < cfg.phash_threshold:
                    return False, f"duplicate_of_{existing_id}"
            self._phash_cache[h] = os.path.basename(image_path)

        return True, "ok"
```

```python
# filters/text_filter.py
import re
from dataclasses import dataclass

@dataclass
class TextFilterConfig:
    min_length: int = 5              # 最少中文字符数
    max_length: int = 512            # 最大字符数（超过截断而非丢弃）
    min_zh_ratio: float = 0.3        # 中文字符最低占比
    max_digit_ratio: float = 0.5     # 数字最大占比（过滤价格、编号等）
    max_repeat_char: int = 5         # 单字符最大连续重复次数
    max_repeat_substr: int = 3       # 子串最大连续重复次数
    blocked_patterns: list = None    # 自定义屏蔽模式

class TextFilter:
    # 中文字符范围
    ZH_PATTERN = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')
    DIGIT_PATTERN = re.compile(r'\d')

    def __init__(self, config: TextFilterConfig = None):
        self.config = config or TextFilterConfig()

    def check(self, text: str) -> tuple[bool, str]:
        """返回 (通过, 原因)"""
        cfg = self.config

        if not text or not text.strip():
            return False, "empty_text"

        # 1. 长度检查
        zh_chars = self.ZH_PATTERN.findall(text)
        if len(zh_chars) < cfg.min_length:
            return False, "too_few_chinese_chars"

        if len(text) > cfg.max_length:
            # 不丢弃，后续会截断
            pass

        # 2. 中文占比
        zh_ratio = len(zh_chars) / max(len(text), 1)
        if zh_ratio < cfg.min_zh_ratio:
            return False, f"low_zh_ratio_{zh_ratio:.2f}"

        # 3. 数字占比（过滤 "售价 299 元，编号 #12345..." 这类无意义文本）
        digits = self.DIGIT_PATTERN.findall(text)
        digit_ratio = len(digits) / max(len(text), 1)
        if digit_ratio > cfg.max_digit_ratio:
            return False, "high_digit_ratio"

        # 4. 连续重复字符 (aaa, 。。。)
        if re.search(r'(.)\1{' + str(cfg.max_repeat_char) + r',}', text):
            return False, "repeated_char"

        # 5. 连续重复子串 (好好学习好好学习)
        if re.search(r'(.{2,})\1{' + str(cfg.max_repeat_substr - 1) + r',}', text):
            return False, "repeated_substring"

        # 6. 自定义屏蔽
        if cfg.blocked_patterns:
            for pat in cfg.blocked_patterns:
                if re.search(pat, text):
                    return False, f"blocked_pattern"

        return True, "ok"
```

```python
# filters/text_dedup.py
"""文本去重：MinHash + LSH"""
from datasketch import MinHash, MinHashLSH

class TextDeduplicator:
    def __init__(self, num_perm: int = 128, jaccard_threshold: float = 0.7):
        self.lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=num_perm)
        self.num_perm = num_perm
        self._count = 0

    def _create_minhash(self, text: str) -> MinHash:
        # 使用字符级 3-gram
        mh = MinHash(num_perm=self.num_perm)
        tokens = set(text[i:i+3] for i in range(len(text) - 2))
        for t in tokens:
            mh.update(t.encode('utf-8'))
        return mh

    def is_duplicate(self, sample_id: str, text: str) -> bool:
        mh = self._create_minhash(text)
        if self.lsh.query(mh):
            return True
        self.lsh.insert(sample_id, mh)
        self._count += 1
        return False
```

---

### 2.3 数据转换模块 (`transforms/`)

将过滤后的数据转为模型训练格式，支持多种任务类型。

```python
# transforms/converter.py
import json, os
from enum import Enum

class TaskType(Enum):
    CONTRASTIVE = "contrastive"      # 标准图文对比学习
    TRIPLET = "triplet"              # (anchor, positive, hard_negative)
    INSTRUCTION = "instruction"      # 带指令的嵌入

def convert_to_training_format(
    input_jsonl: str,
    output_dir: str,
    task_type: TaskType = TaskType.CONTRASTIVE,
    max_text_length: int = 256,
    instruction_templates: list[str] = None,
):
    """
    将统一格式转为训练格式

    对比学习格式 (JSONL):
    {"image_path": "...", "text": "...", "task": "contrastive"}

    指令感知格式 (JSONL):
    {"image_path": "...", "text": "查询：描述这张图片的内容", "instruction": "image_retrieval", "task": "instruction"}
    {"text": "一只猫", "instruction": "text_retrieval", "task": "instruction"}
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"train_{task_type.value}.jsonl")

    instruction_templates = instruction_templates or [
        "检索与以下描述匹配的图片",
        "找到描述此场景的图像",
        "根据文本查询相关图片",
    ]

    stats = {"total": 0, "written": 0}
    with open(input_jsonl) as fin, open(output_file, 'w') as fout:
        for line in fin:
            sample = json.loads(line)
            stats["total"] += 1

            if task_type == TaskType.CONTRASTIVE:
                out = {
                    "image_path": sample["image_path"],
                    "text": sample["text"][:max_text_length],
                    "task": "contrastive",
                }
            elif task_type == TaskType.INSTRUCTION:
                import random
                instruction = random.choice(instruction_templates)
                out = {
                    "image_path": sample["image_path"],
                    "text": sample["text"][:max_text_length],
                    "instruction": instruction,
                    "task": "instruction",
                }
            else:
                continue

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            stats["written"] += 1

    print(f"转换完成: {stats['written']}/{stats['total']} 样本")
    return stats
```

---

### 2.4 难负例挖掘 (`mining/`)

```python
# mining/hard_negative.py
"""
难负例挖掘策略:
1. 文本相似但图像不匹配（基于 BM25 / embedding 相似度）
2. 同类目但不同具体内容（基于聚类）
3. 随机采样（作为 baseline 对比）
"""

import numpy as np
from typing import List, Dict, Tuple

class HardNegativeMiner:
    def __init__(self, text_encoder, image_encoder, device="cuda"):
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.device = device

    def mine_by_text_similarity(
        self,
        samples: List[Dict],
        top_k: int = 5,
        batch_size: int = 256,
    ) -> List[Dict]:
        """
        对每个样本，找到文本最相似但图像不匹配的样本作为难负例。
        输出三元组格式: {anchor_text, pos_image, neg_image, neg_text}
        """
        # 1. 编码所有文本
        all_text_embs = []
        for i in range(0, len(samples), batch_size):
            batch_texts = [s["text"] for s in samples[i:i+batch_size]]
            # encode with text_encoder
            embs = self._encode_texts(batch_texts)
            all_text_embs.append(embs)
        all_text_embs = np.concatenate(all_text_embs, axis=0)

        # 2. 对每个样本找 top-k 相似文本
        triplets = []
        for idx, sample in enumerate(samples):
            sims = all_text_embs[idx] @ all_text_embs.T
            top_indices = np.argsort(sims)[::-1]

            neg_count = 0
            for neg_idx in top_indices:
                if neg_idx == idx:
                    continue
                if neg_count >= top_k:
                    break
                triplets.append({
                    "anchor_text": sample["text"],
                    "pos_image": sample["image_path"],
                    "neg_image": samples[neg_idx]["image_path"],
                    "neg_text": samples[neg_idx]["text"],
                    "similarity": float(sims[neg_idx]),
                })
                neg_count += 1

        return triplets

    def mine_by_cross_modal(
        self,
        samples: List[Dict],
        margin: float = 0.1,
        batch_size: int = 256,
    ) -> List[Dict]:
        """
        跨模态难负例：用当前模型做检索，找到"错误但排名靠前"的图像。
        比文本相似度方法更精准，但需要模型已初步训练好。
        """
        # 1. 编码所有文本和图像
        text_embs, image_embs = [], []
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]
            text_embs.append(self._encode_texts([s["text"] for s in batch]))
            image_embs.append(self._encode_images([s["image_path"] for s in batch]))

        text_embs = np.concatenate(text_embs)
        image_embs = np.concatenate(image_embs)

        # 2. 对每个文本，找检索排名高但不匹配的图像
        sims = text_embs @ image_embs.T  # [N, N]
        triplets = []
        for idx in range(len(samples)):
            ranked = np.argsort(sims[idx])[::-1]
            for rank, neg_idx in enumerate(ranked[1:], 1):  # 跳过自身
                if sims[idx, neg_idx] > sims[idx, idx] - margin:
                    triplets.append({
                        "anchor_text": samples[idx]["text"],
                        "pos_image": samples[idx]["image_path"],
                        "neg_image": samples[neg_idx]["image_path"],
                        "neg_similarity": float(sims[idx, neg_idx]),
                        "rank": rank,
                    })
                else:
                    break  # 后面的太简单，不需要
        return triplets
```

---

### 2.5 Pipeline 编排 (`pipeline.py`)

```python
# pipeline.py
"""
数据集处理主 Pipeline
用法: python pipeline.py --config configs/wukong.yaml
"""
import json, os, argparse, logging
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from sources import WukongSource, COCOCNSource, LAIONCNSource
from filters.image_filter import ImageFilter, ImageFilterConfig
from filters.text_filter import TextFilter, TextFilterConfig
from filters.text_dedup import TextDeduplicator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.image_filter = ImageFilter(ImageFilterConfig(**config.get("image_filter", {})))
        self.text_filter = TextFilter(TextFilterConfig(**config.get("text_filter", {})))
        self.text_dedup = TextDeduplicator(
            jaccard_threshold=config.get("dedup", {}).get("jaccard_threshold", 0.7)
        )
        self.stats = {
            "total_input": 0,
            "image_filtered": 0,
            "text_filtered": 0,
            "text_deduped": 0,
            "output": 0,
        }

    def run(self, output_path: str):
        sources = self._create_sources()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as fout:
            for source in sources:
                logger.info(f"Processing source: {source.__class__.__name__}")
                pbar = tqdm(source.iter_samples(), desc=source.__class__.__name__,
                            total=min(source.estimate_total(), 100_000_000))

                for sample in pbar:
                    self.stats["total_input"] += 1

                    # Step 1: 图像质量过滤
                    img_ok, img_reason = self.image_filter.check(sample["image_path"])
                    if not img_ok:
                        self.stats["image_filtered"] += 1
                        continue

                    # Step 2: 文本质量过滤
                    txt_ok, txt_reason = self.text_filter.check(sample["text"])
                    if not txt_ok:
                        self.stats["text_filtered"] += 1
                        continue

                    # Step 3: 文本去重
                    if self.text_dedup.is_duplicate(sample["id"], sample["text"]):
                        self.stats["text_deduped"] += 1
                        continue

                    # 写入
                    fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    self.stats["output"] += 1

                    # 定期打印统计
                    if self.stats["output"] % 10000 == 0:
                        self._log_stats()

        self._log_stats()
        logger.info(f"Pipeline done. Output: {output_path}")

    def _create_sources(self):
        sources = []
        for src_cfg in self.config["sources"]:
            src_type = src_cfg["type"]
            if src_type == "wukong":
                sources.append(WukongSource(src_cfg["data_dir"]))
            elif src_type == "coco_cn":
                sources.append(COCOCNSource(src_cfg["annotation"], src_cfg["image_dir"]))
            elif src_type == "laion_cn":
                sources.append(LAIONCNSource(src_cfg["parquet_dir"]))
            # ... 可扩展
        return sources

    def _log_stats(self):
        logger.info(
            f"Stats: input={self.stats['total_input']}, "
            f"img_filtered={self.stats['image_filtered']}, "
            f"txt_filtered={self.stats['text_filtered']}, "
            f"txt_deduped={self.stats['text_deduped']}, "
            f"output={self.stats['output']}"
        )


# === 配置文件示例 ===
EXAMPLE_CONFIG = {
    "sources": [
        {"type": "wukong", "data_dir": "/data/wukong"},
        {"type": "coco_cn", "annotation": "/data/coco-cn/annotations.json", "image_dir": "/data/coco-cn/images"},
        {"type": "laion_cn", "parquet_dir": "/data/laion-cn/parquet"},
    ],
    "image_filter": {
        "min_width": 200,
        "min_height": 200,
        "max_aspect_ratio": 5.0,
        "enable_blur_detect": False,   # 大数据集关闭
        "enable_phash_dedup": True,
        "phash_threshold": 8,
    },
    "text_filter": {
        "min_length": 5,
        "max_length": 512,
        "min_zh_ratio": 0.3,
        "max_digit_ratio": 0.5,
    },
    "dedup": {
        "jaccard_threshold": 0.7,
    },
}

if __name__ == "__main__":
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Pipeline 配置文件 (YAML)")
    parser.add_argument("--output", default="output/processed_data.jsonl")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    pipeline = DataPipeline(config)
    pipeline.run(args.output)
```

---

### 2.6 配置文件示例

```yaml
# configs/wukong.yaml
sources:
  - type: wukong
    data_dir: /data/wukong

image_filter:
  min_width: 200
  min_height: 200
  max_aspect_ratio: 5.0
  min_file_size_kb: 5
  enable_blur_detect: false
  enable_phash_dedup: true
  phash_threshold: 8

text_filter:
  min_length: 5
  max_length: 512
  min_zh_ratio: 0.3
  max_digit_ratio: 0.5
  max_repeat_char: 5
  max_repeat_substr: 3

dedup:
  num_perm: 128
  jaccard_threshold: 0.7
```

```yaml
# configs/full_pipeline.yaml  (全量数据)
sources:
  - type: wukong
    data_dir: /data/wukong
  - type: coco_cn
    annotation: /data/coco-cn/annotations.json
    image_dir: /data/coco-cn/images
  - type: laion_cn
    parquet_dir: /data/laion-cn/parquet

image_filter:
  min_width: 256          # 全量训练时提高分辨率门槛
  min_height: 256
  max_aspect_ratio: 4.0
  enable_blur_detect: false
  enable_phash_dedup: true
  phash_threshold: 10     # 更严格的去重

text_filter:
  min_length: 8           # 更高的质量要求
  max_length: 256
  min_zh_ratio: 0.5
  max_digit_ratio: 0.3

dedup:
  num_perm: 256
  jaccard_threshold: 0.6  # 更严格的文本去重
```

---

## 3. 目录结构

```
mini-embed-vision/
├── data_pipeline/                # 数据处理 pipeline
│   ├── __init__.py
│   ├── pipeline.py               # 主编排脚本
│   ├── configs/                  # YAML 配置
│   │   ├── wukong.yaml
│   │   ├── coco_cn.yaml
│   │   └── full_pipeline.yaml
│   ├── sources/                  # 数据源接入
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── wukong.py
│   │   ├── coco_cn.py
│   │   ├── laion_cn.py
│   │   └── synthetic.py
│   ├── filters/                  # 质量过滤
│   │   ├── __init__.py
│   │   ├── image_filter.py
│   │   ├── text_filter.py
│   │   └── text_dedup.py
│   ├── transforms/               # 数据转换
│   │   ├── __init__.py
│   │   └── converter.py
│   └── mining/                   # 难负例挖掘
│       ├── __init__.py
│       └── hard_negative.py
├── data.py                       # v1.0 数据加载（保留兼容）
└── ...
```

---

## 4. 执行流程

```bash
# Step 1: 下载原始数据（手动或脚本）
# - Wukong: https://wukong-dataset.github.io/wukong-dataset/
# - COCO-CN: https://github.com/li-xirong/coco-cn
# - LAION-CN: 用 img2dataset 下载 LAION-5B 后过滤中文

# Step 2: 运行 pipeline
python -m data_pipeline.pipeline --config configs/full_pipeline.yaml --output output/processed.jsonl

# Step 3: 转换为训练格式
python -m data_pipeline.transforms.converter \
    --input output/processed.jsonl \
    --output output/train_data/ \
    --task contrastive

# Step 4: (可选) 难负例挖掘 — 在模型初步训练后运行
python -m data_pipeline.mining.hard_negative \
    --data output/processed.jsonl \
    --checkpoint checkpoints/model_epoch10.pth \
    --output output/hard_negatives.jsonl
```

---

## 5. 依赖

```
# data_pipeline/requirements.txt
Pillow>=10.0
imagehash>=4.3
opencv-python>=4.8
datasketch>=1.6
pyarrow>=14.0
pyyaml>=6.0
tqdm>=4.66
```

---

## 6. 后续优化方向

1. **分布式处理**: 大数据集（Wukong 100M）单机处理太慢，可用 Ray 或 Spark 并行化
2. **流式处理**: 不全量加载到内存，逐条处理写磁盘，支持 TB 级数据
3. **图像 pHash 改进**: 全量数据 pHash 去重内存占用大，改用 Faiss 索引近似去重
4. **质量打分模型**: 训练一个轻量质量评估模型替代规则过滤，覆盖更多噪声类型
5. **数据血缘追踪**: 记录每条样本的来源、过滤结果，方便回溯问题数据
