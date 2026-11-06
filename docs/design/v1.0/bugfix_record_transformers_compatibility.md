# Bugfix Record: Transformers 4.46.3 模型加载兼容性问题

- **日期**: 2025-05-15
- **项目**: mini-embed-vision
- **环境**: macOS (Apple Silicon), Python 3.12, transformers 4.46.3, torch 2.8.0
- **影响文件**: `mini-embed-vision-v1.0/model.py`

---

## 问题描述

使用 `AutoModel.from_pretrained("./mini-embed-vision-v1.0", trust_remote_code=True)` 加载导出模型时，连续遇到三个错误：

### 错误 1: meta tensor copy 失败

```
NotImplementedError: Cannot copy out of meta tensor; no data!
```

**出错位置**: `model.py` 第 97 行 `self.text_encoder.embeddings.position_ids.copy_(correct_pos_ids)`

### 错误 2: meta device 上下文冲突

```
RuntimeError: You are using `from_pretrained` with a meta device context manager.
This is an anti-pattern as `from_pretrained` wants to load existing weights.
```

**出错位置**: `AutoModel.from_pretrained(resolved_text_path)` 内部

### 错误 3: all_tied_weights_keys 属性缺失

```
AttributeError: 'MiniEmbedVisionModel' object has no attribute 'all_tied_weights_keys'.
```

**出错位置**: `modeling_utils.py` → `mark_tied_weights_as_initialized()`

---

## 根因分析

### 新版 transformers 的两阶段加载机制

旧版 transformers (≤4.40) 的 `from_pretrained` 流程：

```
from_pretrained()
  → 在 CPU 上直接创建模型并加载权重（一步完成）
```

新版 transformers (4.46.3) 改为两阶段加载，以节省内存：

```
from_pretrained()
  → 阶段1: 在 meta device 上创建空模型（只有结构，没有数据）
     → 调用 ModelClass(config) → __init__()
  → 阶段2: 加载权重文件，将数据写入模型参数
  → _finalize_model_loading()
     → 检查权重绑定等收尾工作
```

### 三个错误的因果链

```
阶段1: meta device 上调用 __init__()
  │
  ├─ __init__ 内部调用 AutoModel.from_pretrained("BAAI/bge-base-zh-v1.5")
  │   → 检测到外层 meta device 上下文 → 报错2
  │
  ├─ position_ids.copy_(correct_pos_ids)
  │   → meta tensor 没有实际数据，无法 copy → 报错1
  │
  └─ 缺少 post_init() 调用
      → all_tied_weights_keys 属性未注册 → 报错3（阶段2触发）
```

---

## 修复方案

修改文件 `mini-embed-vision-v1.0/model.py` 的 `__init__` 方法：

### 修复 1: 强制子模型在 CPU 加载

```python
# 修复前
self.text_encoder = AutoModel.from_pretrained(resolved_text_path)
self.vision_encoder = CLIPVisionModel.from_pretrained(resolved_vision_path)

# 修复后
with torch.device("cpu"):
    self.text_encoder = AutoModel.from_pretrained(resolved_text_path)

with torch.device("cpu"):
    self.vision_encoder = CLIPVisionModel.from_pretrained(resolved_vision_path)
```

`with torch.device("cpu")` 将子模型的 `from_pretrained` 从外层 meta device 上下文中隔离出来，强制在 CPU 上完成权重加载。

### 修复 2: 移除 position_ids hack

```python
# 删除以下代码（不再需要）
if hasattr(self.text_encoder, 'embeddings') and hasattr(self.text_encoder.embeddings, 'position_ids'):
    max_pos = self.text_encoder.config.max_position_embeddings
    correct_pos_ids = torch.arange(max_pos, dtype=torch.long).unsqueeze(0)
    self.text_encoder.embeddings.position_ids.copy_(correct_pos_ids)
```

`from_pretrained` 会正确加载 BGE 的 position_ids，无需手动修复。这段 hack 在 meta device 场景下反而引发崩溃。

### 修复 3: 保留 post_init()

```python
# __init__ 末尾必须调用
self.post_init()
```

`post_init()` 是 `PreTrainedModel` 的初始化收尾方法，负责注册：
- `all_tied_weights_keys` — 权重绑定信息
- `_no_split_modules` — 模型分片信息
- 其他框架内部状态

新版 transformers 在 `_finalize_model_loading` 阶段会访问这些属性，缺少则报 AttributeError。

---

## 修复后的完整 __init__ 关键代码

```python
def __init__(self, config):
    super().__init__(config)

    # ... 路径解析逻辑 ...

    # 文本编码器（强制 CPU 加载，避免 meta device 上下文冲突）
    with torch.device("cpu"):
        self.text_encoder = AutoModel.from_pretrained(resolved_text_path)
    text_hidden = self.text_encoder.config.hidden_size

    if config.freeze_text:
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        self.text_encoder.eval()

    # 视觉编码器
    with torch.device("cpu"):
        self.vision_encoder = CLIPVisionModel.from_pretrained(resolved_vision_path)
    vis_hidden = self.vision_encoder.config.hidden_size

    # ... LoRA / 投影层 ...

    self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    self.post_init()  # 必须保留
```

---

## 注意事项

1. **每次修改导出目录的 `model.py` 后，需要清除 HuggingFace 缓存**：
   ```bash
   rm -rf ~/.cache/huggingface/modules/transformers_modules/mini_hyphen_embed_hyphen_vision_hyphen_v1_dot_0/
   ```

2. **项目根目录的 `model.py` 也应同步修复**，确保下次 `export_hf_model.py` 导出时使用正确的模板。

3. **Apple Silicon 设备应使用 `mps` 而非 `cuda`**：
   ```python
   device = "mps" if torch.backends.mps.is_available() else "cpu"
   ```

---

## 参考资料

- [Transformers Model Loading](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained)
- [PyTorch Meta Tensors](https://pytorch.org/docs/main/meta.html)
