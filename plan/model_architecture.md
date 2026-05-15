# mini-embed-vision v1.5 模型架构草案

## 1. 架构总览

```
┌─────────────────────────────────────────────────────────┐
│                  MiniEmbedVision-v1.5                     │
│                                                          │
│  ┌──────────────┐              ┌──────────────────┐      │
│  │  文本输入     │              │  图像输入          │      │
│  │  (指令+文本) │              │  (224×224 RGB)   │      │
│  └──────┬───────┘              └────────┬─────────┘      │
│         │                               │                │
│  ┌──────▼───────┐              ┌────────▼─────────┐      │
│  │ Qwen3-Token. │              │  timm Transforms │      │
│  │ (padding=left│              │  Resize+Norm     │      │
│  │  max_len=32K)│              │  ToTensor        │      │
│  └──────┬───────┘              └────────┬─────────┘      │
│         │                               │                │
│  ┌──────▼──────────────────┐   ┌────────▼─────────┐      │
│  │ Qwen3-Embedding-0.6B    │   │  EVA02-Base-P14  │      │
│  │ (28 layers, frozen)      │   │  (12 layers)     │      │
│  │ hidden_size=1024         │   │  embed_dim=768   │      │
│  │ pooling: last_token      │   │  pooling: CLS    │      │
│  └──────┬──────────────────┘   └────────┬─────────┘      │
│         │ [B, 1024]                     │ [B, 768]       │
│  ┌──────▼──────────┐           ┌────────▼─────────┐      │
│  │ Text Projection │           │ Vision Projection│      │
│  │ 1024 → 1024     │           │ 768 → 1024       │      │
│  │ (MLP: FC+GELU   │           │ (MLP: FC+GELU    │      │
│  │  +FC)           │           │  +FC)             │      │
│  └──────┬──────────┘           └────────┬─────────┘      │
│         │ [B, 1024]                     │ [B, 1024]       │
│  ┌──────▼──────────┐           ┌────────▼─────────┐      │
│  │ L2 Normalize    │           │ L2 Normalize     │      │
│  └──────┬──────────┘           └────────┬─────────┘      │
│         │                               │                │
│         └───────────┬───────────────────┘                │
│                     │                                    │
│              ┌──────▼──────┐                              │
│              │ 对比学习 Loss │                              │
│              │ + 难负例 Loss │                              │
│              │ + Rerank Loss│                              │
│              └─────────────┘                              │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 子模型参数

### 2.1 文本编码器: Qwen3-Embedding-0.6B

| 属性 | 值 |
|------|-----|
| 模型ID | `Qwen/Qwen3-Embedding-0.6B` |
| 参数量 | 0.6B |
| 层数 | 28 |
| hidden_size | 1024 |
| 上下文长度 | 32K |
| 嵌入维度 | 1024 (支持 32-1024 裁剪，MRL) |
| Pooling | last_token_pool |
| 指令格式 | `Instruct: {task}\nQuery:{query}` |
| padding_side | left |
| 推荐 | flash_attention_2 |

```python
import torch
from transformers import AutoModel, AutoTokenizer

def load_text_encoder(model_name="Qwen/Qwen3-Embedding-0.6B", freeze=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = AutoModel.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",  # 推荐
        torch_dtype=torch.float16,
    )
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
    return model, tokenizer

def last_token_pool(last_hidden_states, attention_mask):
    """Qwen3-Embedding 的 pooling 方法"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def encode_text(model, tokenizer, texts, instructions=None, max_length=8192):
    """
    编码文本，支持指令感知
    instructions: 可选，每条文本对应的任务指令
    """
    if instructions:
        input_texts = [
            f'Instruct: {inst}\nQuery:{text}'
            for inst, text in zip(instructions, texts)
        ]
    else:
        input_texts = texts

    batch = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**batch)

    embeddings = last_token_pool(outputs.last_hidden_state, batch['attention_mask'])
    return embeddings  # [B, 1024]
```

### 2.2 视觉编码器: EVA02-Base-P14

| 属性 | 值 |
|------|-----|
| 模型ID | `timm/eva02_base_patch14_224.mim_in22k` |
| 参数量 | ~86M |
| 层数 | 12 |
| embed_dim | 768 |
| num_heads | 12 |
| head_dim | 64 |
| patch_size | 14×14 |
| MLP 类型 | SwiGLU (mlp_ratio=2.667) |
| 位置编码 | 绝对位置编码 + RoPE |
| 输入分辨率 | 224×224 (可扩展到 448) |
| Patch tokens | 16×16 = 256 (+ 1 CLS token) |
| class_token | True |
| **默认 Pooling** | **avg (全局平均池化 + fc_norm)** |
| 备选 Pooling | CLS token (global_pool='token') |
| 预训练 | MIM on ImageNet-22K |

> **注意**: EVA02 默认使用 avg pooling（对所有 patch token 取平均 + fc_norm LayerNorm），
> 而非 CLS token。实践表明 avg pooling 在对比学习任务中通常优于 CLS token。
> 如需切换为 CLS token pooling，创建模型时传入 `global_pool='token'`。

```python
import timm

def load_vision_encoder(model_name="timm/eva02_base_patch14_224.mim_in22k"):
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=0,          # 去掉分类头，forward() 直接输出 [B, 768]
        # global_pool='token',  # 如需 CLS token pooling，取消注释
    )
    return model

def encode_image(model, pixel_values):
    """
    pixel_values: [B, 3, 224, 224]
    输出: [B, 768] (avg pooled + fc_norm)
    """
    # 方式1: 直接调用 model()，内部完成 pooling + norm
    features = model(pixel_values)          # [B, 768]
    return features

    # 方式2: 手动分步（需要访问中间特征时使用）
    # raw = model.forward_features(pixel_values)  # [B, 257, 768] (1 CLS + 256 patches)
    # features = model.forward_head(raw, pre_logits=True)  # [B, 768]
```

---

## 3. 投影层设计

v1.0 使用简单的 `nn.Linear`，v1.5 改为 MLP 投影头（参考 CLIP/SigLIP 的做法）：

```python
import torch.nn as nn

class ProjectionHead(nn.Module):
    """双塔投影头：将文本/视觉特征映射到统一嵌入空间"""
    def __init__(self, input_dim, output_dim=1024, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.norm(x)
        return x
```

**参数量对比**：

| 组件 | 输入维度 | 输出维度 | 参数量 |
|------|---------|---------|--------|
| Text Projection | 1024 → 1024 | 1024 | ~2.1M |
| Vision Projection | 768 → 1024 | 1024 | ~1.9M |
| **总可训练** | | | **~90M** (含 EVA02) |

---

## 4. 完整模型定义

```python
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from typing import Optional, Union
import timm

# ═══════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════

class MiniEmbedVisionV15Config(PretrainedConfig):
    model_type = "miniembedvision_v15"

    def __init__(
        self,
        # 嵌入维度
        embed_dim: int = 1024,

        # 文本编码器
        text_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        freeze_text: bool = True,
        text_max_length: int = 8192,
        text_hidden_size: int = 1024,

        # 视觉编码器
        vision_model_name: str = "timm/eva02_base_patch14_224.mim_in22k",
        vision_hidden_size: int = 768,
        image_size: int = 224,

        # 训练策略
        freeze_vision_layers: int = 0,     # 冻结 EVA02 前N层, 0=不冻结
        use_lora_vision: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,

        # 投影层
        projection_dropout: float = 0.1,

        # 重排序
        enable_reranker: bool = False,
        reranker_dim: int = 256,           # 融合层隐藏维度

        # MRL (Matryoshka Representation Learning)
        enable_mrl: bool = False,
        mrl_dims: list = None,             # e.g. [256, 512, 768, 1024]

        # 温度
        init_temperature: float = 0.07,

        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.text_model_name = text_model_name
        self.freeze_text = freeze_text
        self.text_max_length = text_max_length
        self.text_hidden_size = text_hidden_size
        self.vision_model_name = vision_model_name
        self.vision_hidden_size = vision_hidden_size
        self.image_size = image_size
        self.freeze_vision_layers = freeze_vision_layers
        self.use_lora_vision = use_lora_vision
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.projection_dropout = projection_dropout
        self.enable_reranker = enable_reranker
        self.reranker_dim = reranker_dim
        self.enable_mrl = enable_mrl
        self.mrl_dims = mrl_dims or [256, 512, 768, 1024]
        self.init_temperature = init_temperature


# ═══════════════════════════════════════════════════
#  DDP Gather
# ═══════════════════════════════════════════════════

def _concat_all_gather(tensor):
    if not torch.distributed.is_initialized():
        return tensor
    tensors_gather = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)
    return torch.cat(tensors_gather, dim=0)


# ═══════════════════════════════════════════════════
#  Projection Head
# ═══════════════════════════════════════════════════

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.norm(x)
        return x


# ═══════════════════════════════════════════════════
#  Reranker Head (轻量跨模态融合)
# ═══════════════════════════════════════════════════

class CrossModalReranker(nn.Module):
    """
    轻量级重排序头：
    将 text_emb 和 image_emb 拼接后通过 MLP 输出相关性分数。
    不使用完整的 cross-encoder，参数增量小但能提供 rerank 能力。
    """
    def __init__(self, embed_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim * 2, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, text_emb, image_emb):
        """
        text_emb:  [B, D] (未归一化的投影输出)
        image_emb: [B, D]
        返回: [B] 相关性分数
        """
        combined = torch.cat([text_emb, image_emb], dim=-1)  # [B, 2D]
        h = self.gelu(self.fc1(combined))
        score = self.fc2(h).squeeze(-1)  # [B]
        return score


# ═══════════════════════════════════════════════════
#  Pooling
# ═══════════════════════════════════════════════════

def last_token_pool(last_hidden_states, attention_mask):
    """Qwen3-Embedding 使用的 last token pooling"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths
        ]


# ═══════════════════════════════════════════════════
#  Main Model
# ═══════════════════════════════════════════════════

class MiniEmbedVisionV15Model(PreTrainedModel):
    config_class = MiniEmbedVisionV15Config
    base_model_prefix = "miniembed_vision_v15"

    def __init__(self, config: MiniEmbedVisionV15Config):
        super().__init__(config)

        # ── 文本编码器 (frozen) ──
        self.text_encoder = AutoModel.from_pretrained(config.text_model_name)
        if config.freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            self.text_encoder.eval()

        # ── 视觉编码器 ──
        self.vision_encoder = timm.create_model(
            config.vision_model_name,
            pretrained=True,
            num_classes=0,
            img_size=config.image_size,
        )

        # 冻结前N层（低层特征更通用）
        if config.freeze_vision_layers > 0:
            # EVA02 blocks 存储在 self.vision_encoder.blocks
            for i, block in enumerate(self.vision_encoder.blocks):
                if i < config.freeze_vision_layers:
                    for p in block.parameters():
                        p.requires_grad = False

        # LoRA for vision (可选)
        if config.use_lora_vision:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=["qkv", "proj", "fc1", "fc2"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.OTHER,
            )
            self.vision_encoder = get_peft_model(self.vision_encoder, lora_config)

        # ── 投影层 ──
        self.text_proj = ProjectionHead(
            config.text_hidden_size, config.embed_dim, config.projection_dropout
        )
        self.vision_proj = ProjectionHead(
            config.vision_hidden_size, config.embed_dim, config.projection_dropout
        )

        # ── 温度参数 ──
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1 / config.init_temperature))
        )

        # ── 重排序头 (可选) ──
        if config.enable_reranker:
            self.reranker = CrossModalReranker(config.embed_dim, config.reranker_dim)

    def encode_text(self, input_ids, attention_mask, **kwargs):
        """
        文本编码 → 投影 → L2归一化
        输出: [B, embed_dim]
        """
        with torch.no_grad() if not any(
            p.requires_grad for p in self.text_encoder.parameters()
        ) else torch.enable_grad():
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            pooled = last_token_pool(outputs.last_hidden_state, attention_mask)

        projected = self.text_proj(pooled)
        return F.normalize(projected, p=2, dim=-1)

    def encode_image(self, pixel_values, **kwargs):
        """
        图像编码 → 投影 → L2归一化
        pixel_values: [B, 3, H, W]
        输出: [B, embed_dim]

        EVA02 默认使用 avg pooling (全局平均池化 + fc_norm)，
        model(pixel_values) 在 num_classes=0 时直接输出 [B, 768]
        """
        features = self.vision_encoder(pixel_values)  # [B, 768]

        projected = self.vision_proj(features)
        return F.normalize(projected, p=2, dim=-1)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        gather_for_ddp: bool = True,
        return_rerank_score: bool = False,
        **kwargs
    ) -> Union[ModelOutput, tuple]:

        text_emb = None
        image_emb = None

        # ── 编码 ──
        if input_ids is not None and attention_mask is not None:
            text_emb = self.encode_text(input_ids, attention_mask)
        if pixel_values is not None:
            image_emb = self.encode_image(pixel_values)

        # ── 单模态推理 ──
        if text_emb is not None and image_emb is None:
            return ModelOutput(last_hidden_state=text_emb, text_embeds=text_emb)
        if image_emb is not None and text_emb is None:
            return ModelOutput(last_hidden_state=image_emb, image_embeds=image_emb)

        # ── 训练模式 ──
        if text_emb is None or image_emb is None:
            raise ValueError("Training requires both text and image inputs.")

        # Gather for DDP
        if gather_for_ddp and torch.distributed.is_initialized():
            text_emb_all = _concat_all_gather(text_emb)
            image_emb_all = _concat_all_gather(image_emb)
        else:
            text_emb_all = text_emb
            image_emb_all = image_emb

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        logits_per_text = logit_scale * text_emb @ image_emb_all.t()
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            local_bs = text_emb.size(0)
            labels = torch.arange(local_bs, device=text_emb.device)

            if torch.distributed.is_initialized():
                logits_local = logit_scale * text_emb @ image_emb.t()
            else:
                logits_local = logits_per_text

            loss_i = F.cross_entropy(logits_local, labels)
            loss_t = F.cross_entropy(logits_local.t(), labels)
            loss = (loss_i + loss_t) / 2

            # ── 重排序辅助损失 (可选) ──
            if return_rerank_score and self.config.enable_reranker:
                # 正对的 rerank score 应该高，负对应该低
                rerank_scores = self.reranker(
                    self.text_proj(last_token_pool(
                        self.text_encoder(input_ids=input_ids, attention_mask=attention_mask,
                                          return_dict=True).last_hidden_state,
                        attention_mask
                    )),
                    self.vision_proj(self.vision_encoder(pixel_values)),
                )
                rerank_labels = torch.arange(local_bs, device=text_emb.device)
                rerank_loss = F.cross_entropy(rerank_scores.unsqueeze(0).expand(local_bs, -1), rerank_labels)
                loss = loss + 0.1 * rerank_loss  # 加权

        return ModelOutput(
            loss=loss,
            logits_per_text=logits_per_text,
            logits_per_image=logits_per_image,
            text_embeds=text_emb,
            image_embeds=image_emb,
        )


# 注册 AutoModel
MiniEmbedVisionV15Config.register_for_auto_class()
MiniEmbedVisionV15Model.register_for_auto_class("AutoModel")
```

---

## 5. 模型参数统计

```
┌────────────────────────┬───────────┬────────────┐
│ 组件                    │ 参数量     │ 是否可训练  │
├────────────────────────┼───────────┼────────────┤
│ Qwen3-Embedding-0.6B   │ ~600M     │ ✗ (冻结)   │
│ EVA02-Base-P14         │ ~86M      │ ✓          │
│ Text Projection (MLP)  │ ~2.1M     │ ✓          │
│ Vision Projection (MLP)│ ~1.9M     │ ✓          │
│ Logit Scale            │ 1         │ ✓          │
│ Reranker Head (可选)   │ ~0.5M     │ ✓          │
├────────────────────────┼───────────┼────────────┤
│ 总参数                  │ ~690M     │            │
│ 可训练参数              │ ~90M      │            │
│ 冻结参数                │ ~600M     │            │
└────────────────────────┴───────────┴────────────┘
```

---

## 6. 训练策略

### 6.1 分阶段训练

```
阶段 1: 大规模图文对齐预训练
├── 数据: Wukong / LAION-CN (百万级)
├── 冻结: 文本编码器完全冻结
├── 训练: EVA02 全部 12 层 + 投影层
├── Loss: 标准对比学习 (InfoNCE)
├── Epochs: 5-10
├── LR: 2e-4 (warmup 500 steps)
└── 目标: 视觉特征初步对齐到文本空间

阶段 2: 高质量数据精调
├── 数据: COCO-CN / Flickr30K-CN / 合成数据 (万级)
├── 冻结: EVA02 前 6 层冻结
├── 训练: EVA02 后 6 层 + 投影层
├── Loss: 对比学习 + 难负例损失
├── Epochs: 20-50
├── LR: 5e-5 (warmup 200 steps)
└── 目标: 提升细粒度检索质量

阶段 3 (可选): 重排序训练
├── 数据: 阶段2数据 + 挖掘的难负例
├── 训练: 仅 Reranker Head + 投影层微调
├── Loss: 对比学习 + 重排序交叉熵
├── Epochs: 10-20
├── LR: 1e-4
└── 目标: 优化重排序能力
```

### 6.2 关键训练参数

```yaml
# configs/train_stage1.yaml  (预训练)
batch_size: 512              # 大 batch 对对比学习很重要
gradient_accumulation: 8     # 等效 512 = 64 * 8
learning_rate: 2e-4
weight_decay: 0.01
warmup_steps: 500
max_epochs: 10
max_text_length: 256         # 预训练阶段不需要太长文本
image_size: 224
fp16: true
ddp: true

# configs/train_stage2.yaml  (精调)
batch_size: 256
gradient_accumulation: 4
learning_rate: 5e-5
weight_decay: 0.01
warmup_steps: 200
max_epochs: 50
max_text_length: 512         # 精调时用更长文本
image_size: 224              # 或 336（如GPU内存允许）
freeze_vision_layers: 6      # 冻结前6层
hard_negatives: true
fp16: true
```

### 6.3 MRL 训练 (Matryoshka Representation Learning)

支持可变维度输出（32-1024），让用户按需裁剪嵌入维度：

```python
def mrl_loss(text_emb, image_emb, dims=[256, 512, 768, 1024]):
    """在多个维度上同时计算对比损失"""
    total_loss = 0.0
    for d in dims:
        t = F.normalize(text_emb[:, :d], p=2, dim=-1)
        i = F.normalize(image_emb[:, :d], p=2, dim=-1)
        logits = t @ i.t()
        labels = torch.arange(len(t), device=t.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
        total_loss += loss
    return total_loss / len(dims)
```

---

## 7. 推理 API 设计

```python
from transformers import AutoModel, AutoTokenizer
import timm
from PIL import Image
from torchvision import transforms

# ── 加载模型 ──
model = AutoModel.from_pretrained(
    "./mini-embed-vision-v1.5",
    trust_remote_code=True,
    torch_dtype=torch.float16,
).cuda()

tokenizer = AutoTokenizer.from_pretrained(
    "./mini-embed-vision-v1.5",
    padding_side='left'
)

# 图像预处理
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])

# ── 文本嵌入 ──
texts = ["一只橘色的猫趴在沙发上", "蓝色的天空"]
instructions = ["检索与描述匹配的图片"] * len(texts)
input_texts = [f'Instruct: {inst}\nQuery:{t}' for inst, t in zip(instructions, texts)]

batch = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt").to(model.device)
with torch.no_grad():
    text_emb = model.encode_text(batch["input_ids"], batch["attention_mask"])
# text_emb.shape: [2, 1024]

# ── 图像嵌入 ──
img = Image.open("cat.jpg").convert("RGB")
pixel_values = image_transform(img).unsqueeze(0).to(model.device)
with torch.no_grad():
    image_emb = model.encode_image(pixel_values)
# image_emb.shape: [1, 1024]

# ── 相似度计算 ──
similarity = (text_emb @ image_emb.T).squeeze()
print(f"相似度: {similarity}")

# ── 维度裁剪 (MRL) ──
text_emb_256 = text_emb[:, :256]  # 裁剪到 256 维，仍可用
text_emb_512 = text_emb[:, :512]  # 裁剪到 512 维
```

---

## 8. HuggingFace 导出格式

```
mini-embed-vision-v1.5/
├── config.json                    # MiniEmbedVisionV15Config
├── model.safetensors              # 模型权重
├── tokenizer.json                 # Qwen3 tokenizer
├── tokenizer_config.json
├── special_tokens_map.json
├── preprocessing_config.json      # 图像预处理参数
├── modeling_miniembedvision_v15.py  # 模型代码 (trust_remote_code)
└── README.md
```

`config.json` 示例：
```json
{
    "model_type": "miniembedvision_v15",
    "auto_map": {
        "AutoModel": "modeling_miniembedvision_v15.MiniEmbedVisionV15Model"
    },
    "embed_dim": 1024,
    "text_model_name": "Qwen/Qwen3-Embedding-0.6B",
    "vision_model_name": "timm/eva02_base_patch14_224.mim_in22k",
    "freeze_text": true,
    "vision_hidden_size": 768,
    "text_hidden_size": 1024,
    "image_size": 224,
    "enable_reranker": true,
    "reranker_dim": 256,
    "enable_mrl": true,
    "mrl_dims": [256, 512, 768, 1024],
    "init_temperature": 0.07
}
```

---

## 9. 与 v1.0 的关键差异

| 特性 | v1.0 | v1.5 |
|------|------|------|
| 文本编码器 | BGE-base-zh (768d, mean pool) | Qwen3-Embedding-0.6B (1024d, last_token pool) |
| 视觉编码器 | CLIP-ViT-B/32 (768d) | EVA02-Base-P14 (768d) |
| 投影层 | nn.Linear | MLP (FC+GELU+FC+LN) |
| 输出维度 | 768 (固定) | 1024 (支持 32-1024 裁剪, MRL) |
| 指令感知 | 不支持 | 支持 (Instruct: ... Query: ...) |
| 重排序 | 不支持 | 轻量 CrossModalReranker |
| 上下文长度 | 128 | 32K (文本), 实际训练 256-512 |
| 文本编码器冻结 | 全冻结 | 全冻结 |
| 视觉编码器冻结 | 全不冻 | 分层冻结 (前N层冻结) |
| 训练数据 | COCO128 (128条) | 大规模中文图文对 (百万级) |
| 难负例 | 无 | 支持挖掘 + 训练 |

---

## 10. 目录结构 (建议)

```
mini-embed-vision-v1.5/
├── configs/                       # 配置文件
│   ├── train_stage1.yaml          # 预训练
│   ├── train_stage2.yaml          # 精调
│   └── train_stage3_rerank.yaml   # 重排序
├── src/
│   ├── __init__.py
│   ├── model.py                   # MiniEmbedVisionV15Model
│   ├── config.py                  # MiniEmbedVisionV15Config
│   ├── projection.py              # ProjectionHead
│   ├── reranker.py                # CrossModalReranker
│   └── pooling.py                 # last_token_pool
├── data_pipeline/                 # 数据处理 (见 data_pipeline.md)
├── train.py                       # 训练脚本
├── data.py                        # DataLoader (v1.5 版)
├── export_hf.py                   # 导出 HuggingFace 格式
├── eval.py                        # 评估脚本
├── examples/                      # 使用示例
└── README.md
```

---

## 11. 后续考虑

1. **更高分辨率**: EVA02 支持动态分辨率，可在精调阶段尝试 336×336
2. **LoRA for Text**: 如果发现文本编码器也需要适配，可用 LoRA 微调 Qwen3 的低层
3. **多尺度特征融合**: 除了 CLS token，还可以用中间层特征 (multi-scale feature) 提升细粒度检索
4. **知识蒸馏**: 用大模型（如 EVA02-Large）做教师模型蒸馏到 Base 版本
5. **量化部署**: 训练后量化 (PTQ) 到 int8/int4，降低部署成本
