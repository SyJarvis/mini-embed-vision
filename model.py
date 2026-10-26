"""mini-embed-vision model definitions."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, CLIPVisionModel, PreTrainedModel, PretrainedConfig

class MiniEmbedVisionConfig(PretrainedConfig):

    model_type = "miniembedvision"

    def __init__(
        self,
        embed_dim: int =  768,
        text_model_name: str = "BAAI/bge-base-zh-v1.5",
        vision_model_name: str = "openai/clip-vit-base-patch32",
        vision_backend: str = "transformers",
        timm_model_name: str = "eva02_base_patch14_224",
        projection_dropout: float = 0.0,
        text_pooling: str = "cls",
        text_config: dict = None,
        enable_reranker: bool = False,
        reranker_hidden_dim: int = 256,
        reranker_dropout: float = 0.1,
        reranker_loss_weight: float = 0.2,
        freeze_text: bool = True,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.text_model_name = text_model_name
        self.vision_model_name = vision_model_name
        self.vision_backend = vision_backend
        self.timm_model_name = timm_model_name
        self.projection_dropout = projection_dropout
        self.text_pooling = text_pooling
        self.text_config = text_config or {}
        self.enable_reranker = enable_reranker
        self.reranker_hidden_dim = reranker_hidden_dim
        self.reranker_dropout = reranker_dropout
        self.reranker_loss_weight = reranker_loss_weight
        self.freeze_text = freeze_text
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, CLIPVisionModel, PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from typing import Optional, Tuple, Union
import os

try:
    from peft import LoraConfig, get_peft_model, TaskType
    _has_peft = True
except ImportError:
    _has_peft = False

class AllGatherWithGrad(torch.autograd.Function):
    """all_gather 且对本地分片保留梯度（open_clip 风格）。

    标准 all_gather 不可微（其他 rank 的张量不带梯度）。backward 时每个 rank
    只取全局梯度中属于自己分片的部分直接返回：各 rank 的 loss 相互独立，
    本 rank loss 对本地嵌入的梯度恰好是 (dL/dg)[本 rank 分片]，无需跨 rank 聚合。
    """

    @staticmethod
    def forward(ctx, tensor):
        world_size = torch.distributed.get_world_size()
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        return grad_output.chunk(world_size, dim=0)[rank].contiguous()


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, hidden):
        return self.norm(self.fc2(self.dropout(self.activation(self.fc1(hidden)))))


class CrossModalReranker(nn.Module):
    def __init__(self, embed_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        input_dim = embed_dim * 4
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, text_emb, image_emb):
        if text_emb.dim() == 2:
            text_emb = text_emb.unsqueeze(1)
        if image_emb.dim() == 2:
            image_emb = image_emb.unsqueeze(0)
        text_emb = text_emb.expand(-1, image_emb.size(1), -1)
        image_emb = image_emb.expand(text_emb.size(0), -1, -1)
        features = torch.cat(
            [text_emb, image_emb, text_emb * image_emb, (text_emb - image_emb).abs()],
            dim=-1,
        )
        return self.fc2(self.dropout(self.activation(self.fc1(features)))).squeeze(-1)


def _load_timm_vision_model(config, resolved_vision_path):
    try:
        import timm
    except ImportError as exc:
        raise ImportError("timm is required for the EVA02 backend. Install it with: pip install timm") from exc

    model_name = config.timm_model_name or "eva02_base_patch14_224"
    model = timm.create_model(model_name, pretrained=False, num_classes=0)
    if hasattr(model, "rope") and model.rope is not None:
        model.rope.pos_embed = nn.Parameter(model.rope.pos_embed, requires_grad=False)
    if not resolved_vision_path:
        return model
    checkpoint = os.path.join(resolved_vision_path, "model.safetensors")
    if os.path.isfile(checkpoint):
        from safetensors.torch import load_file

        state_dict = load_file(checkpoint)
    else:
        torch_checkpoint = os.path.join(resolved_vision_path, "pytorch_model.bin")
        if not os.path.isfile(torch_checkpoint):
            raise FileNotFoundError(
                f"No EVA02 checkpoint found. Expected model.safetensors or "
                f"pytorch_model.bin under {resolved_vision_path}"
            )
        state_dict = torch.load(torch_checkpoint, map_location="cpu", weights_only=True)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if set(missing_keys) == {"rope.pos_embed"}:
        missing_keys = []
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"Invalid EVA02 checkpoint: missing={missing_keys}, unexpected={unexpected_keys}"
        )
    if hasattr(model, "rope") and hasattr(model.rope, "reset_parameters"):
        model.rope.reset_parameters()
    return model


def _build_text_encoder(config):
    text_config = getattr(config, "text_config", None)
    if text_config:
        return AutoModel.from_config(AutoConfig.for_model(**text_config))
    return AutoModel.from_pretrained(config.text_model_name)

class MiniEmbedVisionModel(PreTrainedModel):
    config_class = MiniEmbedVisionConfig
    base_model_prefix = "miniembed_vision"

    def __init__(self, config):
        super().__init__(config)
        # 解析 vision_model_name
        vision_model_path = config.vision_model_name
        if not vision_model_path:
            resolved_vision_path = ""
        elif vision_model_path.startswith("./") and hasattr(config, "_name_or_path") and config._name_or_path:
            resolved_vision_path = os.path.abspath(os.path.join(config._name_or_path, vision_model_path))
        elif os.path.isdir(vision_model_path):
            resolved_vision_path = os.path.abspath(vision_model_path)
        elif vision_model_path.startswith("./") or os.path.isabs(vision_model_path):
            raise FileNotFoundError(f"Vision encoder path not found: {vision_model_path}")
        else:
            resolved_vision_path = vision_model_path
        
        # 文本编码器
        self.text_encoder = _build_text_encoder(config)
        text_hidden = self.text_encoder.config.hidden_size
        # 若冻结文本编码器 freeze_text
        if config.freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            self.text_encoder.eval()
        
        # 视觉编码器
        if config.vision_backend == "timm":
            self.vision_encoder = _load_timm_vision_model(config, resolved_vision_path)
            vis_hidden = self.vision_encoder.num_features
        elif config.vision_backend == "transformers":
            self.vision_encoder = CLIPVisionModel.from_pretrained(resolved_vision_path)
            vis_hidden = self.vision_encoder.config.hidden_size
        else:
            raise ValueError(f"Unsupported vision_backend: {config.vision_backend}")

        # lora config
        if config.use_lora:
            if not _has_peft:
                raise ImportError("peft is required for LoRA. Please install: pip install peft")
            lora_targets = (
                ["q_proj", "k_proj", "v_proj", "proj", "fc1_g", "fc1_x", "fc2"]
                if config.vision_backend == "timm"
                else ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "proj"]
            )
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=lora_targets,
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type=TaskType.OTHER,
            )
            self.vision_encoder = get_peft_model(self.vision_encoder, lora_config)

        self.text_proj = (
            ProjectionHead(text_hidden, config.embed_dim, config.projection_dropout)
            if config.vision_backend == "timm"
            else nn.Linear(text_hidden, config.embed_dim)
            if text_hidden != config.embed_dim
            else nn.Identity()
        )
        self.vision_proj = (
            ProjectionHead(vis_hidden, config.embed_dim, config.projection_dropout)
            if config.vision_backend == "timm"
            else nn.Linear(vis_hidden, config.embed_dim)
            if vis_hidden != config.embed_dim
            else nn.Identity()
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
        if config.enable_reranker:
            self.reranker = CrossModalReranker(
                config.embed_dim,
                config.reranker_hidden_dim,
                config.reranker_dropout,
            )
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, ProjectionHead):
            nn.init.xavier_uniform_(module.fc1.weight)
            nn.init.zeros_(module.fc1.bias)
            nn.init.xavier_uniform_(module.fc2.weight)
            nn.init.zeros_(module.fc2.bias)
            nn.init.ones_(module.norm.weight)
            nn.init.zeros_(module.norm.bias)
        elif isinstance(module, MiniEmbedVisionModel):
            nn.init.constant_(self.logit_scale, torch.log(torch.tensor(1 / 0.07)))
        elif isinstance(module, CrossModalReranker):
            nn.init.xavier_uniform_(module.fc1.weight)
            nn.init.zeros_(module.fc1.bias)
            nn.init.xavier_uniform_(module.fc2.weight)
            nn.init.zeros_(module.fc2.bias)
    
    def train(self, mode: bool = True):
        """保持冻结的 text_encoder 永远处于 eval 模式（dropout 不生效）。"""
        super().train(mode)
        if getattr(self.config, "freeze_text", False):
            self.text_encoder.eval()
        return self

    def _pool_text(self, last_hidden, attention_mask):
        pooling = getattr(self.config, "text_pooling", "cls")
        if pooling == "cls":
            return last_hidden[:, 0]
        if pooling == "mean":
            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_emb = torch.sum(last_hidden * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            return sum_emb / sum_mask
        raise ValueError(f"Unsupported text_pooling: {pooling}")

    def encode_text(self, input_ids, attention_mask, **kwargs):
        with torch.no_grad() if not any(p.requires_grad for p in self.text_encoder.parameters()) else torch.enable_grad():
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            pooled = self._pool_text(outputs.last_hidden_state, attention_mask)
        emb = self.text_proj(pooled)
        return F.normalize(emb, p=2, dim=-1)

    def encode_image(self, pixel_values, **kwargs):
        if self.config.vision_backend == "timm":
            features = self.vision_encoder(pixel_values)
        else:
            outputs = self.vision_encoder(pixel_values=pixel_values, return_dict=True)
            features = outputs.last_hidden_state[:, 0]
        emb = self.vision_proj(features)
        return F.normalize(emb, p=2, dim=-1)

    def _encode_text_for_training(self, input_ids, attention_mask):
        text_grad_enabled = any(p.requires_grad for p in self.text_encoder.parameters())
        text_proj_grad_enabled = any(p.requires_grad for p in self.text_proj.parameters())
        with torch.set_grad_enabled(text_grad_enabled or text_proj_grad_enabled):
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            pooled = self._pool_text(outputs.last_hidden_state, attention_mask)
            emb = self.text_proj(pooled)
        return F.normalize(emb, p=2, dim=-1)

    def score_rerank(self, text_emb, image_emb):
        if not getattr(self.config, "enable_reranker", False):
            raise RuntimeError("Reranker is disabled. Set enable_reranker=True.")
        return self.reranker(text_emb, image_emb)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        gather_for_ddp: bool = True,
        labels: Optional[torch.Tensor] = None,  # reserved for supervised contrastive
        return_rerank_scores: bool = False,
        return_rerank_loss: bool = False,
        **kwargs
    ) -> Union[ModelOutput, Tuple[torch.Tensor, ...]]:
        """
        模拟 Nomic 风格：既支持文本也支持图像
        Supports:
          - Inference: only text OR only image → returns embeddings.
          - Training: both text AND image → returns logits and (optionally) loss.
        """
        text_emb = None
        image_emb = None

        # Encode modalities if provided
        if input_ids is not None and attention_mask is not None:
            text_emb = (
                self._encode_text_for_training(input_ids, attention_mask)
                if self.training
                else self.encode_text(input_ids, attention_mask)
            )
        if pixel_values is not None:
            image_emb = self.encode_image(pixel_values)

        # Inference mode: single modality
        if text_emb is not None and image_emb is None:
            return ModelOutput(last_hidden_state=text_emb, text_embeds=text_emb)
        if image_emb is not None and text_emb is None:
            return ModelOutput(last_hidden_state=image_emb, image_embeds=image_emb)

        # Training mode: both modalities present
        if text_emb is None or image_emb is None:
            raise ValueError("For training, both text and image inputs are required.")

        # Gather across GPUs for large-batch negatives（可微版本）
        if gather_for_ddp and torch.distributed.is_initialized():
            text_emb_all = AllGatherWithGrad.apply(text_emb)
            image_emb_all = AllGatherWithGrad.apply(image_emb)
        else:
            text_emb_all = text_emb
            image_emb_all = image_emb

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        logits_per_text = logit_scale * text_emb @ image_emb_all.t()   # [B, global_B]
        logits_per_image = logit_scale * image_emb @ text_emb_all.t()  # [B, global_B]

        loss = None
        rerank_scores = None
        rerank_loss = None
        if return_loss:
            # 全局负例：本地 text 与所有 rank 的 image 两两配对，
            # 正样本位于全局对角线上，偏移 = rank * local_batch_size
            local_batch_size = text_emb.size(0)
            if gather_for_ddp and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                labels = torch.arange(
                    rank * local_batch_size,
                    rank * local_batch_size + local_batch_size,
                    device=text_emb.device,
                )
            else:
                labels = torch.arange(local_batch_size, device=text_emb.device)
            # t2i: 本地 text 为锚；i2t: 本地 image 为锚（各 rank 合起来覆盖全局）
            loss_t2i = F.cross_entropy(logits_per_text, labels)
            loss_i2t = F.cross_entropy(logits_per_image, labels)
            loss = (loss_t2i + loss_i2t) / 2

        if getattr(self.config, "enable_reranker", False):
            if return_rerank_scores:
                rerank_scores = self.score_rerank(text_emb, image_emb)
            if return_rerank_loss:
                if gather_for_ddp and torch.distributed.is_initialized():
                    rank = torch.distributed.get_rank()
                    local_batch_size = text_emb.size(0)
                    rerank_scores = self.score_rerank(text_emb, image_emb_all)
                    rerank_labels = torch.zeros_like(rerank_scores)
                    positive_offset = rank * local_batch_size
                    rerank_labels[
                        torch.arange(local_batch_size, device=text_emb.device),
                        positive_offset + torch.arange(local_batch_size, device=text_emb.device),
                    ] = 1.0
                else:
                    rerank_scores = self.score_rerank(text_emb, image_emb)
                    rerank_labels = torch.eye(text_emb.size(0), device=text_emb.device, dtype=torch.float32)
                rerank_loss = F.binary_cross_entropy_with_logits(rerank_scores, rerank_labels)
                if loss is not None:
                    loss = loss + self.config.reranker_loss_weight * rerank_loss
                else:
                    loss = rerank_loss

        return ModelOutput(
            loss=loss,
            rerank_loss=rerank_loss,
            logits_per_text=logits_per_text,
            logits_per_image=logits_per_image,
            rerank_scores=rerank_scores,
            text_embeds=text_emb,
            image_embeds=image_emb,
        )

MiniEmbedVisionConfig.register_for_auto_class()
MiniEmbedVisionModel.register_for_auto_class("AutoModel")
