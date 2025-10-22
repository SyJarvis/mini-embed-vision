"""v0.1版本"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, CLIPVisionModel, PreTrainedModel, PretrainedConfig

class MiniEmbedVisionConfig(PretrainedConfig):

    model_type = "miniembedvision"

    def __init__(
        self,
        embed_dim: int =  768,
        text_model_name: str = "BAAI/bge-base-zh-v1.5",
        vision_model_name: str = "openai/clip-vit-base-patch32",
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

def _concat_all_gather(tensor):
    if not torch.distributed.is_initialized():
        return tensor
    tensors_gather = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)
    return torch.cat(tensors_gather, dim=0)

class MiniEmbedVisionModel(PreTrainedModel):
    config_class = MiniEmbedVisionConfig
    base_model_prefix = "miniembed_vision"

    def __init__(self, config):
        super().__init__(config)
        # 解析 text_model_name
        text_model_path = config.text_model_name
        if text_model_path.startswith("./") or os.path.isdir(text_model_path):
            # 转为绝对路径
            if hasattr(config, '_name_or_path') and config._name_or_path:
                base_dir = config._name_or_path
                resolved_text_path = os.path.abspath(os.path.join(base_dir, text_model_path))
            else:
                resolved_text_path = os.path.abspath(text_model_path)
            # 安全检查
            if not os.path.exists(resolved_text_path):
                raise FileNotFoundError(f"Text encoder path not found: {resolved_text_path}")
        else:
            # 视为hf路径
            resolved_text_path = text_model_path
        # 解析 vision_model_name
        vision_model_path = config.vision_model_name
        if vision_model_path.startswith("./") or os.path.isdir(vision_model_path):
            resolved_vision_path = vision_model_path
            if hasattr(config, '_name_or_path') and config._name_or_path:
                base_dir = config._name_or_path
                resolved_vision_path = os.path.abspath(os.path.join(base_dir, vision_model_path))
            else:
                resolved_vision_path = os.path.abspath(vision_model_path)
            # 安全检查
            if not os.path.exists(resolved_vision_path):
                raise FileNotFoundError(f"Vision encoder path not found: {resolved_vision_path}")
        else:
            resolved_vision_path = vision_model_path
        
        # 文本编码器
        self.text_encoder = AutoModel.from_pretrained(resolved_text_path)
        text_hidden = self.text_encoder.config.hidden_size
        # 若冻结文本编码器 freeze_text
        if config.freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            self.text_encoder.eval()
        
        # 视觉编码器
        self.vision_encoder = CLIPVisionModel.from_pretrained(resolved_vision_path)
        vis_hidden = self.vision_encoder.config.hidden_size

        # lora config
        if config.use_lora:
            if not _has_peft:
                raise ImportError("peft is required for LoRA. Please install: pip install peft")
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "proj"],
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type=TaskType.OTHER,
            )
            self.vision_encoder = get_peft_model(self.vision_encoder, lora_config)

        self.text_proj = nn.Linear(text_hidden, config.embed_dim) if text_hidden != config.embed_dim else nn.Identity()
        self.vision_proj = nn.Linear(vis_hidden, config.embed_dim) if vis_hidden != config.embed_dim else nn.Identity()
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
    
    def _bge_pool(self, last_hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_emb = torch.sum(last_hidden * mask, dim=1)
        sum_mask = mask.sum(dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_emb / sum_mask

    def encode_text(self, input_ids, attention_mask, **kwargs):
        with torch.no_grad() if not any(p.requires_grad for p in self.text_encoder.parameters()) else torch.enable_grad():
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            pooled = self._bge_pool(outputs.last_hidden_state, attention_mask)
        emb = self.text_proj(pooled)
        return F.normalize(emb, p=2, dim=-1)

    def encode_image(self, pixel_values, **kwargs):
        outputs = self.vision_encoder(pixel_values=pixel_values, return_dict=True)
        cls_feat = outputs.last_hidden_state[:, 0]
        emb = self.vision_proj(cls_feat)
        return F.normalize(emb, p=2, dim=-1)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        gather_for_ddp: bool = True,
        labels: Optional[torch.Tensor] = None,  # reserved for supervised contrastive
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
            text_emb = self.encode_text(input_ids, attention_mask)
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

        # Gather across GPUs for large-batch negatives
        if gather_for_ddp and torch.distributed.is_initialized():
            text_emb_all = _concat_all_gather(text_emb)
            image_emb_all = _concat_all_gather(image_emb)
        else:
            text_emb_all = text_emb
            image_emb_all = image_emb

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        logits_per_text = logit_scale * text_emb @ image_emb_all.t()   # [B, global_B]
        logits_per_image = logits_per_text.t()                         # [global_B, B]

        loss = None
        if return_loss:
            # Assume 1:1 pairing in local batch
            local_batch_size = text_emb.size(0)
            global_batch_size = text_emb_all.size(0)
            # Create labels: local i matches global i
            labels = torch.arange(local_batch_size, device=text_emb.device)
            # But logits_per_text is [local_B, global_B], so we need to align
            # Standard CLIP: each local text matches its corresponding image in global list
            # Find the global indices of local images
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                local_image_start = rank * local_batch_size
                image_indices = torch.arange(local_image_start, local_image_start + local_batch_size, device=text_emb.device)
                # We can't easily compute loss without knowing global alignment
                # Simpler: only use local batch for loss (common in practice)
                logits_local = logit_scale * text_emb @ image_emb.t()  # [B, B]
                labels_local = torch.arange(local_batch_size, device=text_emb.device)
                loss_i = F.cross_entropy(logits_local, labels_local)
                loss_t = F.cross_entropy(logits_local.t(), labels_local)
                loss = (loss_i + loss_t) / 2
            else:
                logits_local = logits_per_text  # [B, B]
                labels_local = torch.arange(local_batch_size, device=text_emb.device)
                loss_i = F.cross_entropy(logits_local, labels_local)
                loss_t = F.cross_entropy(logits_local.t(), labels_local)
                loss = (loss_i + loss_t) / 2

        return ModelOutput(
            loss=loss,
            logits_per_text=logits_per_text,
            logits_per_image=logits_per_image,
            text_embeds=text_emb,
            image_embeds=image_emb,
        )

MiniEmbedVisionConfig.register_for_auto_class()
MiniEmbedVisionModel.register_for_auto_class("AutoModel")