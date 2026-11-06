# export_model.py
import os
import json
import torch
from transformers import AutoTokenizer

from utils import count_parameters
# 加载并保存模型权重 
from model import MiniEmbedVisionModel, MiniEmbedVisionConfig

def export_model(
    save_dir: str = "mini-embed-vision-v1.0",
    checkpoint_path: str = "checkpoints/model_epoch100.pth",
    text_model_name: str = "BAAI/bge-base-zh-v1.5",
    vision_model_name: str = "openai/clip-vit-base-patch32",
    vision_backend: str = "transformers",
    timm_model_name: str = "eva02_base_patch14_224",
    text_pooling: str = "mean",
    enable_reranker: bool = False,
    reranker_hidden_dim: int = 256,
    reranker_dropout: float = 0.1,
    reranker_loss_weight: float = 0.2,
):
    
    os.makedirs(save_dir, exist_ok=True)
    # === 确保模型类已注册
    MiniEmbedVisionConfig.register_for_auto_class()
    MiniEmbedVisionModel.register_for_auto_class("AutoModel")

    config = MiniEmbedVisionConfig(
        text_model_name=text_model_name,
        vision_model_name=vision_model_name,
        vision_backend=vision_backend,
        timm_model_name=timm_model_name,
        text_pooling=text_pooling,
        enable_reranker=enable_reranker,
        reranker_hidden_dim=reranker_hidden_dim,
        reranker_dropout=reranker_dropout,
        reranker_loss_weight=reranker_loss_weight,
        embed_dim=768,
        freeze_text=True,
        use_lora=False
    )

    model = MiniEmbedVisionModel(config)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=True)

    stats = count_parameters(model, human_readable=True)
    print("模型参数:")
    print(f"总参数:{stats['total']}")
    print(f"可训练参数:{stats['trainable']} ({stats['trainable_percent']}%)")
    print(f"冻结参数:{stats['frozen']}")

    # ===保存模型
    print("保存完整模型权重（含基座）...")
    model.save_pretrained(save_dir)

    # === 保存 tokenizer 和 image processor 到标准子目录
    tokenizer = AutoTokenizer.from_pretrained(text_model_name, trust_remote_code=True)
    tokenizer.save_pretrained(save_dir)

    print("内嵌文本编码器配置...")
    if vision_backend == "timm":
        image_config = {
            "backend": "timm",
            "architecture": timm_model_name,
            "input_size": 224,
            "interpolation": "bicubic",
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711],
        }
        with open(os.path.join(save_dir, "preprocessor_config.json"), "w", encoding="utf-8") as f:
            json.dump(image_config, f, indent=2)
    else:
        from transformers import AutoImageProcessor
        image_processor = AutoImageProcessor.from_pretrained(vision_model_name)
        image_processor.save_pretrained(save_dir)

    print("更新config.json，指向本地模型...")
    config_dict = {
        "model_type": "miniembedvision",
        "architectures": [
            "MiniEmbedVisionModel"
        ],
        "auto_map": {
            "AutoConfig": "model.MiniEmbedVisionConfig",
            "AutoModel": "model.MiniEmbedVisionModel"
        },
        "text_model_name": "",
        "vision_model_name": "",
        "vision_backend": vision_backend,
        "timm_model_name": timm_model_name,
        "embed_dim": 768,
        "freeze_text": True,
        "torch_dtype": "float32",
        "use_lora": False,
        "text_pooling": model.config.text_pooling,
        "projection_dropout": model.config.projection_dropout,
        "enable_reranker": enable_reranker,
        "reranker_hidden_dim": reranker_hidden_dim,
        "reranker_dropout": reranker_dropout,
        "reranker_loss_weight": reranker_loss_weight,
        "text_config": model.text_encoder.config.to_dict(),
    }

    # 保存 config.json
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    print(f"模型已成功导出到 {save_dir}")

if __name__ == '__main__':
    export_model()
