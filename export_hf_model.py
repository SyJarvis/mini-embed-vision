# export_model.py
import os
import json
import torch
from transformers import AutoTokenizer, AutoImageProcessor

from utils import count_parameters
# 加载并保存模型权重 
from model import MiniEmbedVisionModel, MiniEmbedVisionConfig

def export_model(
    save_dir: str = "mini-embed-vision-v1.0",
    checkpoint_path: str = "checkpoints/model_epoch100.pth",
    text_model_name: str = "BAAI/bge-base-zh-v1.5",
    vision_model_name: str = "openai/clip-vit-base-patch32"
):
    
    os.makedirs(save_dir, exist_ok=True)
    # === 确保模型类已注册
    MiniEmbedVisionConfig.register_for_auto_class()
    MiniEmbedVisionModel.register_for_auto_class("AutoModel")

    config = MiniEmbedVisionConfig(
        text_model_name=text_model_name,
        vision_model_name=vision_model_name,
        embed_dim=768,
        freeze_text=True,
        use_lora=False
    )

    model = MiniEmbedVisionModel(config)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
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

    # === 保存 image processor 到子目录 ===
    image_processor = AutoImageProcessor.from_pretrained(vision_model_name)
    image_processor.save_pretrained(save_dir)
    
    print("保存基座模型到本地子目录...")
    text_encoder_dir = os.path.join(save_dir, "text_encoder")
    vision_encoder_dir = os.path.join(save_dir, "vision_encoder")
    model.text_encoder.save_pretrained(text_encoder_dir)
    model.vision_encoder.save_pretrained(vision_encoder_dir)

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
        "text_model_name": "./text_encoder",
        "vision_model_name": "./vision_encoder",
        "embed_dim": 768,
        "freeze_text": True,
        "torch_dtype": "float32",
        "use_lora": False,
    }

    # 保存 config.json
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    print(f"模型已成功导出到 {save_dir}")

if __name__ == '__main__':
    export_model()