# export_model.py
import os
import json
import torch
from transformers import AutoTokenizer, AutoImageProcessor

from utils import count_parameters

def export_model(
    save_dir: str = "mini-embed-vision-v1.0",
    checkpoint_path: str = "checkpoints/model_epoch100.pth",
    text_model_name: str = "BAAI/bge-base-zh-v1.5",
    image_model_name: str = "openai/clip-vit-base-patch32"
):
    
    os.makedirs(save_dir, exist_ok=True)
    # 构建config.json配置
    config_dict = {
        "model_type": "miniembedvision",
        "text_model_name": "BAAI/bge-base-zh-v1.5",
        "vision_model_name": "openai/clip-vit-base-patch32",
        "embed_dim": 768,
        "freeze_text": True,
        "torch_dtype": "float32",
        "architectures": ["MiniEmbedVisionModel"]
    }    

    # 保存 config.json
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    # 加载并保存模型权重 
    from model import MiniEmbedVisionModel, MiniEmbedVisionConfig

    # 确保模型类已注册
    MiniEmbedVisionConfig.register_for_auto_class()
    MiniEmbedVisionModel.register_for_auto_class("AutoModel")

    # 创建模型并加载权重
    config = MiniEmbedVisionConfig.from_pretrained(save_dir)  # 从刚保存的 config.json 加载
    model = MiniEmbedVisionModel(config)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    stats = count_parameters(model, human_readable=True)
    print("模型参数:")
    print(f"总参数:{stats['total']}")
    print(f"可训练参数:{stats['trainable']} ({stats['trainable_percent']}%)")
    print(f"冻结参数:{stats['frozen']}")

    # 保存模型
    model.save_pretrained(save_dir)
    
    # === 4. 保存 tokenizer 到子目录（推荐做法）===
    tokenizer = AutoTokenizer.from_pretrained(text_model_name, trust_remote_code=True)
    tokenizer.save_pretrained(save_dir)

    # === 5. 保存 image processor 到子目录 ===
    image_processor = AutoImageProcessor.from_pretrained(image_model_name)
    image_processor.save_pretrained(save_dir)

    print(f"模型已成功导出到 {save_dir}")

if __name__ == '__main__':
    export_model()