"""用 v2.2 导出权重做以图搜文：给定一张图片，对候选文本排序。"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer


def build_timm_processor(preprocessor_path: str) -> transforms.Compose:
    with open(preprocessor_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    size = cfg.get("input_size", 224)
    if isinstance(size, list):
        size = size[-1]
    mean = cfg.get("mean", [0.48145466, 0.4578275, 0.40821073])
    std = cfg.get("std", [0.26862954, 0.26130258, 0.27577711])
    interpolation = transforms.InterpolationMode[cfg.get("interpolation", "bicubic").upper()]
    return transforms.Compose(
        [
            transforms.Resize(size, interpolation=interpolation),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="./mini-embed-vision-v2.2")
    parser.add_argument("--image", default="data/0003.jpg", help="查询图片路径")
    parser.add_argument(
        "--texts",
        nargs="+",
        default=[
            "一个人在室内看书",
            "两只猫坐在沙发上",
            "一只狗在草地上奔跑",
            "海边的自行车",
            "Decoding throughput vs concurrency",
            "pi0.7通用模型，使用多种多模态提示进行训练。包括语言、元数据、控制模式和视觉子目标",
            "鼠标"
        ],
        help="候选文本，多个用空格分隔",
    )
    parser.add_argument("--query-prefix", default="query: ", help="E5 查询前缀，与 v2.2 评测一致")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = AutoModel.from_pretrained(args.model_dir, trust_remote_code=True, local_files_only=True)
    model.to(args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True, local_files_only=True)
    image_processor = build_timm_processor(str(Path(args.model_dir) / "preprocessor_config.json"))

    with Image.open(args.image) as img:
        pixel_values = image_processor(img.convert("RGB")).unsqueeze(0).to(args.device)

    prefixed = [(args.query_prefix or "") + t for t in args.texts]
    encoded = tokenizer(prefixed, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = encoded["input_ids"].to(args.device)
    attention_mask = encoded["attention_mask"].to(args.device)

    with torch.no_grad():
        image_embedding = model.encode_image(pixel_values)
        text_embeddings = model.encode_text(input_ids, attention_mask)

    similarities = (image_embedding @ text_embeddings.T).squeeze(0).tolist()

    print(f"查询图片: {args.image}")
    print("以图搜文-相关性排序结果:")
    for rank, (text, score) in enumerate(
        sorted(zip(args.texts, similarities), key=lambda item: item[1], reverse=True), 1
    ):
        print(f"{rank}. {text} -> 相似度: {score:.4f}")


if __name__ == "__main__":
    main()
