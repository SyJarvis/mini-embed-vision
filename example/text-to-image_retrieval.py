"""用 v2.2 导出权重做以文搜图：给定一段文本，对候选图片排序。"""
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
    parser.add_argument("--query", default="鼠标", help="查询文本")
    parser.add_argument(
        "--images",
        nargs="+",
        default=[
            "data/person_reading.jpg",
            "data/0001.jpg",
            "data/0002.jpg",
            "data/000000039769.jpg",
            "data/黑色鼠标.jpg",
            "data/pi07.png",
            "data/绿色鼠标.jpg"
        ],
        help="候选图片路径，多个用空格分隔",
    )
    parser.add_argument("--query-prefix", default="query: ", help="E5 查询前缀，与 v2.2 评测一致")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = AutoModel.from_pretrained(args.model_dir, trust_remote_code=True, local_files_only=True)
    model.to(args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True, use_fast=True, local_files_only=True)
    image_processor = build_timm_processor(str(Path(args.model_dir) / "preprocessor_config.json"))

    prefixed = (args.query_prefix or "") + args.query
    encoded = tokenizer(prefixed, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = encoded["input_ids"].to(args.device)
    attention_mask = encoded["attention_mask"].to(args.device)

    pixels = []
    for path in args.images:
        with Image.open(path) as img:
            pixels.append(image_processor(img.convert("RGB")))
    pixel_values = torch.stack(pixels).to(args.device)

    with torch.no_grad():
        text_embedding = model.encode_text(input_ids, attention_mask)
        image_embeddings = model.encode_image(pixel_values)

    similarities = (text_embedding @ image_embeddings.T).squeeze(0).tolist()

    print(f"查询文本: {args.query}")
    print("以文搜图-相关性排序结果:")
    for rank, (path, score) in enumerate(
        sorted(zip(args.images, similarities), key=lambda item: item[1], reverse=True), 1
    ):
        print(f"{rank}. {path} -> 相似度: {score:.4f}")


if __name__ == "__main__":
    main()
