import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import CLIPImageProcessor
from torchvision import transforms
import json

class MultimodalParquetDataset(Dataset):
    """加载 pack_imagenet_hf.py 产出的 HF parquet 数据集（image/text/label/...）。

    与 MultimodalDataset 输出格式一致：{"input_ids", "attention_mask", "pixel_values"}。
    支持确定性 train/val 划分（按 source_id 哈希），保证多次运行划分一致且不重叠。
    """

    def __init__(self, data_dir, tokenizer, image_processor, max_text_len=128, split="train",
                 val_ratio=0.02, seed=42, image_transform=None, en_ratio=0.0):
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_text_len = max_text_len
        self.image_transform = image_transform
        # 双语配比：数据含 text_en 列时，每个样本以 en_ratio 概率取英文描述。
        # en_ratio=0 纯中文（默认，兼容旧数据），en_ratio=1 纯英文，0.5 混合。
        self.en_ratio = float(en_ratio)
        self.rng = random.Random(seed + (0 if split == "train" else 1))

        ds = load_dataset("parquet", data_dir=data_dir, split="train")
        # 划分：对 source_id 哈希取模，保证确定性与 train/val 不重叠
        keep = []
        for i in range(len(ds)):
            source_id = ds[i]["source_id"]
            bucket = hash_source_id(source_id, seed) % 10000
            if split == "train":
                if bucket >= int(val_ratio * 10000):
                    keep.append(i)
            else:  # val
                if bucket < int(val_ratio * 10000):
                    keep.append(i)
        self.ds = ds.select(keep)
        if len(self.ds) == 0:
            raise ValueError(f"Empty '{split}' split from {data_dir} (val_ratio={val_ratio})")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        text = row["text"]
        if self.en_ratio > 0.0:
            text_en = row.get("text_en") if isinstance(row, dict) else None
            if text_en and self.rng.random() < self.en_ratio:
                text = text_en
        img = row["image"]
        if not isinstance(img, Image.Image):
            import io as _io
            img = Image.open(_io.BytesIO(img)).convert("RGB")

        tok = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_text_len,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"].squeeze(0)
        attention_mask = tok["attention_mask"].squeeze(0)

        if self.image_transform is not None:
            pix = self.image_transform(img)
        elif isinstance(self.image_processor, transforms.Compose):
            pix = self.image_processor(img)
        else:
            pix = self.image_processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "pixel_values": pix}


def hash_source_id(source_id, seed=42):
    """确定性哈希：不依赖 Python 内置 hash()（每次进程不同）。"""
    import hashlib
    digest = hashlib.md5(f"{seed}:{source_id}".encode("utf-8")).hexdigest()
    return int(digest, 16)


class MultimodalDataset(Dataset):
    def __init__(self, root_dir, tokenizer, image_processor, max_text_len=128, image_transform=None, key_field='data'):
        """
        samples: list of {"text": str, "image_path": str}
        tokenizer: AutoTokenizer for BGE
            image_processor: CLIPImageProcessor or torchvision transform
        """
        self.root_dir = root_dir
        self.samples = []  # 存储 (图像路径, 标签) 的列表
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_text_len = max_text_len
        self.image_transform = image_transform
        json_file = os.path.join(root_dir, "image_descriptions.json")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            self.samples = data
        elif isinstance(data, dict) and key_field in data:
            self.samples = data[key_field]
        else:
            raise ValueError(f"Unsupported JSON format. Expected list or dict with key '{key_field}'")
        
        for item in self.samples:
            item['image_path'] = os.path.join(root_dir, item['image_path'])
        # 验证数据
        if len(self.samples) == 0:
            raise ValueError("JSON file is empty or contains no data")
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        text = s["text"]
        img_path = s["image_path"]
        # text tokenization (no padding here)
        tok = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_text_len,
            return_attention_mask=True,
            return_tensors="pt"
        )
        input_ids = tok["input_ids"].squeeze(0)
        attention_mask = tok["attention_mask"].squeeze(0)

        # image
        img = Image.open(img_path).convert("RGB")
        if self.image_transform is not None:
            pix = self.image_transform(img)  # should produce tensor C,H,W
        elif isinstance(self.image_processor, transforms.Compose):
            pix = self.image_processor(img)
        else:
            processed_image = self.image_processor(images=img, return_tensors="pt")
            pix = processed_image["pixel_values"].squeeze(0)  # [C,H,W]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "pixel_values": pix}

def collate_fn(batch, pad_token_id):
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    pixel_values = [b["pixel_values"] for b in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    pixel_values = torch.stack(pixel_values, dim=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "pixel_values": pixel_values}


def main():
    TEXT_MODEL = "BAAI/bge-base-zh-v1.5"
    VISION_MODEL = "openai/clip-vit-base-patch32"
    data_dir = "data/imagenet-1k-zh-captions"
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, use_fast=True)
    image_processor = CLIPImageProcessor.from_pretrained(VISION_MODEL)
    train_dataset = MultimodalParquetDataset(
        data_dir, tokenizer, image_processor, max_text_len=128, split="train", val_ratio=0.02
    )
    val_dataset = MultimodalParquetDataset(
        data_dir, tokenizer, image_processor, max_text_len=128, split="val", val_ratio=0.02
    )
    print(f"train={len(train_dataset)}, val={len(val_dataset)}")

    BATCH_SIZE = 64
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_token_id=pad_token_id))
    batch = next(iter(train_loader))
    print({k: v.shape for k, v in batch.items()})

if __name__ == '__main__':
    main()
