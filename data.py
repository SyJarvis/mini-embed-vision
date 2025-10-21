import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, CLIPFeatureExtractor
from transformers import CLIPImageProcessor
import json

class MultimodalDataset(Dataset):
    def __init__(self, root_dir, tokenizer, image_processor, max_text_len=128, image_transform=None, key_field='data'):
        """
        samples: list of {"text": str, "image_path": str}
        tokenizer: AutoTokenizer for BGE
        image_processor: CLIPImageProcessor for images
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
    root_dir = "/root/research/miniembedding/data/coco128"
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, use_fast=True)
    feature_extractor = CLIPFeatureExtractor.from_pretrained(VISION_MODEL)
    train_dataset = MultimodalDataset(root_dir, tokenizer, feature_extractor)
    
    BATCH_SIZE = 64
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_token_id=pad_token_id))

if __name__ == '__main__':
    main()