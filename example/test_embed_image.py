import os
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor, AutoModel
import requests
model_dir = "./mini-embed-vision-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
image_processor = AutoImageProcessor.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
model.to("cuda")
model.eval()


url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

inputs = image_processor(image, return_tensors="pt").to("cuda")

import torch
# 4. 获取图像嵌入（768 维归一化向量）
with torch.no_grad():
    emb = model.encode_image(inputs.pixel_values)  # shape: [1, 768]
    emb = model.encode_image(inputs["pixel_values"])  # shape: [1, 768]

print("图像嵌入形状:", emb.shape)
print("嵌入范数（应为 1.0）:", torch.norm(emb, dim=-1).item())
print(emb.cpu().numpy().squeeze())