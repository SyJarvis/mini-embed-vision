from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import requests
import torch

model_dir = "./mini-embed-vision-v1.0"
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to("cuda")
image_processor = AutoImageProcessor.from_pretrained(model_dir)

urls = [
    "http://images.cocodataset.org/val2017/000000039769.jpg"
]

def get_image_embeddings(urls, batch_size=4):
    embeddings = []
    for i in range(0, len(urls), batch_size):
        batch_urls = urls[i:i+batch_size]
        images = []
        for url in batch_urls:
            img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
            images.append(img)
        
        inputs = image_processor(images, return_tensors="pt").to("cuda")

        with torch.no_grad():
            emb = model.encode_image(inputs["pixel_values"])
        
        embeddings.append(emb)
    
    return torch.cat(embeddings, dim=0)

img_embs = get_image_embeddings(urls)
print(f"图像嵌入形状: {img_embs.shape}")  # [N, 768]
print(f"嵌入范数（应≈1.0）: {torch.norm(img_embs, dim=-1).mean().item()}")