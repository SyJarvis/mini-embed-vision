from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor, AutoModel
import torch

model_dir = "./mini-embed-vision-v1.0"
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
image_processor = AutoImageProcessor.from_pretrained(model_dir, trust_remote_code=True)

query_text = "一个男人在图书馆里看书"

urls = [
    "data/0001.jpg",  # 猫
    "data/0002.jpg",  # 狗
    "data/000000039769.jpg",  # 两只猫
    "data/person_reading.jpg" # 人
]

# 编码查询文本
with torch.no_grad():
    query_input = tokenizer(
        query_text, return_tensors="pt", padding=True, truncation=True, max_length=768
    ).to("cuda")
    text_embedding = model.encode_text(**query_input) # [1, 768]

# 编码候选图像
images = []
for url in urls:
    img = Image.open(url).convert("RGB")
    images.append(img)

with torch.no_grad():
    image_inputs = image_processor(images, return_tensors="pt").to("cuda")
    image_embeddings = model.encode_image(**image_inputs) # [N, 768]

# 计算余弦相似度
# def similarity(a, b):
#     a = a / a.norm(dim=-1, keepdim=True)
#     b = b / b.norm(dim=-1, keepdim=True)
#     return torch.matmul(a, b.T)

# similarities = similarity(query_embedding, image_embeddings).squeeze(0)
similarities = torch.matmul(text_embedding, image_embeddings.T)  # [1, N]
similarities = similarities.squeeze(0)  # [N]
# similarities = torch.matmul(query_embedding, image_embeddings.T)

# 排序
ranked_result = sorted(zip(urls, similarities.tolist()), key=lambda x: x[1], reverse=True)
print(f"查询文本: {query_text}")
print("以图搜文-相关性排序结果:")
for i, (url, score) in enumerate(ranked_result, 1):
    print(f"{i}. {url} -> 相似度: {score:.4f}")