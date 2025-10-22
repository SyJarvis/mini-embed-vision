import os
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor, AutoModel
import requests
import torch

model_dir = "./mini-embed-vision-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
image_processor = AutoImageProcessor.from_pretrained(model_dir)
vision_model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
vision_model.to("cuda")

# 3. 准备图像（示例：COCO 猫图）
print("正在下载示例图像...")
url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # 两只猫
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# 预处理图像
image_inputs = image_processor(images=image, return_tensors="pt").to("cuda")

# 4. 准备文本查询（模拟搜索 query）
queries = [
    "两只猫坐在沙发上",
    "一只狗在草地上奔跑",
    "可爱的猫咪",
    "海洋中的鲸鱼",
    "室内宠物"
]

# 5. 编码图像（得到 [1, 768] 归一化向量）
with torch.no_grad():
    # outputs = vision_model(pixel_values=image_inputs["pixel_values"])  # shape: [1, 768]
    image_embedding = vision_model.encode_image(image_inputs["pixel_values"])
    # image_embedding = outputs.image_embeds    
print(f"图像嵌入形状: {image_embedding.shape}")

texts = queries
# 6. 编码所有文本（得到 [N, 768] 归一化向量）
text_embeddings = []
for text in texts:
    # BGE tokenizer 需要 return_tensors="pt"
    text_inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to("cuda")

    with torch.no_grad():
        text_embedding = vision_model(input_ids=text_inputs["input_ids"], attention_mask=text_inputs["attention_mask"])
        # emb = vision_model.encode_text(**text_inputs)  # [1, 768]
    text_embeddings.append(text_embedding.last_hidden_state)

text_embeddings = torch.cat(text_embeddings, dim=0)  # [N, 768]
print(f"文本嵌入形状: {text_embeddings.shape}")

# 7. 计算余弦相似度（因已 L2 归一化，点积 = 余弦相似度）
# image_embedding: [1, 768]
# text_embeddings: [N, 768]
similarities = torch.matmul(image_embedding, text_embeddings.T)  # [1, N]
similarities = similarities.squeeze(0)  # [N]

# 6. 计算余弦相似度（因已 L2 归一化，点积即余弦）
similarities = (image_embedding @ text_embeddings.T).squeeze(0)  # [N]

# 8. 输出结果（按相似度排序）
print("以文搜图-图文检索结果（按相关性排序）:")
results = list(zip(queries, similarities.tolist()))
results.sort(key=lambda x: x[1], reverse=True)

for i, (query, score) in enumerate(results, 1):
    print(f"{i}. {query} → 相似度: {score:.4f}")