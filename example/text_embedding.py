from transformers import AutoTokenizer, AutoModel
import torch

model_dir = "./mini-embed-vision-v1.0"
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_dir)

texts = [
    "一只橘猫在窗台上晒太阳",
    "红色跑车在高速公路上疾驰",
    "春天的樱花盛开，美不胜收",
    "人工智能正在改变世界",
    "中国传统文化博大精深"
]

def get_text_embeddings(texts, batch_size=8):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=768,
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            emb = model.encode_text(**inputs)
        embeddings.append(emb.cpu())
    
    return torch.cat(embeddings, dim=0)

text_embs = get_text_embeddings(texts)
print(f"文本嵌入形状: {text_embs.shape}")
print(f"嵌入范数（应≈1.0）: {torch.norm(text_embs, dim=-1).mean().item()}")