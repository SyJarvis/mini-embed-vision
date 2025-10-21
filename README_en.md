---
language:
- en
license: apache-2.0
---

<div align="center">

![logo](./images/logb.png)

# mini-embed-vision

Lightweight Chinese Multimodal Embedding Model for Vision-Language Tasks

[中文](./README.md) | English

</div>

## 📌 Overview

* **mini-embed-vision** is a lightweight Chinese multimodal embedding model designed to provide individual developers with a reproducible, low-cost, and high-performance solution for joint vision-language representation learning.
* Built upon a **contrastive learning framework**, this project **freezes the text encoder** and aligns image and text embedding spaces using a trainable projection layer—significantly reducing training cost while maintaining strong cross-modal alignment.
* **Base models**:
  - Text encoder: `BAAI/bge-base-zh-v1.5`
  - Vision encoder: `openai/clip-vit-base-patch32`
* **Use cases**: Chinese image-text retrieval, multimodal search, content understanding, and deployment on resource-constrained edge devices.

---

## 📦 Project Structure

```
mini-embed-vision/
├── Model.py              # Multimodal model architecture
├── train.py              # Training script
├── data.py               # Data loading and preprocessing (based on COCO128, extensible)
├── example/              # Usage examples
│   ├── test_embed_image.py     # Image embedding example
│   └── test_image_text_ser.py  # Image-text retrieval example
├── requirements.txt      # Dependencies
└── README_en.md          # This document
```

> ✅ Compatible with mainstream Hugging Face ecosystem libraries such as `transformers`, `peft`, and `datasets`.

---

## 🚀 Quick Start

### 0. Environment Setup

#### 0.1 Clone the repository
```bash
git clone https://github.com/SyJarvis/mini-embed-vision.git
cd mini-embed-vision
```

#### 0.2 Install dependencies
```bash
pip install -r requirements.txt
```

---

### 1. Using the Pretrained Model

#### 1.1 Download the model (recommended via mirror for users in China)

> 💡 Due to limited access to Hugging Face servers in some regions, we recommend using the [HF-Mirror](https://hf-mirror.com) for faster downloads.

**Option 1: Using `huggingface-cli` with mirror**
```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download syjarvis/mini-embed-vision-v1.0 --local-dir ./mini-embed-vision-v1.0
```

**Option 2: Using ModelScope**
```bash
modelscope download --model shangye/mini-embed-vision-v1.0
```

#### 1.2 Run examples

- **Image embedding extraction**
  ```bash
  python example/test_embed_image.py
  ```

- **Image-text retrieval demo**
  ```bash
  python example/test_image_text_ser.py
  ```

---

### 2. Inference Examples

#### Image Embedding Extraction
```python
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor, AutoModel
import requests
import torch

model_dir = "./mini-embed-vision-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
image_processor = AutoImageProcessor.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to("cuda")

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
inputs = image_processor(image, return_tensors="pt").to("cuda")

with torch.no_grad():
    emb = model.encode_image(inputs["pixel_values"])  # shape: [1, 768]

print("✅ Image embedding shape:", emb.shape)
print("✅ Embedding norm (should be ~1.0):", torch.norm(emb, dim=-1).item())
```

#### Multimodal Image-Text Retrieval
```python
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor, AutoModel
import requests
import torch

model_dir = "./mini-embed-vision-v1.0"
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
image_processor = AutoImageProcessor.from_pretrained(model_dir)

# Load image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
image_inputs = image_processor(image, return_tensors="pt").to("cuda")

# Text queries
queries = [
    "Two cats sitting on a sofa",
    "A dog running on the grass",
    "Cute kittens",
    "A whale in the ocean",
    "Indoor pets"
]

# Encode image
with torch.no_grad():
    image_embedding = model.get_image_features(**image_inputs)  # [1, 768]

# Encode texts
text_embeddings = []
for text in queries:
    encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        emb = model.get_text_features(**encoded)  # [1, 768]
    text_embeddings.append(emb)

text_embeddings = torch.cat(text_embeddings, dim=0)  # [N, 768]

# Compute cosine similarities (embeddings are L2-normalized)
similarities = torch.matmul(image_embedding, text_embeddings.T).squeeze(0)

# Sort and display results
print("\n🔍 Image-Text Retrieval Results (sorted by relevance):")
results = sorted(zip(queries, similarities.tolist()), key=lambda x: x[1], reverse=True)
for i, (query, score) in enumerate(results, 1):
    print(f"{i}. {query} → Similarity: {score:.4f}")
```

---

### 3. Training from Scratch

> 🛠️ The training script is in `train.py`, currently supporting the COCO128 dataset. Contributions to extend support for more datasets or training strategies are welcome!

Basic training command:
```bash
python train.py
```

---

## 📝 Changelog

<details>
<summary><b>2025-10-20</b></summary>

- ✅ Implemented basic multimodal embedding architecture  
- ✅ Added support for image/text embedding extraction and cross-modal retrieval  
- ✅ Released v1.0 pretrained model on Hugging Face Hub

</details>

---

## 📌 Acknowledgment

> [!NOTE]  
> If you find `mini-embed-vision` helpful, please consider giving the repository a ⭐ on GitHub!  
> As the project is still evolving, we welcome bug reports, suggestions, and PRs via Issues.

---

## 📜 License

This project is licensed under the [Apache License 2.0](LICENSE).

---

## 🙏 Acknowledgements

* [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5)
* [OpenAI CLIP](https://github.com/openai/CLIP)
* [contrastors](https://github.com/nomic-ai/contrastors)