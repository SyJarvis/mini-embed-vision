import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

from nomic_runtime.configuration_hf_nomic_bert import NomicBertConfig
from nomic_runtime.modeling_hf_nomic_bert import NomicBertModel, NomicVisionModel


class LocalNomicEmbedVision:
    def __init__(
        self,
        vision_path="/home/runke.zhong/.cache/modelscope/hub/models/tiansz/nomic-embed-vision-v1___5",
        text_path="nomic-embed-text-v1.5",
        device="cuda:0",
        batch_size=64,
    ):
        self.device = device
        self.batch_size = batch_size
        from safetensors.torch import load_file

        vision_config = NomicBertConfig.from_pretrained(vision_path)
        self.vision_model = NomicVisionModel(vision_config)
        self.vision_model.load_state_dict(load_file(f"{vision_path}/model.safetensors"))
        self.vision_model.eval().to(device)

        text_config = NomicBertConfig.from_pretrained(text_path)
        self.text_model = NomicBertModel(text_config, add_pooling_layer=False)
        self.text_model.load_state_dict(load_file(f"{text_path}/model.safetensors"))
        self.text_model.eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(text_path, local_files_only=True)

        self.image_processor = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

    @torch.inference_mode()
    def encode_texts(self, texts):
        embeddings = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                [f"search_query: {text}" for text in batch],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            output = self.text_model(
                input_ids=encoded["input_ids"].to(self.device),
                attention_mask=encoded["attention_mask"].to(self.device),
                return_dict=True,
            )
            hidden = output.last_hidden_state
            mask = encoded["attention_mask"].to(self.device).unsqueeze(-1)
            mask = mask.expand(hidden.size()).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = F.layer_norm(pooled, normalized_shape=(pooled.shape[-1],))
            embeddings.append(F.normalize(pooled, dim=-1).cpu())
        return torch.cat(embeddings)

    @torch.inference_mode()
    def encode_images(self, images):
        embeddings = []
        for start in range(0, len(images), self.batch_size):
            batch = images[start : start + self.batch_size]
            rgb_images = [image.convert("RGB") if image.mode != "RGB" else image for image in batch]
            pixel_values = torch.stack(
                [self.image_processor(image) for image in rgb_images]
            ).to(self.device)
            output = self.vision_model(pixel_values, return_dict=True)
            embeddings.append(F.normalize(output.last_hidden_state[:, 0], dim=-1).cpu())
        return torch.cat(embeddings)
