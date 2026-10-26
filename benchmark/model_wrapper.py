"""
MiniEmbedVision MTEB wrapper for multi-modal evaluation.

Implements the MTEB AbsEncoder interface for the MiniEmbedVision model,
supporting text-only, image-only, and retrieval (text-query, image-corpus) tasks.

Usage:
    from benchmark.model_wrapper import MiniEmbedVisionMTEBWrapper
    model = MiniEmbedVisionMTEBWrapper("checkpoints/best.pth")
"""

import os
import sys
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer

# Ensure project root is on sys.path for local imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from model import MiniEmbedVisionModel, MiniEmbedVisionConfig

from mteb.models.abs_encoder import AbsEncoder

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from typing_extensions import Unpack

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs, PromptType


class MiniEmbedVisionMTEBWrapper(AbsEncoder):
    """MTEB wrapper for MiniEmbedVision multi-modal embedding model.

    Supports:
    - Text inputs: tokenizes via BGE-base-zh-v1.5 → encode_text() → mean pooling + L2 norm
    - Image inputs: processes via CLIP-ViT-B/32 processor → encode_image() → CLS token + L2 norm
    - Both modalities share the same 768-d embedding space, enabling cross-modal retrieval.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str | None = None,
        *,
        batch_size: int = 32,
        max_length: int = 512,
        text_model_name: str = "BAAI/bge-base-zh-v1.5",
        vision_model_name: str = "openai/clip-vit-base-patch32",
        vision_backend: str = "transformers",
        timm_model_name: str = "eva02_base_patch14_224",
        enable_reranker: bool = False,
        local_vision_path: str | None = None,
        local_text_path: str | None = None,
        **kwargs: Any,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.max_length = max_length

        # Use local model dirs if they exist (network may be blocked)
        _project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if local_vision_path is None:
            local_candidate = (
                os.path.join(_project_dir, "eva02_base_patch14_224.mim_in22k")
                if vision_backend == "timm"
                else os.path.join(_project_dir, "clip-vit-base-patch32")
            )
            if os.path.isdir(local_candidate):
                local_vision_path = local_candidate
        if local_text_path is None:
            local_candidate = os.path.join(_project_dir, "bge-base-zh-v1.5")
            if os.path.isdir(local_candidate):
                local_text_path = local_candidate

        resolved_vision = local_vision_path or vision_model_name
        resolved_text = local_text_path or text_model_name

        # ── load model from .pth checkpoint ──────────────────────────
        print(f"[Wrapper] Loading checkpoint: {checkpoint_path}")
        print(f"[Wrapper] Device: {self.device}")
        if not os.path.isdir(checkpoint_path):
            print(f"[Wrapper] Text encoder: {resolved_text}")
            print(f"[Wrapper] Vision encoder: {resolved_vision}")

        if os.path.isdir(checkpoint_path):
            print("[Wrapper] Mode: exported Hugging Face model")
            self.model = AutoModel.from_pretrained(
                checkpoint_path,
                trust_remote_code=True,
                local_files_only=True,
            )
            self.config = self.model.config
            self.model.to(self.device)
            self.model.eval()
        else:
            self.config = MiniEmbedVisionConfig(
                text_model_name=resolved_text,
                vision_model_name=resolved_vision,
                vision_backend=vision_backend,
                timm_model_name=timm_model_name,
                enable_reranker=enable_reranker,
                freeze_text=True,
            )
            self.model = MiniEmbedVisionModel(self.config)
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            self.model.load_state_dict(state, strict=False)
            self.model.to(self.device)
            self.model.eval()

            epoch = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
            print(f"[Wrapper] Checkpoint loaded (epoch={epoch})")

        # ── tokenizer & image processor ──────────────────────────────
        if os.path.isdir(checkpoint_path):
            self.tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_path, use_fast=True, local_files_only=True
            )
            processor_path = os.path.join(checkpoint_path, "preprocessor_config.json")
            backend = self.config.vision_backend
            if backend == "timm":
                self.image_processor = self._build_timm_processor(processor_path)
            else:
                from transformers import CLIPImageProcessor

                self.image_processor = CLIPImageProcessor.from_pretrained(
                    checkpoint_path, local_files_only=True
                )
            self.embedding_dim = getattr(self.config, "embed_dim", 768)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(resolved_text, use_fast=True)
            if vision_backend == "timm":
                processor_path = os.path.join(
                    os.path.dirname(os.path.abspath(checkpoint_path)),
                    "preprocessor_config.json",
                )
                self.image_processor = self._build_timm_processor(processor_path)
            else:
                from transformers import CLIPImageProcessor

                self.image_processor = CLIPImageProcessor.from_pretrained(resolved_vision)

        # ── MTEB model metadata ──────────────────────────────────────
        self.embedding_dim = getattr(self.config, "embed_dim", 768)
        print(f"[Wrapper] dim={self.embedding_dim}, device={self.device}")

        from mteb.models.model_meta import ModelMeta

        self.mteb_model_meta = ModelMeta(
            loader=None,
            name="teamix/mini-embed-vision",
            revision="v1.5",
            release_date=None,
            languages=["cmn", "eng-Latn"],
            modalities=["text", "image"],
            embed_dim=self.embedding_dim,
            framework=["PyTorch"],
            similarity_fn_name=None,
            use_instructions=False,
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=self.max_length,
            license=None,
            open_weights=None,
            public_training_code=None,
            public_training_data=None,
            training_datasets=None,
        )

    def _build_timm_processor(self, processor_path: str) -> transforms.Compose:
        import json

        with open(processor_path, "r", encoding="utf-8") as stream:
            config = json.load(stream)
        size = config.get("input_size", 224)
        if isinstance(size, list):
            size = size[-1]
        mean = config.get("mean", [0.48145466, 0.4578275, 0.40821073])
        std = config.get("std", [0.26862954, 0.26130258, 0.27577711])
        interpolation = transforms.InterpolationMode[
            config.get("interpolation", "bicubic").upper()
        ]
        return transforms.Compose(
            [
                transforms.Resize(size, interpolation=interpolation),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


    # ── encode: main MTEB interface ──────────────────────────────────

    def encode(
        self,
        inputs: "DataLoader[BatchedInput]",
        *,
        task_metadata: "TaskMetadata",
        hf_split: str,
        hf_subset: str,
        prompt_type: "PromptType | None" = None,
        **kwargs: "Unpack[EncodeKwargs]",
    ) -> "Array":
        """Encode text and/or image inputs depending on task modality.

        Dispatches to encode_text() or encode_image() based on the presence
        of the "text" / "image" keys in each batch.  For retrieval tasks MTEB
        already filters the dataset columns per modality (query → text only,
        corpus → image only for t2i), so this method rarely sees both keys.
        """
        all_embeddings: list[torch.Tensor] = []
        total_items = 0

        for batch_idx, batch in enumerate(inputs):
            has_text = "text" in batch and len(batch["text"]) > 0
            has_image = "image" in batch and len(batch["image"]) > 0

            if has_text and not has_image:
                # ── text-only (e.g. retrieval queries) ────────────────
                texts: list[str] = batch["text"]
                batch_emb = self._encode_texts(texts)
                all_embeddings.append(batch_emb.cpu())
                total_items += len(texts)

            elif has_image and not has_text:
                # ── image-only (e.g. classification, retrieval corpus) ─
                images: list = batch["image"]
                batch_emb = self._encode_images(images)
                all_embeddings.append(batch_emb.cpu())
                total_items += len(images)

            else:
                # ── both (or neither) — defensive fallback ────────────
                # For our task set this code path should never be hit;
                # MTEB already filters down to the active modality per
                # prompt_type.  We fall back based on prompt_type heuristics.
                if prompt_type is not None and prompt_type.value == "query":
                    texts = batch.get("text", [])
                    if texts:
                        batch_emb = self._encode_texts(texts)
                        all_embeddings.append(batch_emb.cpu())
                        total_items += len(texts)
                        continue
                images = batch.get("image", [])
                if images:
                    batch_emb = self._encode_images(images)
                    all_embeddings.append(batch_emb.cpu())
                    total_items += len(images)
                    continue
                # Last resort: try text
                texts = batch.get("text", [])
                if texts:
                    batch_emb = self._encode_texts(texts)
                    all_embeddings.append(batch_emb.cpu())
                    total_items += len(texts)

        if not all_embeddings:
            raise RuntimeError(
                f"No embeddings generated (task={task_metadata.name}, "
                f"modalities={task_metadata.modalities})"
            )

        result = torch.cat(all_embeddings, dim=0)
        return result.numpy()

    # ── modality-specific encoding helpers ───────────────────────────

    def _encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Encode a batch of texts → [N, embed_dim], L2-normalised."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            autocast_device = "cuda" if self.device.startswith("cuda") else self.device
            with torch.autocast(
                device_type=autocast_device,
                dtype=torch.bfloat16,
                enabled=autocast_device == "cuda",
            ):
                embeddings = self.model.encode_text(input_ids, attention_mask)

        return embeddings.float()

    def _encode_images(self, images: list) -> torch.Tensor:
        """Encode a batch of PIL images → [N, embed_dim], L2-normalised."""
        from PIL import Image as PILImage

        # Ensure all images are PIL RGB
        rgb_images = []
        for img in images:
            if not isinstance(img, PILImage.Image):
                img = PILImage.open(img).convert("RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")
            rgb_images.append(img)

        if isinstance(self.image_processor, transforms.Compose):
            pixel_values = torch.stack([self.image_processor(image) for image in rgb_images]).to(self.device)
        else:
            pixel_values = self.image_processor(
                images=rgb_images, return_tensors="pt"
            )["pixel_values"].to(self.device)

        with torch.no_grad():
            autocast_device = "cuda" if self.device.startswith("cuda") else self.device
            with torch.autocast(
                device_type=autocast_device,
                dtype=torch.bfloat16,
                enabled=autocast_device == "cuda",
            ):
                embeddings = self.model.encode_image(pixel_values)

        return embeddings.float()

    # ── properties ──────────────────────────────────────────────────

    @property
    def dimension(self) -> int:
        return self.embedding_dim

    @property
    def max_seq_length(self) -> int:
        return self.max_length
