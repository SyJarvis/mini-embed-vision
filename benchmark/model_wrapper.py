"""
MiniEmbedVision model wrapper for MTEB benchmark.

Implements the MTEB AbsEncoder interface for the multimodal embedding model.
For text-only benchmarks, uses the text encoder (BGE-base-zh-v1.5).
"""

import sys
import os
from typing import TYPE_CHECKING

import torch

# 确保项目根目录在 sys.path 中，以便导入本地 mteb
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mteb.models.abs_encoder import AbsEncoder

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from typing_extensions import Unpack

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs, PromptType


class MiniEmbedVisionWrapper(AbsEncoder):
    """
    MTEB wrapper for MiniEmbedVision model.

    Implements AbsEncoder.encode() to process text inputs from MTEB DataLoader,
    using the model's encode_text() method with BGE pooling + L2 normalization.
    """

    def __init__(
        self,
        model_path: str = "./mini-embed-vision-v1.0",
        device: str = None,
        max_length: int = 512,
        batch_size: int = 32,
        **kwargs,
    ):
        from transformers import AutoModel, AutoTokenizer

        self.model_path = model_path
        self.max_length = max_length
        self.batch_size = batch_size

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[Wrapper] Loading model from: {model_path}")
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

        self.embedding_dim = getattr(self.model.config, "embed_dim", 768)
        print(f"[Wrapper] Loaded. dim={self.embedding_dim}, device={self.device}")

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
        """
        MTEB AbsEncoder interface.

        Flattens all texts from the DataLoader batches, then encodes them
        using the model's encode_text() with BGE pooling + L2 normalization.
        """
        # 将 DataLoader 中的所有 batch 展平为一个文本列表
        all_texts = [text for batch in inputs for text in batch["text"]]
        total = len(all_texts)
        print(f"[Wrapper] Encoding {total} texts (task={task_metadata.name}, split={hf_split})...")

        all_embeddings = []

        with torch.no_grad():
            for i in range(0, total, self.batch_size):
                batch_texts = all_texts[i : i + self.batch_size]

                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)

                embeddings = self.model.encode_text(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                )

                all_embeddings.append(embeddings.cpu())

                if (i + self.batch_size) % (self.batch_size * 10) == 0 or i + self.batch_size >= total:
                    print(f"  [{min(i + self.batch_size, total)}/{total}]")

        result = torch.cat(all_embeddings, dim=0)
        return result.numpy()

    @property
    def dimension(self) -> int:
        return self.embedding_dim

    @property
    def max_seq_length(self) -> int:
        return self.max_length
