import torch
import torch.nn.functional as F


class RetrievalRerankPipeline:
    def __init__(self, model, device=None):
        if not getattr(model.config, "enable_reranker", False):
            raise ValueError("model must be created with enable_reranker=True")
        self.device = device or next(model.parameters()).device
        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def __call__(
        self,
        query_embeddings,
        image_embeddings,
        top_k=20,
        rerank_k=100,
    ):
        queries = F.normalize(query_embeddings.to(self.device), dim=-1).float()
        images = F.normalize(image_embeddings.to(self.device), dim=-1).float()
        retrieval_k = min(max(top_k, rerank_k), images.size(0))
        retrieval_scores = queries @ images.t()
        retrieval_indices = torch.argsort(retrieval_scores, dim=1, descending=True)[:, :retrieval_k]

        candidate_images = images[retrieval_indices]
        rerank_scores = self.model.score_rerank(queries, candidate_images)
        final_scores, relative_indices = torch.sort(rerank_scores, dim=1, descending=True)
        final_indices = torch.gather(retrieval_indices, 1, relative_indices)
        return final_indices[:, :top_k], final_scores[:, :top_k], retrieval_indices[:, :retrieval_k], retrieval_scores
