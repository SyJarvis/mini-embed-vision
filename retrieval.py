import numpy as np
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False
from sklearn.neighbors import NearestNeighbors

def build_index(embeddings, index_type="faiss", n_list=100):
    """
    embeddings: numpy array [N, D]
    return index object and a search function index.search(queries, k)
    """
    if _HAS_FAISS:
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embeddings.astype(np.float32))
        return index
    else:
        nn = NearestNeighbors(n_neighbors=100, metric="cosine")  # cosine distance -> smaller = better
        nn.fit(embeddings)
        return nn

def search(index, queries, k=10):
    if _HAS_FAISS:
        D, I = index.search(queries.astype(np.float32), k)
        return I, D
    else:
        distances, indices = index.kneighbors(queries, n_neighbors=k)
        return indices, distances

def compute_recall_at_k(text_embs, img_embs, k_list=[1,5,10]):
    """
    text_embs: N x D, img_embs: N x D, aligned in same order (ground truth)
    """
    idx_map, dists = search(build_index(img_embs), text_embs, k=max(k_list))
    recalls = {}
    for k in k_list:
        correct = 0
        for i in range(len(text_embs)):
            topk = idx_map[i][:k]
            if i in topk:
                correct += 1
        recalls[f"R@{k}"] = correct / len(text_embs)
    return recalls