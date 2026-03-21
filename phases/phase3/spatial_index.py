"""
SpatialIndex: O(log n) nearest neighbor search for GravitationalRelevance.
"""
import torch
from torch import Tensor
import numpy as np
from typing import Optional

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sklearn.neighbors import KDTree
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class SpatialIndex:
    def __init__(self, feature_dim: int = 512, rebuild_threshold: int = 1000, device: str = 'cuda'):
        self.feature_dim = feature_dim
        self.rebuild_threshold = rebuild_threshold
        self.device = device
        self.indexed_vectors = np.zeros((0, feature_dim), dtype=np.float32)
        self.buffer = []
        self._index = None
        self._index_type = None
        self._index_size = 0
        self._select_backend()

    def _select_backend(self):
        if FAISS_AVAILABLE: self._backend = 'faiss'
        elif SKLEARN_AVAILABLE: self._backend = 'sklearn'
        else: self._backend = 'brute'

    def add(self, vectors: Tensor):
        np_vecs = vectors.detach().cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(np_vecs, axis=1, keepdims=True) + 1e-8
        np_vecs = np_vecs / norms
        for v in np_vecs:
            self.buffer.append(v)
        if len(self.buffer) >= self.rebuild_threshold:
            self._rebuild_index()

    def search(self, queries: Tensor, k: int = 32) -> Tensor:
        is_single = queries.dim() == 1
        if is_single: queries = queries.unsqueeze(0)
        qs = queries.detach().cpu().numpy().astype(np.float32)
        qs = qs / (np.linalg.norm(qs, axis=1, keepdims=True) + 1e-8)
        all_vectors = self._get_all_vectors()
        n = len(all_vectors)
        if n == 0:
            res = torch.zeros((queries.shape[0], k), dtype=torch.long, device=self.device)
            return res.squeeze(0) if is_single else res
        actual_k = min(k, n)
        if self._backend == 'faiss' and self._index is not None and self._index_size > 0:
            indices = self._faiss_search(qs, actual_k)
        elif self._backend == 'sklearn' and self._index is not None:
            indices = self._sklearn_search(qs, actual_k)
        else:
            indices = self._brute_search(qs, actual_k, all_vectors)
        indices = np.array(indices, copy=True)
        if indices.ndim == 1: indices = indices.reshape(queries.shape[0], -1)
        if indices.shape[1] < k:
            pad = np.zeros((indices.shape[0], k - indices.shape[1]), dtype=np.int64)
            indices = np.concatenate([indices, pad], axis=1)
        indices = np.ascontiguousarray(indices)
        res = torch.tensor(indices, dtype=torch.long, device=self.device)
        return res.squeeze(0) if is_single else res

    def _rebuild_index(self):
        all_vecs = self._get_all_vectors()
        if len(all_vecs) == 0: return
        if self._backend == 'faiss': self._build_faiss(all_vecs)
        elif self._backend == 'sklearn': self._build_sklearn(all_vecs)
        self.indexed_vectors = all_vecs
        self.buffer = []
        self._index_size = len(all_vecs)

    def _get_all_vectors(self) -> np.ndarray:
        if len(self.buffer) == 0: return self.indexed_vectors
        buf = np.stack(self.buffer, axis=0)
        if len(self.indexed_vectors) == 0: return buf
        return np.concatenate([self.indexed_vectors, buf], axis=0)

    def _build_faiss(self, vectors: np.ndarray):
        index = faiss.IndexFlatIP(self.feature_dim)
        index.add(vectors)
        self._index = index
        self._index_type = 'faiss'

    def _faiss_search(self, queries: np.ndarray, k: int) -> np.ndarray:
        _, indices = self._index.search(queries, k)
        return indices

    def _build_sklearn(self, vectors: np.ndarray):
        self._index = KDTree(vectors, metric='euclidean')
        self._index_type = 'sklearn'

    def _sklearn_search(self, queries: np.ndarray, k: int) -> np.ndarray:
        _, indices = self._index.query(queries, k=k)
        return indices

    def _brute_search(self, queries: np.ndarray, k: int, vectors: np.ndarray) -> np.ndarray:
        sims = queries @ vectors.T
        return np.argsort(sims, axis=1)[:, ::-1][:, :k]

    def reset(self):
        self.indexed_vectors = np.zeros((0, self.feature_dim), dtype=np.float32)
        self.buffer = []
        self._index = None
        self._index_size = 0

    def save_state(self) -> dict:
        return {'indexed_vectors': self.indexed_vectors, 'buffer': self.buffer,
                'config': {'feature_dim': self.feature_dim, 'rebuild_threshold': self.rebuild_threshold}}

    def load_state(self, state: dict):
        self.indexed_vectors = state['indexed_vectors']
        self.buffer = state['buffer']
        if len(self.indexed_vectors) > 0: self._rebuild_index()
