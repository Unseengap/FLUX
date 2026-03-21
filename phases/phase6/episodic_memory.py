import torch
import numpy as np
from typing import Dict, List, Optional

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

class EpisodicMemory:
    """
    Tier 2: External vector store for facts and events.
    Analogy: Your diary — specific events and facts.
    """
    def __init__(self, feature_dim: int = 256):
        self.feature_dim = feature_dim
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(feature_dim)
        else:
            self.index = None
            self.vectors = []
        self.metadata = []

    def write(self, vector: torch.Tensor, info: Dict):
        v = vector.detach().cpu().numpy().astype('float32')
        if len(v.shape) == 1:
            v = v.reshape(1, -1)

        if self.index is not None:
            self.index.add(v)
        else:
            self.vectors.append(v)

        self.metadata.append(info)

    def search(self, query_vector: torch.Tensor, k: int = 5) -> List[Dict]:
        q = query_vector.detach().cpu().numpy().astype('float32').reshape(1, -1)

        if self.index is not None:
            scores, indices = self.index.search(q, k)
            results = []
            for idx in indices[0]:
                if idx != -1 and idx < len(self.metadata):
                    results.append(self.metadata[idx])
            return results
        else:
            # Simple brute force if FAISS not available
            return self.metadata[:k] # Placeholder for real search
