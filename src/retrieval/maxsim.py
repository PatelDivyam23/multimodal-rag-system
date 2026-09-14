"""MaxSim scoring over ragged multi-vector page embeddings."""
from __future__ import annotations

import numpy as np
import torch

PAD = -1e4


def build_padded(vectors: list[np.ndarray]) -> torch.Tensor:
    """Stack variable-length page embeddings into [n_pages, max_len, dim]."""
    n = len(vectors)
    dim = vectors[0].shape[1]
    max_len = max(v.shape[0] for v in vectors)

    padded = torch.full((n, max_len, dim), PAD, dtype=torch.float32)
    for i, v in enumerate(vectors):
        padded[i, : v.shape[0]] = torch.from_numpy(v).float()
    return padded


def maxsim(padded: torch.Tensor, query: torch.Tensor) -> np.ndarray:
    """query: [n_tok, dim] -> one score per page."""
    sim = torch.einsum("pkd,qd->pqk", padded, query)
    return sim.max(dim=2).values.sum(dim=1).numpy()