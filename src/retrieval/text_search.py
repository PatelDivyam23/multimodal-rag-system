"""Dense (BGE) + lexical (BM25) retrieval over text chunks."""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.utils.logging import get_logger

log = get_logger()


def tokenize(text: str) -> list[str]:
    """Must match the tokenizer used at ingestion time."""
    return re.findall(r"[a-z0-9][a-z0-9\-_]*", text.lower())


class TextIndex:
    """Chunk-level dense + BM25 search, results aggregated to pages."""

    def __init__(self, cfg):
        self.cfg = cfg
        idx = Path(cfg.paths.index)

        self.chunks = pd.read_parquet(idx / "chunks.parquet")
        self.vectors = np.load(idx / "text_vectors.npy")        # normalized
        with open(idx / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)

        log.info("Text index: %d chunks | dense %s",
                 len(self.chunks), self.vectors.shape)
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = SentenceTransformer(self.cfg.text.embed_model, device=device)

    # ---------- dense ----------
    def search_dense(self, query: str, k: int) -> list[tuple[str, float]]:
        self._ensure_model()
        q = self._model.encode([query], normalize_embeddings=True)[0]
        scores = self.vectors @ q                                # cosine
        top = np.argsort(-scores)[: k * 3]                       # over-fetch chunks
        return self._to_pages(top, scores, k)

    # ---------- bm25 ----------
    def search_bm25(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[: k * 3]
        return self._to_pages(top, scores, k)

    # ---------- chunk hits -> page hits ----------
    def _to_pages(self, idxs, scores, k: int) -> list[tuple[str, float]]:
        seen: dict[str, float] = {}
        for i in idxs:
            if scores[i] <= 0:
                continue
            page = self.chunks.iloc[i].image_path
            if page not in seen:                                 # keep best chunk
                seen[page] = float(scores[i])
        ranked = sorted(seen.items(), key=lambda x: -x[1])
        return ranked[:k]