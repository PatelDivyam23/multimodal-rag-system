"""Ephemeral in-memory index for user-uploaded documents."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from src.retrieval.maxsim import build_padded, maxsim


class SessionIndex:
    """One uploaded document, held for the lifetime of a session."""

    def __init__(self, vectors, meta, name,
                 text_vecs=None, bm25=None, chunks=None):
        self.padded = build_padded(vectors)
        self.meta = meta.reset_index(drop=True)
        self.name = name
        self.n_pages = len(vectors)
        self.n_docs = 1

        self.text_vecs = text_vecs          # [n_chunks, dim], normalized
        self.bm25 = bm25
        self.chunks = chunks                # DataFrame: image_path, text

    @property
    def has_text(self) -> bool:
        return self.bm25 is not None and self.chunks is not None

    # ---------- individual retrievers ----------
    def _visual_pages(self, query_emb, k: int) -> list[str]:
        scores = maxsim(self.padded, query_emb)
        top = np.argsort(-scores)[:k]
        return [self.meta.iloc[i].image_path for i in top]

    def _chunks_to_pages(self, scores: np.ndarray, k: int) -> list[str]:
        order = np.argsort(-scores)[: k * 3]
        seen: dict[str, float] = {}
        for i in order:
            if scores[i] <= 0:
                continue
            page = self.chunks.iloc[i].image_path
            seen.setdefault(page, float(scores[i]))
        return [p for p, _ in sorted(seen.items(), key=lambda x: -x[1])][:k]

    def _dense_pages(self, text_model, query: str, k: int) -> list[str]:
        if self.text_vecs is None or not len(self.text_vecs):
            return []
        q = text_model.encode([query], normalize_embeddings=True)[0]
        return self._chunks_to_pages(self.text_vecs @ q, k)

    def _bm25_pages(self, query: str, k: int) -> list[str]:
        if self.bm25 is None:
            return []
        from src.ingest.embed_text import tokenize
        return self._chunks_to_pages(np.asarray(self.bm25.get_scores(tokenize(query))), k)

    # ---------- public search ----------
    def search(self, query: str, query_emb, k: int = 5,
               mode: str = "visual", text_model=None, rrf_k: int = 10) -> pd.DataFrame:
        if mode != "visual" and not self.has_text:
            mode = "visual"

        if mode == "visual":
            order = self._visual_pages(query_emb, k)
            provenance = {p: f"visual@{i}" for i, p in enumerate(order, 1)}
        elif mode == "dense":
            order = self._dense_pages(text_model, query, k)
            provenance = {p: f"dense@{i}" for i, p in enumerate(order, 1)}
        elif mode == "bm25":
            order = self._bm25_pages(query, k)
            provenance = {p: f"bm25@{i}" for i, p in enumerate(order, 1)}
        else:                                           # fused
            lists = {
                "visual": self._visual_pages(query_emb, k * 4),
                "dense": self._dense_pages(text_model, query, k * 4),
                "bm25": self._bm25_pages(query, k * 4),
            }
            fused: dict[str, float] = defaultdict(float)
            contrib: dict[str, list[str]] = defaultdict(list)
            for src, pages in lists.items():
                for rank, p in enumerate(pages, 1):
                    fused[p] += 1.0 / (rrf_k + rank)
                    contrib[p].append(f"{src}@{rank}")
            order = [p for p, _ in sorted(fused.items(), key=lambda x: -x[1])][:k]
            provenance = {p: ",".join(contrib[p]) for p in order}

        # rows = []
        # for rank, page in enumerate(order, 1):
        #     m = self.meta[self.meta.image_path == page].iloc[0]
        #     rows.append({
        #         "rank": rank,
        #         "paper_id": self.name,
        #         "page": int(m.page_no) + 1,
        #         "fig": bool(m.has_figure),
        #         "image_path": page,
        #         "hits": provenance.get(page, ""),
        #     })
        # return pd.DataFrame(rows)
        rows = []
        for rank, page in enumerate(order, 1):
            m = self.meta[self.meta.image_path == page].iloc[0]
            rows.append({
                "rank": rank,
                "paper_id": m.doc_name if "doc_name" in self.meta.columns else self.name,
                "page": int(m.page_no) + 1,
                "fig": bool(m.has_figure),
                "image_path": page,
                "hits": provenance.get(page, ""),
            })
        return pd.DataFrame(rows)

    def page_text(self, image_path: str) -> str:
        if self.chunks is None:
            row = self.meta[self.meta.image_path == image_path]
            return row.iloc[0].text if len(row) else ""
        sel = self.chunks[self.chunks.image_path == image_path]
        return "\n".join(sel.text.tolist())