"""Ephemeral in-memory index for user-uploaded documents."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.retrieval.maxsim import build_padded, maxsim


class SessionIndex:
    """Holds one uploaded document's pages for the lifetime of a session."""

    def __init__(self, vectors: list[np.ndarray], meta: pd.DataFrame, name: str):
        self.padded = build_padded(vectors)
        self.meta = meta.reset_index(drop=True)
        self.name = name
        self.n_pages = len(vectors)

    def search(self, query_emb, k: int = 5) -> pd.DataFrame:
        scores = maxsim(self.padded, query_emb)
        top = np.argsort(-scores)[:k]

        rows = []
        for rank, i in enumerate(top, 1):
            m = self.meta.iloc[i]
            rows.append({
                "rank": rank,
                "score": round(float(scores[i]), 2),
                "paper_id": self.name,
                "page": int(m.page_no) + 1,
                "fig": bool(m.has_figure),
                "image_path": m.image_path,
                "hits": f"visual@{rank}",
            })
        return pd.DataFrame(rows)

    def page_text(self, image_path: str) -> str:
        row = self.meta[self.meta.image_path == image_path]
        return row.iloc[0].text if len(row) else ""