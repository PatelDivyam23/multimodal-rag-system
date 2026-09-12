"""Reciprocal Rank Fusion across visual, dense and lexical retrievers."""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.retrieval.text_search import TextIndex
from src.retrieval.visual_search import VisualIndex
from src.utils.config import load_config
from src.utils.logging import get_logger

log = get_logger()


def rrf(ranked_lists: dict[str, list[str]], k: int = 60) -> dict[str, dict]:
    """Reciprocal Rank Fusion.  score(d) = sum_r 1/(k + rank_r(d))"""
    fused: dict[str, float] = defaultdict(float)
    contrib: dict[str, dict] = defaultdict(dict)

    for source, pages in ranked_lists.items():
        for rank, page in enumerate(pages, 1):
            fused[page] += 1.0 / (k + rank)
            contrib[page][source] = rank

    return {
        page: {"score": score, "ranks": contrib[page]}
        for page, score in sorted(fused.items(), key=lambda x: -x[1])
    }

# page ranking is deleted instead of demoted
# def cap_per_paper(pages: list[str], meta: pd.DataFrame, max_per: int) -> list[str]:
#     """Source diversification: at most `max_per` pages from any one document."""
#     counts: dict[str, int] = defaultdict(int)
#     kept = []
#     for p in pages:
#         paper = meta.loc[p].paper_id
#         if counts[paper] < max_per:
#             kept.append(p)
#             counts[paper] += 1
#     return kept

def cap_per_paper(pages: list[str], meta: pd.DataFrame, max_per: int) -> list[str]:
    """Demote pages beyond `max_per` per document instead of discarding them."""
    counts: dict[str, int] = defaultdict(int)
    kept, overflow = [], []
    for p in pages:
        paper = meta.loc[p].paper_id
        if counts[paper] < max_per:
            kept.append(p)
            counts[paper] += 1
        else:
            overflow.append(p)
    return kept + overflow          # nothing is lost


class HybridRetriever:
    def __init__(self, cfg, load_visual: bool = True):
        self.cfg = cfg
        self.text = TextIndex(cfg)
        self.visual = VisualIndex(cfg) if load_visual else None
        self.pages = pd.read_parquet(Path(cfg.paths.index) / "pages.parquet") \
                       .set_index("image_path")

    # ---------- individual retrievers ----------
    def _visual(self, q: str, k: int) -> list[str]:
        if self.visual is None:
            return []
        df = self.visual.search(q, k=k)
        return df.image_path.tolist()

    def _dense(self, q: str, k: int) -> list[str]:
        return [p for p, _ in self.text.search_dense(q, k)]

    def _bm25(self, q: str, k: int) -> list[str]:
        return [p for p, _ in self.text.search_bm25(q, k)]

    # ---------- fused ----------
    def search(self, query: str, k: int = 5, mode: str = "fused") -> pd.DataFrame:
        r = self.cfg.retrieval

        lists: dict[str, list[str]] = {}
        if mode in ("fused", "visual"):
            lists["visual"] = self._visual(query, r.visual_top_k)
        if mode in ("fused", "dense"):
            lists["dense"] = self._dense(query, r.dense_top_k)
        if mode in ("fused", "bm25"):
            lists["bm25"] = self._bm25(query, r.bm25_top_k)

        fused = rrf(lists, k=r.rrf_k)
        order = list(fused.keys())

        if r.mmr.enabled:
            order = cap_per_paper(order, self.pages, r.mmr.max_per_paper)

        rows = []
        for rank, page in enumerate(order[:k], 1):
            m = self.pages.loc[page]
            rows.append({
                "rank": rank,
                "rrf": round(fused[page]["score"], 4),
                "paper_id": m.paper_id,
                "page": int(m.page_no) + 1,
                "fig": bool(m.has_figure),
                "image_path": page,              # ← add this
                "hits": ",".join(f"{s}@{rk}" for s, rk in fused[page]["ranks"].items()),
            })
        return pd.DataFrame(rows)


def main(config_name: str = "v1", query: str = "process state transition diagram",
         k: int = 5, compare: bool = False):
    cfg = load_config(config_name)
    ret = HybridRetriever(cfg)
    ret.visual._ensure_model()
    ret.text._ensure_model()                                   # warm up

    modes = ["visual", "dense", "bm25", "fused"] if compare else ["fused"]

    for mode in modes:
        t0 = time.perf_counter()
        df = ret.search(query, k=k, mode=mode)
        dt = (time.perf_counter() - t0) * 1000
        print(f"\n{'═' * 70}\n🔍 [{mode.upper()}]  {query!r}   ({dt:.0f} ms)")
        print(df.to_string(index=False) if len(df) else "  (no results)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="v1", dest="config_name")
    p.add_argument("--query", default="process state transition diagram")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--compare", action="store_true")
    main(**vars(p.parse_args()))

# comapre now
#python -m src.retrieval.fusion --query "process state transition diagram" --compare