"""Load the ragged ColQwen2 index and run MaxSim retrieval."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.utils.config import load_config
from src.utils.logging import get_logger

log = get_logger()


class VisualIndex:
    """Ragged multi-vector store + brute-force MaxSim."""

    def __init__(self, cfg):
        self.cfg = cfg
        idx = Path(cfg.paths.index)

        vecs = np.load(idx / "visual_vectors.npy")
        self.offsets = np.load(idx / "visual_offsets.npy")
        self.keys = pd.read_parquet(idx / "visual_keys.parquet")["image_path"].tolist()
        self.pages = pd.read_parquet(idx / "pages.parquet").set_index("image_path")

        self.vectors = torch.from_numpy(vecs).float()          # [N, 128]
        self.n_pages = len(self.keys)
        self.dim = self.vectors.shape[1]

        # ── CHANGE 1: build padded tensor for vectorised MaxSim ──────────
        lens = np.diff(self.offsets)
        self.max_len = int(lens.max())
        pad = torch.full((self.n_pages, self.max_len, self.dim), -1e4)
        for p in range(self.n_pages):
            lo, hi = self.offsets[p], self.offsets[p + 1]
            pad[p, : hi - lo] = self.vectors[lo:hi]
        self.padded = pad
        # ─────────────────────────────────────────────────────────────────

        log.info("Index: %d pages | %d vectors | %.1f MB raw | %.0f MB padded",
                 self.n_pages, self.vectors.shape[0],
                 vecs.nbytes / 1e6, pad.element_size() * pad.nelement() / 1e6)

        self._model = None
        self._processor = None

    # ---------- query encoding ----------
    def _ensure_model(self):
        if self._model is not None:
            return
        from colpali_engine.models import ColQwen2, ColQwen2Processor
        from transformers import BitsAndBytesConfig

        q = self.cfg.visual.quantization
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=q.quant_type,
            bnb_4bit_compute_dtype=getattr(torch, q.compute_dtype),
            bnb_4bit_use_double_quant=q.double_quant,
        ) if q.enabled else None

        log.info("Loading query encoder ...")
        self._model = ColQwen2.from_pretrained(
            self.cfg.visual.model_id,
            torch_dtype=torch.float16,
            quantization_config=quant,
            device_map="cuda:0",
        ).eval()
        self._processor = ColQwen2Processor.from_pretrained(self.cfg.visual.model_id)

    @torch.no_grad()
    def encode_query(self, query: str) -> torch.Tensor:
        self._ensure_model()
        batch = self._processor.process_queries([query]).to(self._model.device)
        emb = self._model(**batch)                             # [1, n_tok, 128]
        return emb[0].float().cpu()

    # ── CHANGE 2: vectorised MaxSim (replaces the old loop) ──────────────
    def maxsim_scores(self, q: torch.Tensor) -> np.ndarray:
        """q: [n_tok, 128] -> score per page."""
        sim = torch.einsum("pkd,qd->pqk", self.padded, q)      # [pages, q_tok, patches]
        return sim.max(dim=2).values.sum(dim=1).numpy()
    # ─────────────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 5) -> pd.DataFrame:
        q = self.encode_query(query)
        scores = self.maxsim_scores(q)
        top = np.argsort(-scores)[:k]

        rows = []
        for rank, i in enumerate(top, 1):
            key = self.keys[i]
            meta = self.pages.loc[key]
            rows.append({
                "rank": rank,
                "score": round(float(scores[i]), 2),
                "paper_id": meta.paper_id,
                "page": int(meta.page_no) + 1,
                "figure": bool(meta.has_figure),
                "image_path": key,
            })
        return pd.DataFrame(rows)


def main(config_name: str = "v1", query: str | None = None, k: int = 5):
    cfg = load_config(config_name)
    index = VisualIndex(cfg)

    index._ensure_model()          # ── CHANGE 3: warm up before timing ──

    queries = [query] if query else [
        "attention mechanism architecture diagram",
        "table comparing model accuracy results",
        "training loss curve over epochs",
    ]

    for qtext in queries:
        t0 = time.perf_counter()
        df = index.search(qtext, k=k)
        dt = (time.perf_counter() - t0) * 1000
        print(f"\n {qtext!r}   ({dt:.0f} ms)")
        print(df.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="v1", dest="config_name")
    p.add_argument("--query", default=None)
    p.add_argument("-k", type=int, default=5)
    main(**vars(p.parse_args()))

#now try visual search for docs
#python -m src.retrieval.visual_search --query "figure showing accuracy versus context length" -k 5