"""Chunk page text -> BGE-small dense embeddings + BM25 lexical index."""
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.logging import get_logger

log = get_logger()


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Paragraph-aware splitter with character overlap."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""

    for p in paras:
        if len(cur) + len(p) + 2 <= size:
            cur = f"{cur}\n\n{p}" if cur else p
        else:
            if cur:
                chunks.append(cur)
            # long single paragraph -> hard split
            while len(p) > size:
                chunks.append(p[:size])
                p = p[size - overlap:]
            cur = p
    if cur:
        chunks.append(cur)

    # add overlap between adjacent chunks
    if overlap and len(chunks) > 1:
        out = [chunks[0]]
        for prev, nxt in zip(chunks, chunks[1:]):
            out.append(prev[-overlap:] + "\n" + nxt)
        chunks = out

    return [c for c in chunks if len(c.strip()) > 40]


def tokenize(text: str) -> list[str]:
    """BM25 tokenizer: keep alphanumerics + hyphens (part numbers, citations)."""
    return re.findall(r"[a-z0-9][a-z0-9\-_]*", text.lower())


def main(config_name: str = "v1") -> None:
    cfg = load_config(config_name)
    index_dir = Path(cfg.paths.index)

    pages = pd.read_parquet(index_dir / "pages.parquet")
    log.info("Chunking %d pages | size=%d overlap=%d",
             len(pages), cfg.text.chunk_size, cfg.text.chunk_overlap)

    # ---- 1. chunk ----
    records = []
    for row in pages.itertuples():
        for j, ch in enumerate(chunk_text(row.text, cfg.text.chunk_size,
                                          cfg.text.chunk_overlap)):
            records.append({
                "chunk_id": f"{row.image_path}#c{j}",
                "image_path": row.image_path,
                "paper_id": row.paper_id,
                "page_no": row.page_no,
                "has_figure": row.has_figure,
                "text": ch,
            })

    chunks = pd.DataFrame(records)
    if chunks.empty:
        log.error("No chunks produced — is page text empty?")
        return

    log.info("chunks: %d | mean chars: %.0f | pages covered: %d/%d",
             len(chunks), chunks.text.str.len().mean(),
             chunks.image_path.nunique(), len(pages))

    # ---- 2. dense embeddings ----
    from sentence_transformers import SentenceTransformer

    device = cfg.text.device if torch.cuda.is_available() else "cpu"
    log.info("Loading %s on %s ...", cfg.text.embed_model, device)
    model = SentenceTransformer(cfg.text.embed_model, device=device)

    emb = model.encode(
        chunks.text.tolist(),
        batch_size=cfg.text.batch_size,
        normalize_embeddings=True,          # cosine == dot product
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    np.save(index_dir / "text_vectors.npy", emb)
    log.info("dense: %s | %.1f MB", emb.shape, emb.nbytes / 1e6)

    del model
    torch.cuda.empty_cache()

    # ---- 3. BM25 ----
    from rank_bm25 import BM25Okapi

    log.info("Building BM25 ...")
    corpus = [tokenize(t) for t in tqdm(chunks.text, desc="tokenizing")]
    bm25 = BM25Okapi(corpus)
    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    # ---- 4. persist chunks + manifest ----
    chunks.to_parquet(index_dir / "chunks.parquet", index=False)

    mpath = index_dir / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    manifest.update({
        "text_model": cfg.text.embed_model,
        "n_chunks": len(chunks),
        "text_dim": int(emb.shape[1]),
        "chunk_size": cfg.text.chunk_size,
        "chunk_overlap": cfg.text.chunk_overlap,
    })
    mpath.write_text(json.dumps(manifest, indent=2))

    log.info("─" * 55)
    log.info("chunks.parquet    : %d rows", len(chunks))
    log.info("text_vectors.npy  : %s", emb.shape)
    log.info("bm25.pkl          : %d docs", len(corpus))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="v1", dest="config_name")
    main(**vars(p.parse_args()))

#run this after
#python -m src.ingest.embed_text