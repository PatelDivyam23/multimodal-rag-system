"""Ingest a user-uploaded PDF into an ephemeral session index."""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pandas as pd
import pymupdf
import numpy as np

from src.ingest.embed_core import embed_images
from src.ingest.pdf_to_images import page_has_figure
from src.retrieval.session_index import SessionIndex
from src.utils.logging import get_logger

log = get_logger()


def render_upload(pdf_path: Path, out_dir: Path, cfg) -> pd.DataFrame:
    """Render an uploaded PDF to images in a temp directory."""
    name = re.sub(r"[^a-z0-9]+", "_", pdf_path.stem.lower()).strip("_")[:40] or "upload"
    doc = pymupdf.open(pdf_path)
    zoom = cfg.render.dpi / 72.0
    records = []

    for i, page in enumerate(doc):
        img_path = out_dir / f"{name}__page_{i:04d}.png"
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)

        long_edge = max(pix.width, pix.height)
        if long_edge > cfg.render.max_long_edge:
            s = cfg.render.max_long_edge / long_edge
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom * s, zoom * s), alpha=False)
        pix.save(img_path)

        records.append({
            "page_no": i,
            "image_path": str(img_path),
            "has_figure": page_has_figure(page),
            "text": page.get_text("text").strip(),
        })

    doc.close()
    return pd.DataFrame(records), name


# def ingest_upload(pdf_file: str, cfg, model, processor,
#                   on_progress=None) -> tuple[SessionIndex, Path]:
#     """Render + embed an uploaded PDF. Caller owns the returned temp dir."""
#     tmp = Path(tempfile.mkdtemp(prefix="rag_upload_"))
#     meta, name = render_upload(Path(pdf_file), tmp, cfg)
#     log.info("Uploaded '%s': %d pages", name, len(meta))

#     paths = [Path(p) for p in meta.image_path]
#     vectors = embed_images(model, processor, paths,
#                            batch_size=cfg.visual.batch_size,
#                            on_progress=on_progress)

#     return SessionIndex(vectors, meta, name), tmp

# def ingest_upload(pdf_file, cfg, model, processor,
#                   text_model=None, on_progress=None):
#     """Render, embed and index an uploaded PDF. Caller owns the temp dir."""
#     from rank_bm25 import BM25Okapi

#     from src.ingest.embed_text import chunk_text, tokenize

#     tmp = Path(tempfile.mkdtemp(prefix="rag_upload_"))
#     meta, name = render_upload(Path(pdf_file), tmp, cfg)
#     log.info("Uploaded '%s': %d pages", name, len(meta))

#     paths = [Path(p) for p in meta.image_path]
#     vectors = embed_images(model, processor, paths,
#                            batch_size=cfg.visual.batch_size,
#                            on_progress=on_progress)

#     text_vecs = bm25 = chunks = None
#     if text_model is not None:
#         records = []
#         for row in meta.itertuples():
#             for ch in chunk_text(row.text, cfg.text.chunk_size,
#                                  cfg.text.chunk_overlap):
#                 records.append({"image_path": row.image_path, "text": ch})

#         if records:
#             chunks = pd.DataFrame(records)
#             text_vecs = text_model.encode(
#                 chunks.text.tolist(),
#                 batch_size=cfg.text.batch_size,
#                 normalize_embeddings=True,
#                 convert_to_numpy=True,
#             ).astype(np.float32)
#             bm25 = BM25Okapi([tokenize(t) for t in chunks.text])
#             log.info("Built %d text chunks for upload", len(chunks))

#     idx = SessionIndex(vectors, meta, name,
#                        text_vecs=text_vecs, bm25=bm25, chunks=chunks)
#     return idx, tmp

#for multiple files

def ingest_upload(pdf_files, cfg, model, processor,
                  text_model=None, on_progress=None):
    """Render, embed and index one or more uploaded PDFs."""
    from rank_bm25 import BM25Okapi

    from src.ingest.embed_text import chunk_text, tokenize

    if isinstance(pdf_files, (str, Path)):
        pdf_files = [pdf_files]

    tmp = Path(tempfile.mkdtemp(prefix="rag_upload_"))

    frames, names = [], []
    for pdf in pdf_files:
        meta, name = render_upload(Path(pdf), tmp, cfg)
        meta["doc_name"] = name
        frames.append(meta)
        names.append(name)

    meta = pd.concat(frames, ignore_index=True)
    log.info("Uploaded %d document(s): %d pages", len(names), len(meta))

    paths = [Path(p) for p in meta.image_path]
    vectors = embed_images(model, processor, paths,
                           batch_size=cfg.visual.batch_size,
                           on_progress=on_progress)

    text_vecs = bm25 = chunks = None
    if text_model is not None:
        records = []
        for row in meta.itertuples():
            for ch in chunk_text(row.text, cfg.text.chunk_size,
                                 cfg.text.chunk_overlap):
                records.append({"image_path": row.image_path, "text": ch})

        if records:
            chunks = pd.DataFrame(records)
            text_vecs = text_model.encode(
                chunks.text.tolist(),
                batch_size=cfg.text.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype(np.float32)
            bm25 = BM25Okapi([tokenize(t) for t in chunks.text])
            log.info("Built %d text chunks for upload", len(chunks))

    label = names[0] if len(names) == 1 else f"{len(names)} documents"
    idx = SessionIndex(vectors, meta, label,
                       text_vecs=text_vecs, bm25=bm25, chunks=chunks)
    idx.n_docs = len(names)
    return idx, tmp