"""Ingest a user-uploaded PDF into an ephemeral session index."""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pandas as pd
import pymupdf

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


def ingest_upload(pdf_file: str, cfg, model, processor,
                  on_progress=None) -> tuple[SessionIndex, Path]:
    """Render + embed an uploaded PDF. Caller owns the returned temp dir."""
    tmp = Path(tempfile.mkdtemp(prefix="rag_upload_"))
    meta, name = render_upload(Path(pdf_file), tmp, cfg)
    log.info("Uploaded '%s': %d pages", name, len(meta))

    paths = [Path(p) for p in meta.image_path]
    vectors = embed_images(model, processor, paths,
                           batch_size=cfg.visual.batch_size,
                           on_progress=on_progress)

    return SessionIndex(vectors, meta, name), tmp