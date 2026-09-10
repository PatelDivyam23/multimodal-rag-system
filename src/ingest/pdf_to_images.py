"""Render PDFs to page images + extract per-page text and figure flags."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import pymupdf
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.logging import get_logger

log = get_logger()


def make_paper_id(pdf_path: Path) -> str:
    """Filesystem-safe, stable id derived from the filename."""
    stem = pdf_path.stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return stem[:60] or "paper"


def page_has_figure(page: pymupdf.Page) -> bool:
    """Heuristic: raster images, or many vector drawings (charts/diagrams)."""
    if len(page.get_images(full=True)) > 0:
        return True
    return len(page.get_drawings()) > 40


def render_pdf(pdf_path: Path, out_dir: Path, cfg) -> list[dict]:
    paper_id = make_paper_id(pdf_path)
    doc = pymupdf.open(pdf_path)
    records: list[dict] = []

    zoom = cfg.render.dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)

    for i, page in enumerate(doc):
        img_name = f"{paper_id}__page_{i:04d}.{cfg.render.format}"
        img_path = out_dir / img_name

        if not img_path.exists():                       # resumable
            pix = page.get_pixmap(matrix=matrix, alpha=False)

            long_edge = max(pix.width, pix.height)
            if long_edge > cfg.render.max_long_edge:
                scale = cfg.render.max_long_edge / long_edge
                pix = page.get_pixmap(
                    matrix=pymupdf.Matrix(zoom * scale, zoom * scale), alpha=False
                )
            pix.save(img_path)

        text = page.get_text("text").strip()
        records.append(
            {
                "paper_id": paper_id,
                "source_pdf": pdf_path.name,
                "page_no": i,
                "n_pages": doc.page_count,
                "image_path": str(img_path.relative_to(cfg.project_root)),
                "has_figure": page_has_figure(page),
                "n_chars": len(text),
                "text": text,
            }
        )

    doc.close()
    return records


def main(config_name: str = "v1") -> None:
    cfg = load_config(config_name)

    pdf_dir = Path(cfg.paths.pdfs)
    img_dir = Path(cfg.paths.page_images)
    index_dir = Path(cfg.paths.index)
    img_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        log.error("No PDFs found in %s", pdf_dir)
        return

    log.info("Found %d PDFs | dpi=%d | max_edge=%d",
             len(pdfs), cfg.render.dpi, cfg.render.max_long_edge)

    all_records: list[dict] = []
    for pdf in tqdm(pdfs, desc="rendering", unit="pdf"):
        try:
            all_records.extend(render_pdf(pdf, img_dir, cfg))
        except Exception as exc:                        # keep going on bad PDFs
            log.warning("FAILED %s -> %s", pdf.name, exc)

    df = pd.DataFrame(all_records)
    out = index_dir / "pages.parquet"
    df.to_parquet(out, index=False)

    log.info("─" * 55)
    log.info("papers        : %d", df.paper_id.nunique())
    log.info("pages         : %d", len(df))
    log.info("with figures  : %d (%.0f%%)",
             df.has_figure.sum(), 100 * df.has_figure.mean())
    log.info("empty text    : %d  <- scanned/OCR-needed if high",
             (df.n_chars < 50).sum())
    log.info("manifest      : %s", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="v1", dest="config_name")
    main(**vars(p.parse_args()))

#put a few pdf (data/pdfs/)and run
#python -m src.ingest.pdf_to_images