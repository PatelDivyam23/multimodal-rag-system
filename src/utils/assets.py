"""Resolve index assets from local disk or from a HF Dataset repo."""
from __future__ import annotations

import os
from pathlib import Path

from src.utils.logging import get_logger

log = get_logger()


def ensure_assets(cfg) -> None:
    """If HF_INDEX_REPO is set, download index and images and repoint config."""
    repo = os.getenv("HF_INDEX_REPO")
    if not repo:
        return

    from huggingface_hub import snapshot_download

    log.info("Fetching assets from %s ...", repo)
    local = Path(snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        token=os.getenv("HF_TOKEN"),
    ))

    cfg.paths.index = str(local / "index" / cfg.index_version)
    cfg.paths.page_images = str(local / "page_images")
    cfg.project_root = local
    log.info("Assets ready at %s", local)


def resolve_image(cfg, image_path: str) -> Path:
    """Locate a page image locally or inside the downloaded dataset."""
    direct = Path(cfg.project_root) / image_path
    if direct.exists():
        return direct
    return Path(cfg.paths.page_images) / Path(image_path).name