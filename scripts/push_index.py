"""Upload index artifacts and page images to a HF Dataset repo."""
from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv
from huggingface_hub import HfApi

from src.utils.config import load_config
from src.utils.logging import get_logger

# load_dotenv()
load_dotenv(override=True)
log = get_logger()


def main(repo_id: str, config_name: str = "v1"):
    cfg = load_config(config_name)

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN missing from .env (needs write access)")

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=False)

    log.info("Uploading index/%s ...", cfg.index_version)
    api.upload_folder(
        folder_path=cfg.paths.index,
        path_in_repo=f"index/{cfg.index_version}",
        repo_id=repo_id,
        repo_type="dataset",
        ignore_patterns=["_visual_shards/*"],
    )

    log.info("Uploading page images ...")
    api.upload_folder(
        folder_path=cfg.paths.page_images,
        path_in_repo="page_images",
        repo_id=repo_id,
        repo_type="dataset",
    )

    log.info("Done: https://huggingface.co/datasets/%s", repo_id)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True, dest="repo_id")
    p.add_argument("--config", default="v1", dest="config_name")
    main(**vars(p.parse_args()))