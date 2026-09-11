"""Helper: run a query and print candidate pages for manual labelling."""
from __future__ import annotations

import argparse

from src.retrieval.fusion import HybridRetriever
from src.utils.config import load_config


def main(config_name: str = "v1", query: str = "", k: int = 10):
    cfg = load_config(config_name)
    ret = HybridRetriever(cfg)
    ret.visual._ensure_model()
    ret.text._ensure_model()

    df = ret.search(query, k=k, mode="fused")
    print(f"\n {query!r}\n")
    for r in df.itertuples():
        pid = r.paper_id
        page0 = r.page - 1                      # back to 0-indexed
        img = f"data/page_images/{pid}__page_{page0:04d}.png"
        fig = " " if r.fig else "  "
        print(f"  {r.rank:2d}. {fig} {pid}:{page0}")
        print(f"        {img}")

    print("\n  paste into queries.yaml as:")
    print(f'  - q: "{query}"')
    print('    pages: ["<paper_id>:<page_no>"]')
    print("    type: figure|text|keyword\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="v1", dest="config_name")
    p.add_argument("--query", required=True)
    p.add_argument("-k", type=int, default=10)
    main(**vars(p.parse_args()))