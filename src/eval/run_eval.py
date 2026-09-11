"""Evaluate retrieval modes: recall@k, MRR, nDCG@10, split by query type."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd
import yaml

from src.retrieval.fusion import HybridRetriever
from src.utils.config import load_config
from src.utils.logging import get_logger

log = get_logger()

MODES = ["visual", "dense", "bm25", "fused"]


def page_key(paper_id: str, page_no: int) -> str:
    return f"{paper_id}:{page_no}"


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    hit = len(set(retrieved[:k]) & gold)
    return hit / len(gold) if gold else 0.0


def mrr(retrieved: list[str], gold: set[str]) -> float:
    for i, p in enumerate(retrieved, 1):
        if p in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: set[str], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(i + 1)
              for i, p in enumerate(retrieved[:k], 1) if p in gold)
    idcg = sum(1.0 / math.log2(i + 1)
               for i in range(1, min(len(gold), k) + 1))
    return dcg / idcg if idcg else 0.0


def main(config_name: str = "v1", k_max: int = 10, out: str | None = None):
    cfg = load_config(config_name)

    qfile = cfg.project_root / "src" / "eval" / "queries.yaml"
    queries = yaml.safe_load(qfile.read_text(encoding="utf-8"))
    log.info("Loaded %d queries from %s", len(queries), qfile.name)

    ret = HybridRetriever(cfg)
    ret.visual._ensure_model()
    ret.text._ensure_model()

    rows = []
    for qi, item in enumerate(queries, 1):
        gold = set(item["pages"])
        qtype = item.get("type", "unknown")

        for mode in MODES:
            t0 = time.perf_counter()
            df = ret.search(item["q"], k=k_max, mode=mode)
            dt = (time.perf_counter() - t0) * 1000

            retrieved = [page_key(r.paper_id, r.page - 1) for r in df.itertuples()]
            rows.append({
                "query": item["q"],
                "type": qtype,
                "mode": mode,
                "r@1": recall_at_k(retrieved, gold, 1),
                "r@5": recall_at_k(retrieved, gold, 5),
                "r@10": recall_at_k(retrieved, gold, 10),
                "mrr": mrr(retrieved, gold),
                "ndcg@10": ndcg_at_k(retrieved, gold, 10),
                "ms": round(dt),
            })
        log.info("[%d/%d] %s", qi, len(queries), item["q"][:55])

    res = pd.DataFrame(rows)

    # ---------- overall ----------
    overall = (res.groupby("mode")[["r@1", "r@5", "r@10", "mrr", "ndcg@10", "ms"]]
                  .mean().round(3).reindex(MODES))
    print("\n" + "═" * 72)
    print("OVERALL")
    print(overall.to_string())

    # ---------- by query type ----------
    print("\n" + "═" * 72)
    print("BY QUERY TYPE  (recall@5)")
    pivot = (res.pivot_table(index="type", columns="mode", values="r@5", aggfunc="mean")
                .round(3).reindex(columns=MODES))
    print(pivot.to_string())

    print("\n" + "═" * 72)
    print("BY QUERY TYPE  (nDCG@10)")
    pivot2 = (res.pivot_table(index="type", columns="mode", values="ndcg@10", aggfunc="mean")
                 .round(3).reindex(columns=MODES))
    print(pivot2.to_string())

    # ---------- worst failures ----------
    fused = res[res["mode"] == "fused"].nsmallest(5, "ndcg@10")
    print("\n" + "═" * 72)
    print("WORST FUSED QUERIES  🔍")
    print(fused[["query", "type", "r@5", "ndcg@10"]].to_string(index=False))

    # ---------- persist ----------
    out_dir = cfg.project_root / "src" / "eval" / "results"
    out_dir.mkdir(exist_ok=True)
    stem = out or f"eval_{cfg.index_version}"
    res.to_csv(out_dir / f"{stem}_raw.csv", index=False)
    overall.to_csv(out_dir / f"{stem}_summary.csv")
    (out_dir / f"{stem}_summary.json").write_text(
        json.dumps({"config": cfg.index_version,
                    "n_queries": len(queries),
                    "overall": overall.to_dict()}, indent=2))
    log.info("Saved -> %s", out_dir / f"{stem}_summary.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="v1", dest="config_name")
    p.add_argument("--k-max", type=int, default=10, dest="k_max")
    p.add_argument("--out", default=None)
    main(**vars(p.parse_args()))