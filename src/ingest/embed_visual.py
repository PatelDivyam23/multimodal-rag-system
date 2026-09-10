"""Embed page images with ColQwen2 (4-bit) -> ragged multi-vector store."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.logging import get_logger

log = get_logger()


def load_model(cfg):
    from colpali_engine.models import ColQwen2, ColQwen2Processor
    from transformers import BitsAndBytesConfig

    q = cfg.visual.quantization
    quant_cfg = None
    if q.enabled:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=q.load_in_4bit,
            bnb_4bit_quant_type=q.quant_type,
            bnb_4bit_compute_dtype=getattr(torch, q.compute_dtype),
            bnb_4bit_use_double_quant=q.double_quant,
        )

    log.info("Loading %s (4bit=%s) ...", cfg.visual.model_id, q.enabled)
    model = ColQwen2.from_pretrained(
        cfg.visual.model_id,
        torch_dtype=torch.float16,
        quantization_config=quant_cfg,
        device_map="cuda:0",
    ).eval()
    processor = ColQwen2Processor.from_pretrained(cfg.visual.model_id)

    mem = torch.cuda.memory_allocated() / 1e9
    log.info("Model loaded | VRAM allocated: %.2f GB", mem)
    return model, processor


# @torch.no_grad()
# def embed_page(model, processor, img_path: Path) -> np.ndarray:
#     """Return [n_patches, dim] float16 embeddings for one page."""
#     image = Image.open(img_path).convert("RGB")
#     batch = processor.process_images([image]).to(model.device)
#     emb = model(**batch)                      # [1, n_patches, dim]
#     return emb[0].to(torch.float16).cpu().numpy()


#this is for more batches
@torch.no_grad()
def embed_batch(model, processor, img_paths: list[Path]) -> list[np.ndarray]:
    """Return a list of [n_patches, dim] arrays — padding removed."""
    images = [Image.open(p).convert("RGB") for p in img_paths]
    batch = processor.process_images(images).to(model.device)
    emb = model(**batch)                              # [B, seq, dim]

    mask = batch["attention_mask"].bool().cpu()       # [B, seq]
    out = []
    for i in range(len(images)):
        valid = emb[i][mask[i]]                       # ← drop padding
        out.append(valid.to(torch.float16).cpu().numpy())
    return out


def main(config_name: str = "v1", limit: int | None = None) -> None:
    cfg = load_config(config_name)
    index_dir = Path(cfg.paths.index)
    shard_dir = index_dir / "_visual_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    pages = pd.read_parquet(index_dir / "pages.parquet")
    if limit:
        pages = pages.head(limit)
        log.info("LIMIT active: %d pages (smoke test)", limit)

    # ---- resume: which pages already embedded? ----
    done_keys: set[str] = set()
    for shard in sorted(shard_dir.glob("shard_*.npz")):
        done_keys.update(np.load(shard, allow_pickle=True)["keys"].tolist())
    if done_keys:
        log.info("Resuming: %d pages already embedded", len(done_keys))

    todo = pages[~pages.image_path.isin(done_keys)]
    if todo.empty:
        log.info("Nothing to embed.")
    else:
        model, processor = load_model(cfg)

        buf_vecs, buf_keys, buf_lens = [], [], []
        shard_id = len(list(shard_dir.glob("shard_*.npz")))
        ckpt = cfg.visual.checkpoint_every

        def flush():
            nonlocal shard_id, buf_vecs, buf_keys, buf_lens
            if not buf_vecs:
                return
            np.savez(
                shard_dir / f"shard_{shard_id:04d}.npz",
                vectors=np.concatenate(buf_vecs, axis=0),
                lengths=np.array(buf_lens, dtype=np.int32),
                keys=np.array(buf_keys, dtype=object),
            )
            shard_id += 1
            buf_vecs, buf_keys, buf_lens = [], [], []

        # pbar = tqdm(todo.itertuples(), total=len(todo), desc="embedding", unit="pg")
        # for n, row in enumerate(pbar, 1):
        #     img = cfg.project_root / row.image_path
        #     try:
        #         vec = embed_page(model, processor, img)
        #     except torch.cuda.OutOfMemoryError:
        #         log.error("OOM on %s -> flushing and aborting", row.image_path)
        #         flush()
        #         raise

        #     buf_vecs.append(vec)
        #     buf_lens.append(vec.shape[0])
        #     buf_keys.append(row.image_path)
        #     pbar.set_postfix(patches=vec.shape[0])

        #     if n % cfg.visual.empty_cache_every == 0:
        #         torch.cuda.empty_cache()
        #     if n % ckpt == 0:
        #         flush()
        #         log.info("checkpoint @ %d pages ", n)



        #for more batch size
        bs = cfg.visual.batch_size
        rows = list(todo.itertuples())
        pbar = tqdm(total=len(rows), desc="embedding", unit="pg")
        n = 0

        for start in range(0, len(rows), bs):
            group = rows[start : start + bs]
            paths = [cfg.project_root / r.image_path for r in group]
            try:
                vecs = embed_batch(model, processor, paths)
            except torch.cuda.OutOfMemoryError:
                log.error("OOM at page %d -> flushing and aborting", n)
                flush()
                raise

            for r, vec in zip(group, vecs):
                buf_vecs.append(vec)
                buf_lens.append(vec.shape[0])
                buf_keys.append(r.image_path)
                n += 1

            pbar.update(len(group))
            pbar.set_postfix(patches=vecs[0].shape[0])

            if n % cfg.visual.empty_cache_every < bs:
                torch.cuda.empty_cache()
            if n % ckpt < bs:
                flush()
                log.info("checkpoint @ %d pages ", n)

        pbar.close()

        flush()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ---- merge shards into final index ----
    vecs, keys, lens = [], [], []
    for shard in sorted(shard_dir.glob("shard_*.npz")):
        z = np.load(shard, allow_pickle=True)
        vecs.append(z["vectors"])
        lens.extend(z["lengths"].tolist())
        keys.extend(z["keys"].tolist())

    if not vecs:
        log.error("No shards produced.")
        return

    all_vecs = np.concatenate(vecs, axis=0).astype(np.float16)
    offsets = np.zeros(len(lens) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lens)

    np.save(index_dir / "visual_vectors.npy", all_vecs)
    np.save(index_dir / "visual_offsets.npy", offsets)
    pd.DataFrame({"image_path": keys}).to_parquet(
        index_dir / "visual_keys.parquet", index=False
    )

    manifest = {
        "index_version": cfg.index_version,
        "visual_model": cfg.visual.model_id,
        "quantized_4bit": cfg.visual.quantization.enabled,
        "n_pages": len(keys),
        "n_vectors": int(all_vecs.shape[0]),
        "dim": int(all_vecs.shape[1]),
        "mean_patches_per_page": round(float(np.mean(lens)), 1),
        "size_mb": round(all_vecs.nbytes / 1e6, 1),
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    log.info("─" * 55)
    for k, v in manifest.items():
        log.info("%-22s: %s", k, v)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="v1", dest="config_name")
    p.add_argument("--limit", type=int, default=None)
    main(**vars(p.parse_args()))

#dry run on some pages first(4 pages here)
#python -m src.ingest.embed_visual --limit 4

#then remove shards for 4 pages and full run
#Remove-Item -Recurse -Force index\v1\_visual_shards
#python -m src.ingest.embed_visual