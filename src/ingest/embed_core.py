"""Reusable ColQwen2 loading and image embedding."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.utils.logging import get_logger

log = get_logger()


def load_visual_model(cfg):
    """Load ColQwen2 and its processor per config."""
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
    return model, processor


@torch.no_grad()
def embed_images(model, processor, paths: list[Path],
                 batch_size: int = 1, on_progress=None) -> list[np.ndarray]:
    """Embed page images. Returns one [n_patches, dim] array per image."""
    out: list[np.ndarray] = []

    for start in range(0, len(paths), batch_size):
        group = paths[start: start + batch_size]
        images = [Image.open(p).convert("RGB") for p in group]
        batch = processor.process_images(images).to(model.device)
        emb = model(**batch)
        mask = batch["attention_mask"].bool().cpu()

        for i in range(len(images)):
            out.append(emb[i][mask[i]].to(torch.float16).cpu().numpy())

        if on_progress:
            on_progress(len(out), len(paths))
        if len(out) % 10 == 0:
            torch.cuda.empty_cache()

    return out