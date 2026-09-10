from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def load_config(name: str = "v1") -> SimpleNamespace:
    """Load configs/<name>.yaml with paths resolved to absolute."""
    path = PROJECT_ROOT / "configs" / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    for key, val in raw.get("paths", {}).items():
        raw["paths"][key] = str(PROJECT_ROOT / val)

    cfg = _to_ns(raw)
    cfg.project_root = PROJECT_ROOT
    return cfg


if __name__ == "__main__":
    cfg = load_config()
    print("index:", cfg.index_version)
    print("model:", cfg.visual.model_id)
    print("4bit :", cfg.visual.quantization.load_in_4bit)
    print("pdfs :", cfg.paths.pdfs)

#test it
# python -m src.utils.config 