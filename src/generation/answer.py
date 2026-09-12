"""Modality-routed answer generation: Gemini VLM for figures, Groq for text."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.generation.prompts import SYSTEM, TEXT_USER, VLM_USER
from src.retrieval.fusion import HybridRetriever
from src.utils.config import load_config
from src.utils.logging import get_logger

load_dotenv()
log = get_logger()


def page_label(row) -> str:
    """Short citation key: strip numeric prefix from paper_id."""
    pid = row.paper_id
    parts = pid.split("_", 1)
    short = parts[1] if len(parts) > 1 and parts[0].isdigit() else pid
    return f"{short}:{row.page}"


class Generator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.chunks = pd.read_parquet(Path(cfg.paths.index) / "chunks.parquet")
        self._gemini = None
        self._groq = None

    # ---------- clients ----------
    def _gemini_client(self):
        if self._gemini is None:
            from google import genai
            key = os.getenv("GEMINI_API_KEY")
            if not key:
                raise RuntimeError("GEMINI_API_KEY missing from .env")
            self._gemini = genai.Client(api_key=key)
        return self._gemini

    def _groq_client(self):
        if self._groq is None:
            from groq import Groq
            key = os.getenv("GROQ_API_KEY")
            if not key:
                raise RuntimeError("GROQ_API_KEY missing from .env")
            self._groq = Groq(api_key=key)
        return self._groq

    # ---------- routing ----------
    def route(self, hits: pd.DataFrame) -> str:
        if not self.cfg.generation.route_on_figures:
            return "text"
        return "vlm" if bool(hits.fig.any()) else "text"

    # ---------- VLM path ----------
    def answer_vlm(self, question: str, hits: pd.DataFrame) -> str:
        from PIL import Image

        client = self._gemini_client()
        labels = [page_label(r) for r in hits.itertuples()]
        images = [Image.open(self.cfg.project_root / r.image_path).convert("RGB")
                  for r in hits.itertuples()]

        prompt = VLM_USER.format(
            question=question,
            n=len(images),
            page_list="\n".join(f"  {i+1}. [{l}]" for i, l in enumerate(labels)),
        )

        resp = client.models.generate_content(
            model=self.cfg.generation.vlm.model,
            contents=[prompt, *images],
            config={"system_instruction": SYSTEM, "temperature": 0.2},
        )
        return resp.text

    # ---------- text path ----------
    def answer_text(self, question: str, hits: pd.DataFrame) -> str:
        client = self._groq_client()

        blocks = []
        for r in hits.itertuples():
            txt = "\n".join(
                self.chunks[self.chunks.image_path == r.image_path].text.tolist()
            )
            if txt.strip():
                blocks.append(f"[{page_label(r)}]\n{txt.strip()}")

        if not blocks:
            return "The retrieved pages contain no extractable text."

        resp = client.chat.completions.create(
            model=self.cfg.generation.text_llm.model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": TEXT_USER.format(
                    question=question, context="\n\n---\n\n".join(blocks))},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content

    # ---------- entry point ----------
    def answer(self, question: str, hits: pd.DataFrame) -> dict:
        if hits.empty:
            return {"answer": "No relevant pages found.", "route": "none", "hits": hits}

        hits = hits.head(self.cfg.generation.max_context_pages)
        route = self.route(hits)

        t0 = time.perf_counter()
        # try:
        #     text = self.answer_vlm(question, hits) if route == "vlm" \
        #            else self.answer_text(question, hits)
        # except Exception as exc:
        #     log.warning("%s path failed (%s) — falling back to text", route, exc)
        #     route = "text-fallback"
        #     text = self.answer_text(question, hits)
        try:
            text = self.answer_vlm(question, hits) if route == "vlm" \
                   else self.answer_text(question, hits)
        except Exception as exc:
            log.warning("%s path failed (%s) — falling back to text", route, exc)
            route = "text-fallback"
            try:
                text = self.answer_text(question, hits)
            except Exception as exc2:
                route = "failed"
                text = f"Generation failed: {exc2}"

        return {
            "answer": text,
            "route": route,
            "ms": round((time.perf_counter() - t0) * 1000),
            "hits": hits,
            "sources": [page_label(r) for r in hits.itertuples()],
        }


class RAGPipeline:
    """Retrieve -> route -> generate."""

    def __init__(self, cfg, mode: str = "visual"):
        self.cfg = cfg
        self.mode = mode
        self.retriever = HybridRetriever(cfg)
        self.generator = Generator(cfg)

    def warmup(self):
        self.retriever.visual._ensure_model()
        self.retriever.text._ensure_model()

    def run(self, question: str, k: int | None = None) -> dict:
        k = k or self.cfg.generation.max_context_pages
        t0 = time.perf_counter()
        hits = self.retriever.search(question, k=k, mode=self.mode)
        retrieve_ms = round((time.perf_counter() - t0) * 1000)

        out = self.generator.answer(question, hits)
        out["retrieve_ms"] = retrieve_ms
        return out


def main(config_name: str = "v1", question: str = "", mode: str = "visual", k: int = 5):
    cfg = load_config(config_name)
    pipe = RAGPipeline(cfg, mode=mode)
    pipe.warmup()

    out = pipe.run(question, k=k)

    print("\n" + "═" * 72)
    print(f" {question}")
    print("═" * 72)
    print(f"\n{out['answer']}\n")
    print("─" * 72)
    print(f" sources : {', '.join(out['sources'])}")
    print(f" route   : {out['route']}")
    print(f"  retrieve {out['retrieve_ms']} ms | generate {out['ms']} ms")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="v1", dest="config_name")
    p.add_argument("--question", required=True)
    p.add_argument("--mode", default="visual",
                   choices=["visual", "dense", "bm25", "fused"])
    p.add_argument("-k", type=int, default=5)
    main(**vars(p.parse_args()))

#confirm first key is loaded->
#  python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('gemini:', bool(os.getenv('GEMINI_API_KEY'))); print('groq:', bool(os.getenv('GROQ_API_KEY')))"
#python -m src.generation.answer --question "What are the states in the process state transition diagram?"