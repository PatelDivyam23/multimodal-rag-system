"""Gradio demo for the multimodal RAG system (local)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation.answer import RAGPipeline, page_label   # noqa: E402
from src.utils.config import load_config                    # noqa: E402
from src.utils.logging import get_logger                    # noqa: E402

log = get_logger()

CFG = load_config(os.getenv("RAG_CONFIG", "v1"))
PIPE = RAGPipeline(CFG, mode="visual")

log.info("Warming up models ...")
PIPE.warmup()
log.info("Ready")


EXAMPLES = [
    "What are the states in the process state transition diagram?",
    "Show me the diagram of the CPU switching between two processes",
    "How does the operating system decide which process runs next?",
    "What is round robin scheduling?",
    "Explain the four conditions required for deadlock",
]

ROUTE_LABEL = {
    "vlm": "**Vision model** - answered from page images",
    "text": "**Text model** - answered from extracted text",
    "text-fallback": "**Text fallback** - vision path unavailable",
    "failed": "Generation failed",
    "none": "No pages retrieved",
}


def run(question: str, mode: str, k: int):
    if not question.strip():
        return "Enter a question to get started.", "", [], None

    PIPE.mode = mode
    out = PIPE.run(question, k=int(k))
    hits = out["hits"]

    gallery = [
        (str(CFG.project_root / r.image_path),
         f"#{r.rank} - {page_label(r)}" + (" [figure]" if r.fig else ""))
        for r in hits.itertuples()
    ]

    meta = (
        f"{ROUTE_LABEL.get(out['route'], out['route'])}\n\n"
        f"**Retriever:** `{mode}` - **Pages:** {len(hits)} - "
        f"**Retrieve:** {out['retrieve_ms']} ms - "
        f"**Generate:** {out['ms']} ms"
    )

    table = hits[["rank", "paper_id", "page", "fig"]].rename(
        columns={"paper_id": "document", "fig": "has figure"}
    )

    return out["answer"], meta, gallery, table


with gr.Blocks(title="Multimodal RAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Multimodal RAG over Visual Documents

        Searches document pages **as images** using ColQwen2 late-interaction
        retrieval, so diagrams, charts and tables become retrievable content.

        When a retrieved page contains a figure, the page image is sent to a
        vision model. Otherwise the extracted text goes to a text LLM.

        Corpus: 234 pages of OS lecture material, 176,670 patch vectors, 45 MB index.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            question = gr.Textbox(
                label="Question",
                placeholder="e.g. What are the states in the process state transition diagram?",
                lines=2,
            )
        with gr.Column(scale=1):
            mode = gr.Radio(
                ["visual", "fused", "dense", "bm25"],
                value="visual",
                label="Retriever",
                info="visual scored best: recall@10 0.888",
            )
            k = gr.Slider(1, 8, value=3, step=1, label="Pages retrieved")

    ask = gr.Button("Ask", variant="primary")
    gr.Examples(EXAMPLES, inputs=question)

    answer = gr.Markdown()
    meta = gr.Markdown()

    gr.Markdown("### Retrieved pages")
    gallery = gr.Gallery(columns=3, height=420, object_fit="contain",
                         show_label=False)

    with gr.Accordion("Retrieval details", open=False):
        table = gr.Dataframe(label="Ranking", wrap=True)

    gr.Markdown(
        """
        ---
        **Retrieval evaluation** (30 labelled queries)

        | Mode | recall@1 | recall@5 | recall@10 | nDCG@10 |
        |---|---|---|---|---|
        | **visual** | **0.561** | **0.837** | **0.888** | **0.758** |
        | dense | 0.439 | 0.714 | 0.755 | 0.641 |
        | bm25 | 0.480 | 0.663 | 0.704 | 0.624 |
        | fused | 0.520 | 0.806 | 0.827 | 0.721 |

        Switch the retriever to `bm25` and ask the same diagram question to see
        why visual retrieval matters.
        """
    )

    ask.click(run, [question, mode, k], [answer, meta, gallery, table])
    question.submit(run, [question, mode, k], [answer, meta, gallery, table])


if __name__ == "__main__":
    demo.launch(show_error=True)