"""Gradio demo for the multimodal RAG system (local)."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation.answer import RAGPipeline, page_label   
from src.ingest.upload import ingest_upload                 
from src.utils.config import load_config                    
from src.utils.logging import get_logger                    

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


# --------------------------------------------------------------------------
# handlers
# --------------------------------------------------------------------------

def handle_upload(pdf_file, state, progress=gr.Progress()):
    if pdf_file is None:
        return state, "No file selected.", gr.update()

    if state and state.get("tmp"):
        shutil.rmtree(state["tmp"], ignore_errors=True)

    def cb(done, total):
        progress(done / total, desc=f"Embedding page {done}/{total}")

    idx, tmp = ingest_upload(
        pdf_file,
        CFG,
        PIPE.retriever.visual._model,
        PIPE.retriever.visual._processor,
        text_model=PIPE.retriever.text._model,
        on_progress=cb,
    )
    state = {"index": idx, "tmp": tmp}

    msg = (
        f"Indexed **{idx.name}** - {idx.n_pages} pages, "
        f"{int(idx.meta.has_figure.sum())} with figures"
        + (f", {len(idx.chunks)} text chunks" if idx.has_text else "")
        + ". Select *uploaded* as the source to query it."
    )
    return state, msg, gr.update(value="uploaded")


def run(question, mode, k, source, state):
    if not question.strip():
        return "Enter a question to get started.", "", [], None

    session_index = None

    if source == "uploaded":
        if not state or "index" not in state:
            return "Upload a PDF first.", "", [], None

        session_index = state["index"]
        q_emb = PIPE.retriever.visual.encode_query(question)
        hits = session_index.search(
            question,
            q_emb,
            k=int(k),
            mode=mode,
            text_model=PIPE.retriever.text._model,
            rrf_k=CFG.retrieval.rrf_k,
        )
        out = PIPE.generator.answer(question, hits, session_index=session_index)
        out["retrieve_ms"] = 0
    else:
        PIPE.mode = mode
        out = PIPE.run(question, k=int(k))
        hits = out["hits"]

    gallery = [
        (
            str(r.image_path if source == "uploaded"
                else CFG.project_root / r.image_path),
            f"#{r.rank} - page {r.page}" + (" [figure]" if r.fig else ""),
        )
        for r in hits.itertuples()
    ]

    meta = (
        f"{ROUTE_LABEL.get(out['route'], out['route'])}\n\n"
        f"**Source:** `{source}` - **Retriever:** `{mode}` - "
        f"**Pages:** {len(hits)} - "
        f"**Retrieve:** {out['retrieve_ms']} ms - "
        f"**Generate:** {out['ms']} ms"
    )

    cols = ["rank", "page", "fig", "hits"]
    table = hits[[c for c in cols if c in hits.columns]].rename(
        columns={"fig": "has figure", "hits": "found by"}
    )

    return out["answer"], meta, gallery, table


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------

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

    state = gr.State()

    source = gr.Radio(
        ["corpus", "uploaded"],
        value="corpus",
        label="Document source",
        info="corpus = the indexed OS slides; uploaded = your own PDF",
    )

    with gr.Accordion("Upload your own PDF (local only)", open=False):
        gr.Markdown(
            "Ingestion runs ColQwen2 on the local GPU at roughly 1.2 s/page, "
            "so a 20-page PDF takes about 25 seconds. Uploaded documents are "
            "held in memory for this session only and are never added to the "
            "indexed corpus.\n\n"
            "**Not available in a hosted demo** - per-upload GPU embedding "
            "exceeds free-tier quotas. Clone the repo to use this."
        )
        pdf_in = gr.File(label="PDF", file_types=[".pdf"])
        upload_btn = gr.Button("Ingest PDF")
        upload_status = gr.Markdown()

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
                info="visual scored best on the corpus: recall@10 0.888",
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
        **Retrieval evaluation** (30 labelled queries, indexed corpus)

        | Mode | recall@1 | recall@5 | recall@10 | nDCG@10 |
        |---|---|---|---|---|
        | **visual** | **0.561** | **0.837** | **0.888** | **0.758** |
        | dense | 0.439 | 0.714 | 0.755 | 0.641 |
        | bm25 | 0.480 | 0.663 | 0.704 | 0.624 |
        | fused | 0.520 | 0.806 | 0.827 | 0.721 |

        Switch the retriever to `bm25` and ask the same diagram question to see
        why visual retrieval matters.

        Uploaded documents build all three indexes, though pages with no
        extractable text fall back to visual retrieval only.
        """
    )

    upload_btn.click(
        handle_upload,
        [pdf_in, state],
        [state, upload_status, source],
    )

    ask.click(run, [question, mode, k, source, state],
              [answer, meta, gallery, table])
    question.submit(run, [question, mode, k, source, state],
                    [answer, meta, gallery, table])


if __name__ == "__main__":
    demo.launch(show_error=True)