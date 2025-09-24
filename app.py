# app.py
import gradio as gr
import pandas as pd
from retrieval import BM25Retriever
from reranker import SBERTReRanker

# load data + models (adjust paths if needed)
df = pd.read_csv("Reranking_prac.csv")
bm25 = BM25Retriever(df["content"].tolist())
sbert = SBERTReRanker()

def search_and_rerank(query):
    q = (query or "").strip()
    if not q:
        return "❌ Please enter a query.", "❌ Please enter a query."

    # Stage 1: BM25
    bm25_results = bm25.search(q, topk=5)  # list of (doc_index, score)
    bm25_docs = [df.iloc[idx]["content"] for idx, _ in bm25_results]

    bm25_md = ""
    for rank, (doc_idx, score) in enumerate(bm25_results, start=1):
        title = df.iloc[doc_idx]["title"]
        snippet = df.iloc[doc_idx]["content"][:300]
        bm25_md += f"### {rank}. {title}\n\n**BM25 score:** {score:.2f}\n\n{snippet}...\n\n---\n"

    # Stage 2: SBERT rerank (reranker returns list of (pos_in_bm25_docs, score))
    reranked = sbert.rerank(q, bm25_docs, topk=1)
    sbert_md = ""
    for rank, (pos_in_docs, score) in enumerate(reranked, start=1):
        original_doc_idx = bm25_results[pos_in_docs][0]  # map back to original df index
        title = df.iloc[original_doc_idx]["title"]
        snippet = df.iloc[original_doc_idx]["content"][:300]
        sbert_md += f"### {rank}. {title}\n\n**Cosine:** {score:.2f}\n\n{snippet}...\n\n---\n"

    return bm25_md, sbert_md


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1 style='text-align:center; margin-bottom: 10px;'>🔎 Mini Perplexity-Style Search</h1>")

    # top search row: left spacer | centered search box + button | right spacer
    with gr.Row():
        gr.Column(scale=1)  # left spacer
        with gr.Column(scale=8):
            with gr.Row():
                query_in = gr.Textbox(
                    placeholder="Ask anything (e.g. Who discovered relativity?)",
                    show_label=False,
                    lines=1,
                )
                search_btn = gr.Button("Search")
        gr.Column(scale=1)  # right spacer

    # results row: two equal columns
    with gr.Row():
        with gr.Column():
            gr.Markdown("##BM25 Retrieval")
            bm25_out = gr.Markdown()
        with gr.Column():
            gr.Markdown("##SBERT Rerank")
            sbert_out = gr.Markdown()

    # wire events: both Enter and button click trigger the same function
    search_btn.click(fn=search_and_rerank, inputs=query_in, outputs=[bm25_out, sbert_out])
    query_in.submit(fn=search_and_rerank, inputs=query_in, outputs=[bm25_out, sbert_out])

if __name__ == "__main__":
    demo.launch(debug=True)
