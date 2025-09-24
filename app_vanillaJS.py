# app.py
from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import html
from retrieval import BM25Retriever
from reranker import SBERTReRanker

app = Flask(__name__, static_folder="static", template_folder="templates")

# Load data and models (adjust file paths as needed)
df = pd.read_csv("Reranking_prac.csv")
bm25 = BM25Retriever(df["content"].tolist())
sbert = SBERTReRanker()


def build_snippet(text, q, length=300):
    # Simple snippet: show the first occurrence of any query term if possible
    q_terms = [t.lower() for t in q.split() if t.strip()]
    lowered = text.lower()
    pos = -1
    for term in q_terms:
        pos = lowered.find(term)
        if pos >= 0:
            break
    if pos == -1:
        snippet = text[:length]
    else:
        start = max(0, pos - 60)
        snippet = text[start:start + length]
        if start > 0:
            snippet = "..." + snippet
    return snippet


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():
    payload = request.get_json() or {}
    query = (payload.get("query") or "").strip()
    topk = int(payload.get("topk", 5))
    if not query:
        return jsonify({"error": "Please provide a query."}), 400

    # Stage 1: BM25 retrieval
    bm25_results = bm25.search(query, topk=topk)  # returns list of (doc_index, score)
    bm25_list = []
    for rank, (doc_idx, score) in enumerate(bm25_results, start=1):
        title = df.iloc[doc_idx].get("title", f"Doc {doc_idx}")
        content = str(df.iloc[doc_idx]["content"])
        snippet = build_snippet(content, query, length=300)
        bm25_list.append({
            "rank": rank,
            "doc_idx": int(doc_idx),
            "title": html.escape(str(title)),
            "score": float(score),
            "snippet": html.escape(snippet)
        })

    # Stage 2: SBERT re-rank (reranker expects the bm25_docs; returns list of (pos_in_bm25_docs, score))
    bm25_docs = [df.iloc[idx]["content"] for idx, _ in bm25_results]
    try:
        reranked = sbert.rerank(query, bm25_docs, topk=topk)
    except Exception as e:
        # If reranker fails, return BM25 only and an error message for sbert
        return jsonify({
            "bm25": bm25_list,
            "sbert_error": str(e)
        })

    sbert_list = []
    for rank, (pos_in_docs, score) in enumerate(reranked, start=1):
        original_doc_idx = int(bm25_results[pos_in_docs][0])
        title = df.iloc[original_doc_idx].get("title", f"Doc {original_doc_idx}")
        content = str(df.iloc[original_doc_idx]["content"])
        snippet = build_snippet(content, query, length=300)
        sbert_list.append({
            "rank": rank,
            "doc_idx": original_doc_idx,
            "title": html.escape(str(title)),
            "score": float(score),
            "snippet": html.escape(snippet)
        })

    return jsonify({
        "bm25": bm25_list,
        "sbert": sbert_list
    })


# Optional static file serving if needed
@app.route("/static/<path:pth>")
def static_proxy(pth):
    return send_from_directory("static", pth)


if __name__ == "__main__":
    # debug True for development; set to False in production
    app.run(host="0.0.0.0", port=7860, debug=True)