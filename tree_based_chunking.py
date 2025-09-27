import re
import uuid
import json
from typing import Dict, List
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = []
    for page in reader.pages:
        text.append(page.extract_text())
    return "\n".join(text)


def build_tree(text: str) -> Dict[str, Dict]:
    """
    Builds a hierarchical tree:
    Root -> Section -> Subsections -> Paragraphs
    """
    tree = {}
    root_id = str(uuid.uuid4())
    tree[root_id] = {
        "title": "Resources for CDC",
        "text": "",
        "parent": None,
        "children": [],
        "level": 0
    }

    # Identify major sections (all-caps headings or big words)
    section_pattern = r"\n([A-Z][A-Z\s/]+)\n"
    sections = re.split(section_pattern, text)

    # First element before the first heading = intro
    intro_text = sections[0].strip()
    if intro_text:
        intro_id = str(uuid.uuid4())
        tree[intro_id] = {
            "title": "Introduction",
            "text": intro_text,
            "parent": root_id,
            "children": [],
            "level": 1
        }
        tree[root_id]["children"].append(intro_id)

    # Loop through sections
    for i in range(1, len(sections), 2):
        section_title = sections[i].strip()
        section_body = sections[i+1].strip()

        section_id = str(uuid.uuid4())
        tree[section_id] = {
            "title": section_title,
            "text": "",
            "parent": root_id,
            "children": [],
            "level": 1
        }
        tree[root_id]["children"].append(section_id)

        # Split section body into subsections (heuristic: numbered or bulleted parts)
        subsections = re.split(r"\n\d+\.|\n[a-z]\.", section_body)
        for subsection in subsections:
            subsection = subsection.strip()
            if not subsection:
                continue

            # Paragraph-level splitting
            paragraphs = [p.strip() for p in subsection.split("\n") if p.strip()]
            if not paragraphs:
                continue

            sub_id = str(uuid.uuid4())
            tree[sub_id] = {
                "title": paragraphs[0][:50],  # first line as title
                "text": "\n".join(paragraphs),
                "parent": section_id,
                "children": [],
                "level": 2
            }
            tree[section_id]["children"].append(sub_id)

            # Paragraph-level children
            for p in paragraphs:
                para_id = str(uuid.uuid4())
                tree[para_id] = {
                    "title": p[:40],
                    "text": p,
                    "parent": sub_id,
                    "children": [],
                    "level": 3
                }
                tree[sub_id]["children"].append(para_id)

    return tree


embedder = SentenceTransformer("all-MiniLM-L6-v2")

class TreeRAGIndex:
    def __init__(self, embedder, dim=384):  # 384 dims for MiniLM
        self.embedder = embedder
        self.index = faiss.IndexFlatIP(dim)  # inner product similarity
        self.id_map = {}
        self.meta = {}

    def add_tree(self, tree: Dict[str, Dict]):
        texts = [node["prefix"] for node in tree.values()]
        ids = list(tree.keys())
        vecs = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        self.index.add(vecs)
        for i, node_id in enumerate(ids):
            self.id_map[len(self.id_map)] = node_id
            self.meta[node_id] = tree[node_id]

    def search(self, query: str, top_k: int = 5):
        qvec = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, idxs = self.index.search(qvec, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            node_id = self.id_map[idx]
            results.append((score, self.meta[node_id]))
        return results


# ====== USAGE ======
if __name__ == "__main__":
    pdf_path = "ffa41df7-771b-469e-94e8-8498480687dd.pdf"
    text = extract_text_from_pdf(pdf_path)
    tree = build_tree(text)

    rag_index = TreeRAGIndex(embedder)
    rag_index.add_tree(tree)

    query = "What resources should I use to prepare for consulting interviews?"
    results = rag_index.search(query, top_k=5)

    print("\n=== Retrieval Results ===")
    for score, meta in results:
        print(f"[{meta['title'][:40]}...] (score={score:.2f})")
        print(meta["text"][:200], "...\n")

# if __name__ == "__main__":
#     pdf_path = "D:/Resources/Resources for CDC.pdf"
#     text = extract_text_from_pdf(pdf_path)
#     tree = build_tree(text)

#     with open("chunked_tree.json", "w", encoding="utf-8") as f:
#         json.dump(tree, f, indent=2, ensure_ascii=False)

#     print("Tree JSON saved as chunked_tree.json")