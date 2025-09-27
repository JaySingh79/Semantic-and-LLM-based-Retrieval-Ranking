from sentence_transformers import SentenceTransformer, util

class SBERTReRanker:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def rerank(self, query, docs, topk=5):
        query_emb = self.model.encode(query, convert_to_tensor=True)
        doc_embs = self.model.encode(docs, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, doc_embs)[0]
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:topk]
        return ranked