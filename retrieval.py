from rank_bm25 import BM25Okapi
import pandas as pd

class BM25Retriever:
    def __init__(self, docs):
        self.docs = [d.split(" ") for d in docs]
        self.bm25 = BM25Okapi(self.docs)
    
    def search(self, query, topk=5):
        scores = self.bm25.get_scores(query.split(" "))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:topk]
        return ranked