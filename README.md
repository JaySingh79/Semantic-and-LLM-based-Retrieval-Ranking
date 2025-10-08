# Semantic and LLM-based Retrieval Ranking

This repository provides an analysis and implementation of retrieval and ranking approaches, focusing on:

- LLM-based rerankers (using Large Language Models for post-retrieval ranking)
- Graph-based retrievers and rankers
- Semantic similarity-based retrieval methods

## File Structure

- `app.py`, `app_vanillaJS.py` – Main application scripts (Python and JS integration)
- `retrieval.py`, `reranker.py` – Retrieval and reranking logic
- `tree_based_chunking.py`, `chunked_tree.json` – Tree-based data chunking for enhanced retrieval
- `pdf_parsing.py` – Utilities for parsing PDF documents
- `Reranking_prac.csv` – Example reranking practice dataset
- `requirements.txt` – Package dependencies
- `static/`, `Templates/` – Web application assets

## Features

- Compare traditional retrieval methods (BM25, SBERT) with LLM-based and graph approaches
- Explore semantic chunking strategies using tree structures (RAG-style)
- Prototype reranking algorithms and data workflows

## Getting Started

1. **Install dependencies:**
