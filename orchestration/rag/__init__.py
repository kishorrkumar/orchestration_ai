"""
RAG (Retrieval-Augmented Generation) module for local document knowledge.
"""

from .engine import (
    Document,
    DocumentChunk,
    DocumentParser,
    TextChunker,
    BM25Retriever,
    RAGEngine,
    default_rag_engine,
)

__all__ = [
    "Document",
    "DocumentChunk",
    "DocumentParser",
    "TextChunker",
    "BM25Retriever",
    "RAGEngine",
    "default_rag_engine",
]
