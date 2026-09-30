"""
RAG (Retrieval-Augmented Generation) module for local document knowledge.
"""

from .engine import (
    BM25Retriever,
    Document,
    DocumentChunk,
    DocumentParser,
    RAGEngine,
    TextChunker,
    default_rag_engine,
)

__all__ = [
    "BM25Retriever",
    "Document",
    "DocumentChunk",
    "DocumentParser",
    "RAGEngine",
    "TextChunker",
    "default_rag_engine",
]
