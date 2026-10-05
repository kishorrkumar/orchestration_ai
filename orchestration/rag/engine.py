"""Backward-compatible facade for dormant RAG engine."""
from ..dormant.rag.engine import (
    BM25Retriever,
    DocumentChunk,
    DocumentParser,
    RAGEngine,
    TextChunker,
    default_rag_engine,
)

__all__ = [
    "BM25Retriever",
    "DocumentChunk",
    "DocumentParser",
    "RAGEngine",
    "TextChunker",
    "default_rag_engine",
]
