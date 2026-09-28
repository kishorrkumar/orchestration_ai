"""
Tests for RAG (Retrieval-Augmented Generation) Engine.
"""

import pytest
from orchestration.rag.engine import (
    DocumentParser,
    TextChunker,
    BM25Retriever,
    RAGEngine,
    DocumentChunk,
)


def test_document_parser_text():
    text = "Hello world. This is a test file for the RAG system."
    extracted, pages = DocumentParser.parse_bytes("test.txt", text.encode("utf-8"))
    assert "Hello world" in extracted
    assert pages == 1


def test_text_chunker():
    chunker = TextChunker(chunk_size=10, overlap=3)
    long_text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen"
    chunks = chunker.chunk(long_text, doc_id="doc1", doc_name="test.txt")
    assert len(chunks) >= 2
    assert chunks[0].doc_id == "doc1"
    assert chunks[0].doc_name == "test.txt"


def test_bm25_retriever():
    retriever = BM25Retriever()
    chunk1 = DocumentChunk(
        chunk_id="c1",
        doc_id="d1",
        doc_name="physics.txt",
        text="Quantum mechanics is a fundamental theory in physics that describes the physical properties of nature.",
        chunk_index=0,
    )
    chunk2 = DocumentChunk(
        chunk_id="c2",
        doc_id="d2",
        doc_name="baking.txt",
        text="To bake sourdough bread you need active starter, flour, water, and sea salt. Ferment for eight hours.",
        chunk_index=0,
    )

    retriever.build_index([chunk1, chunk2])

    results = retriever.retrieve("physics quantum properties", top_k=2)
    assert len(results) >= 1
    top_chunk, score = results[0]
    assert top_chunk.chunk_id == "c1"
    assert score > 0.1

    bread_results = retriever.retrieve("sourdough starter recipe", top_k=1)
    assert len(bread_results) == 1
    assert bread_results[0][0].chunk_id == "c2"


def test_rag_engine_lifecycle():
    engine = RAGEngine()
    doc1 = engine.add_text(
        title="company_policy.md",
        text="Annual vacation policy allows up to twenty paid days off per calendar year for full time employees. Contact HR to schedule.",
    )
    assert doc1.doc_id in [d["doc_id"] for d in engine.list_documents()]

    # Query
    results = engine.query("vacation days policy")
    assert len(results) >= 1
    assert "twenty paid days" in results[0][0].text

    # Grounded response
    spoken_answer = engine.generate_grounded_response("How many vacation days do I get?")
    assert spoken_answer is not None
    assert "company_policy.md" in spoken_answer

    # Delete
    deleted = engine.delete_document(doc1.doc_id)
    assert deleted is True
    assert len(engine.list_documents()) == 0
