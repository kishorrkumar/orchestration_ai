"""
Local RAG (Retrieval-Augmented Generation) Engine.

Enables uploading text/markdown/PDF files to ground voice agents with document knowledge.
Built with SOLID principles, 100% open-source, offline, and lightweight.
"""

from __future__ import annotations

import io
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    doc_name: str
    text: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


@dataclass
class Document:
    doc_id: str
    filename: str
    file_type: str
    raw_text: str
    chunks: list[DocumentChunk]
    created_at: float = field(default_factory=time.time)
    num_pages: int = 1


class DocumentParser:
    """Extracts raw text from multiple document formats."""

    @staticmethod
    def parse_bytes(filename: str, content: bytes) -> tuple[str, int]:
        """
        Parses byte content of a file and returns (extracted_text, num_pages).
        Supports: .txt, .md, .csv, .json, .log, .pdf
        """
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            return DocumentParser._parse_pdf(content)
        else:
            # Plain text formats
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                text = content.decode("latin-1", errors="replace")
            return text.strip(), 1

    @staticmethod
    def _parse_pdf(content: bytes) -> tuple[str, int]:
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content))
            pages_text = []
            num_pages = len(reader.pages)
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages_text.append(f"[Page {page_idx + 1}]\n{text.strip()}")
            combined = "\n\n".join(pages_text)
            return (combined if combined else "No extractable text found in PDF.", max(1, num_pages))
        except ImportError:
            # Fallback if pypdf is unavailable: basic ASCII/text scan
            printable = "".join(chr(b) if 32 <= b <= 126 or b in (10, 13, 9) else " " for b in content)
            cleaned = re.sub(r"\s+", " ", printable).strip()
            return cleaned[:10000], 1
        except Exception as e:
            return f"Error extracting PDF: {e!s}", 1


class TextChunker:
    """Splits documents into coherent overlapping chunks preserving sentence boundaries."""

    def __init__(self, chunk_size: int = 400, overlap: int = 60):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, doc_id: str, doc_name: str) -> list[DocumentChunk]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()] if text.strip() else []

        chunks: list[DocumentChunk] = []
        current_words: list[str] = []
        chunk_idx = 0

        for para in paragraphs:
            para_words = para.split()
            if not para_words:
                continue

            if len(current_words) + len(para_words) <= self.chunk_size:
                current_words.extend(para_words)
            else:
                if current_words:
                    chunk_text = " ".join(current_words)
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{doc_id}_c{chunk_idx}",
                            doc_id=doc_id,
                            doc_name=doc_name,
                            text=chunk_text,
                            chunk_index=chunk_idx,
                        )
                    )
                    chunk_idx += 1
                    overlap_words = current_words[-self.overlap :] if self.overlap < len(current_words) else current_words
                    current_words = list(overlap_words)

                # If single paragraph is larger than chunk_size, split by word window
                while len(para_words) > self.chunk_size:
                    window = para_words[: self.chunk_size]
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{doc_id}_c{chunk_idx}",
                            doc_id=doc_id,
                            doc_name=doc_name,
                            text=" ".join(window),
                            chunk_index=chunk_idx,
                        )
                    )
                    chunk_idx += 1
                    para_words = para_words[self.chunk_size - self.overlap :]

                current_words.extend(para_words)

        if current_words:
            chunk_text = " ".join(current_words)
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_id}_c{chunk_idx}",
                    doc_id=doc_id,
                    doc_name=doc_name,
                    text=chunk_text,
                    chunk_index=chunk_idx,
                )
            )

        return chunks


class BM25Retriever:
    """
    Pure Python & NumPy BM25 Okapi retriever.
    Fast, local, offline, zero API keys or external services required.
    """

    STOPWORDS: set[str] = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "were", "will", "with", "i", "you", "we", "they",
        "this", "what", "which", "who", "whom", "tell", "me", "about",
    }

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: list[DocumentChunk] = []
        self.doc_len: list[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: dict[str, int] = {}
        self.term_freqs: list[dict[str, int]] = []
        self.idf: dict[str, float] = {}

    def _tokenize(self, text: str) -> list[str]:
        words = re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())
        return [w for w in words if w not in self.STOPWORDS]

    def build_index(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = list(chunks)
        self.doc_len = []
        self.term_freqs = []
        self.doc_freqs = {}

        total_tokens = 0
        for chunk in self.chunks:
            tokens = self._tokenize(chunk.text)
            self.doc_len.append(len(tokens))
            total_tokens += len(tokens)

            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.term_freqs.append(tf)

            for t in tf:
                self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1

        n_docs = len(self.chunks)
        self.avg_doc_len = total_tokens / max(1, n_docs)

        # Precompute IDF
        self.idf = {}
        for term, df in self.doc_freqs.items():
            # Standard Lucene/BM25 IDF formula
            self.idf[term] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[DocumentChunk, float]]:
        if not self.chunks:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores: list[float] = [0.0] * len(self.chunks)

        for i, tf in enumerate(self.term_freqs):
            doc_len = self.doc_len[i]
            score = 0.0
            for token in query_tokens:
                if token not in tf:
                    continue
                f = tf[token]
                idf = self.idf.get(token, 0.5)
                denom = f + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1.0, self.avg_doc_len)))
                score += idf * (f * (self.k1 + 1.0)) / max(1e-6, denom)

            # Bonus for exact query phrase match
            if query.lower() in self.chunks[i].text.lower():
                score += 2.0

            scores[i] = score

        # Rank
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
        results = []
        for idx in ranked_indices[:top_k]:
            if scores[idx] > 0.05:
                results.append((self.chunks[idx], float(scores[idx])))

        return results


class RAGEngine:
    """
    High-level orchestrator for document ingestion, retrieval,
    and conversational groundings.
    """

    def __init__(self, chunk_size: int = 250, overlap: int = 40):
        self._lock = threading.Lock()
        self.parser = DocumentParser()
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        self.retriever = BM25Retriever()
        self._documents: dict[str, Document] = {}
        self._all_chunks: list[DocumentChunk] = []

    def add_document(self, filename: str, content: bytes, doc_id: str | None = None) -> Document:
        with self._lock:
            doc_id = doc_id or f"doc_{int(time.time() * 1000)}"
            raw_text, num_pages = self.parser.parse_bytes(filename, content)
            chunks = self.chunker.chunk(raw_text, doc_id=doc_id, doc_name=filename)

            doc = Document(
                doc_id=doc_id,
                filename=filename,
                file_type=os.path.splitext(filename)[1].lower() or ".txt",
                raw_text=raw_text,
                chunks=chunks,
                num_pages=num_pages,
            )

            self._documents[doc_id] = doc
            self._rebuild_index_locked()
            return doc

    def add_text(self, title: str, text: str, doc_id: str | None = None) -> Document:
        return self.add_document(filename=title, content=text.encode("utf-8"), doc_id=doc_id)

    def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id in self._documents:
                del self._documents[doc_id]
                self._rebuild_index_locked()
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._documents.clear()
            self._all_chunks.clear()
            self.retriever.build_index([])

    def _rebuild_index_locked(self) -> None:
        self._all_chunks = []
        for doc in self._documents.values():
            self._all_chunks.extend(doc.chunks)
        self.retriever.build_index(self._all_chunks)

    def list_documents(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "doc_id": d.doc_id,
                    "filename": d.filename,
                    "file_type": d.file_type,
                    "num_pages": d.num_pages,
                    "chunks_count": len(d.chunks),
                    "created_at": d.created_at,
                    "preview": d.raw_text[:200] + ("..." if len(d.raw_text) > 200 else ""),
                }
                for d in self._documents.values()
            ]

    def query(self, query_text: str, top_k: int = 3) -> list[tuple[DocumentChunk, float]]:
        with self._lock:
            return self.retriever.retrieve(query_text, top_k=top_k)

    def get_grounded_context(self, query_text: str, top_k: int = 2) -> str | None:
        results = self.query(query_text, top_k=top_k)
        if not results:
            return None
        snippets = []
        for chunk, score in results:
            snippets.append(f"[{chunk.doc_name}]: {chunk.text.strip()}")
        return "\n\n".join(snippets)

    def generate_grounded_response(self, query_text: str) -> str | None:
        """
        Synthesizes a crisp, natural spoken response directly answering
        the user's inquiry from retrieved knowledge chunks.
        """
        results = self.query(query_text, top_k=2)
        if not results:
            return None

        top_chunk, score = results[0]
        # Clean text into 1-2 spoken sentences
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", top_chunk.text) if s.strip()]

        # Find sentence containing the most query words
        query_words = set(re.findall(r"\w+", query_text.lower()))
        best_sentence = sentences[0] if sentences else top_chunk.text
        best_overlap = -1

        for sent in sentences:
            sent_words = set(re.findall(r"\w+", sent.lower()))
            overlap = len(query_words.intersection(sent_words))
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sent

        # Short clean answer
        cleaned_ans = re.sub(r"\[Page \d+\]", "", best_sentence).strip()
        if len(cleaned_ans) > 220:
            cleaned_ans = cleaned_ans[:217] + "..."

        doc_name = top_chunk.doc_name
        return f"According to {doc_name}: {cleaned_ans}"


# Global singleton instance
default_rag_engine = RAGEngine()
