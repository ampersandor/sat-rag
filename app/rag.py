"""Simple retrieval-augmented QA over alignment manuals."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class DocumentChunk:
    """A chunk of documentation used for retrieval."""

    title: str
    content: str
    source: str
    order: int


class KnowledgeBase:
    """Load documentation text files and expose searchable chunks."""

    def __init__(self, data_dir: Path, chunk_size: int = 800, overlap: int = 120) -> None:
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: List[DocumentChunk] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None

    def load(self) -> None:
        files = sorted(self.data_dir.glob("*.txt"))
        order = 0
        for path in files:
            text = path.read_text(encoding="utf-8")
            paragraphs = split_paragraphs(text)
            chunk_id = 0
            for chunk in chunk_paragraphs(paragraphs, self.chunk_size, self.overlap):
                title = derive_title(chunk, path.stem, chunk_id)
                self.chunks.append(
                    DocumentChunk(title=title, content=chunk, source=path.name, order=order)
                )
                order += 1
                chunk_id += 1
        if not self.chunks:
            raise ValueError(f"No documentation found in {self.data_dir}")
        self._fit()

    def _fit(self) -> None:
        texts = [chunk.content for chunk in self.chunks]
        self._vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), min_df=1
        )
        self._matrix = self._vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 3) -> Sequence[tuple[DocumentChunk, float]]:
        if not query.strip():
            return []
        if self._vectorizer is None or self._matrix is None:
            raise RuntimeError("Knowledge base has not been loaded")
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).ravel()
        if not np.any(scores):
            return []
        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append((self.chunks[int(idx)], float(scores[int(idx)])))
        return results


class QAEngine:
    """High-level question answering engine using TF-IDF retrieval."""

    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self.knowledge_base = knowledge_base

    def answer(self, question: str, top_k: int = 3) -> dict:
        hits = self.knowledge_base.search(question, top_k=top_k)
        if not hits:
            return {
                "question": question,
                "answer": "I could not find relevant information in the manuals.",
                "sources": [],
                "matches": [],
            }
        summary_lines = [
            f"- {hit.title}: {summarise_text(hit.content)}" for hit, _ in hits
        ]
        answer = "Here is what the manuals cover:\n" + "\n".join(summary_lines)
        return {
            "question": question,
            "answer": answer,
            "sources": [hit.source for hit, _ in hits],
            "matches": [
                {
                    "title": hit.title,
                    "source": hit.source,
                    "score": score,
                    "content": hit.content,
                }
                for hit, score in hits
            ],
        }


def split_paragraphs(text: str) -> List[str]:
    """Split text into non-empty paragraphs."""
    parts = [part.strip() for part in re.split(r"\n\s*\n", text)]
    return [part for part in parts if part]


def chunk_paragraphs(paragraphs: Iterable[str], chunk_size: int, overlap: int) -> List[str]:
    """Combine neighbouring paragraphs into overlapping chunks of roughly chunk_size characters."""
    chunks: List[str] = []
    buffer: List[str] = []
    total = 0
    for para in paragraphs:
        para_len = len(para)
        if total + para_len > chunk_size and buffer:
            chunks.append("\n\n".join(buffer))
            # Start a new buffer retaining trailing paragraphs for overlap
            if overlap > 0:
                buffer = paragraph_tail(buffer, overlap)
            else:
                buffer = []
            total = sum(len(part) for part in buffer)
        buffer.append(para)
        total += para_len
    if buffer:
        chunks.append("\n\n".join(buffer))
    return chunks


def paragraph_tail(buffer: Sequence[str], target_chars: int) -> List[str]:
    """Return the trailing paragraphs whose combined length is at least target_chars."""
    if target_chars <= 0:
        return []
    selected: List[str] = []
    total = 0
    for paragraph in reversed(buffer):
        selected.append(paragraph)
        total += len(paragraph)
        if total >= target_chars:
            break
    return list(reversed(selected))


def derive_title(chunk: str, fallback: str, chunk_id: int) -> str:
    """Generate a short title for a chunk using the first heading-like line."""
    if chunk:
        first_line = chunk.splitlines()[0]
        if first_line and len(first_line) < 120:
            return first_line.strip()
    return f"{fallback} section {chunk_id + 1}"


def summarise_text(text: str, max_sentences: int = 2) -> str:
    """Return the first couple of sentences for reporting."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    if not sentences:
        return text.strip()
    return " ".join(sentences[:max_sentences])
