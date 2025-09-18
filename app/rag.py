"""Simple retrieval-augmented QA over alignment manuals."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


DEFAULT_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "if",
    "in",
    "into",
    "is",
    "it",
    "no",
    "not",
    "of",
    "on",
    "or",
    "such",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "will",
    "with",
}


class SimpleTfidfVectorizer:
    """Minimal TF-IDF vectorizer avoiding the heavy scikit-learn dependency."""

    def __init__(self, stop_words: set[str] | None = None, use_bigrams: bool = True) -> None:
        self.stop_words = stop_words or DEFAULT_STOP_WORDS
        self.use_bigrams = use_bigrams
        self.vocabulary_: dict[str, int] = {}
        self.idf_: List[float] | None = None

    def fit_transform(self, documents: Sequence[str]) -> List[List[float]]:
        tokens_per_doc = [self._tokenise(doc) for doc in documents]
        self._build_vocabulary(tokens_per_doc)
        return self._to_matrix(tokens_per_doc)

    def transform(self, documents: Sequence[str]) -> List[List[float]]:
        if self.idf_ is None:
            raise RuntimeError("Vectorizer has not been fitted")
        tokens_per_doc = [self._tokenise(doc) for doc in documents]
        return self._to_matrix(tokens_per_doc)

    def _tokenise(self, document: str) -> List[str]:
        words = [
            word
            for word in re.findall(r"\b\w+\b", document.lower())
            if word and word not in self.stop_words
        ]
        if not words:
            return []
        if not self.use_bigrams:
            return words
        tokens = list(words)
        tokens.extend(" ".join(pair) for pair in zip(words, words[1:]))
        return tokens

    def _build_vocabulary(self, tokens_per_doc: Sequence[List[str]]) -> None:
        vocab: dict[str, int] = {}
        doc_freq: Counter[str] = Counter()
        for tokens in tokens_per_doc:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] += 1
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)
        self.vocabulary_ = vocab
        n_docs = len(tokens_per_doc)
        idf = [0.0 for _ in range(len(vocab))]
        for token, idx in vocab.items():
            df = doc_freq[token]
            idf[idx] = math.log((1 + n_docs) / (1 + df)) + 1.0
        self.idf_ = idf

    def _to_matrix(self, tokens_per_doc: Sequence[List[str]]) -> List[List[float]]:
        if self.idf_ is None:
            raise RuntimeError("Vectorizer has not been fitted")
        if not self.vocabulary_:
            return [[0.0] * 0 for _ in range(len(tokens_per_doc))]
        matrix = [
            [0.0 for _ in range(len(self.vocabulary_))]
            for _ in range(len(tokens_per_doc))
        ]
        for row, tokens in enumerate(tokens_per_doc):
            if not tokens:
                continue
            counts: Counter[int] = Counter(
                self.vocabulary_.get(token)
                for token in tokens
                if token in self.vocabulary_
            )
            total = float(len(tokens))
            for idx, freq in counts.items():
                if idx is None:
                    continue
                matrix[row][idx] = (freq / total) * self.idf_[idx]
        return matrix


def vector_norms(matrix: Sequence[Sequence[float]]) -> List[float]:
    norms: List[float] = []
    for row in matrix:
        norm = math.sqrt(sum(value * value for value in row))
        if norm == 0.0:
            norm = 1e-12
        norms.append(norm)
    return norms


def cosine_similarity_dense(
    matrix: Sequence[Sequence[float]],
    matrix_norms: Sequence[float],
    query_vec: Sequence[float],
    query_norm: float,
) -> List[float]:
    scores: List[float] = []
    for row, norm in zip(matrix, matrix_norms):
        dot = sum(value * weight for value, weight in zip(row, query_vec))
        denom = norm * query_norm
        if denom == 0.0:
            scores.append(0.0)
        else:
            scores.append(dot / denom)
    return scores


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
        self._vectorizer: SimpleTfidfVectorizer | None = None
        self._matrix: List[List[float]] | None = None
        self._doc_norms: List[float] | None = None

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
        self._vectorizer = SimpleTfidfVectorizer()
        self._matrix = self._vectorizer.fit_transform(texts)
        self._doc_norms = vector_norms(self._matrix)

    def search(self, query: str, top_k: int = 3) -> Sequence[tuple[DocumentChunk, float]]:
        if not query.strip():
            return []
        if self._vectorizer is None or self._matrix is None or self._doc_norms is None:
            raise RuntimeError("Knowledge base has not been loaded")
        query_vec = self._vectorizer.transform([query])
        query_norm = vector_norms(query_vec)[0]
        if query_norm == 0.0:
            return []
        scores = cosine_similarity_dense(self._matrix, self._doc_norms, query_vec[0], query_norm)
        if not any(scores):
            return []
        ranked = sorted(
            ((idx, score) for idx, score in enumerate(scores)),
            key=lambda pair: pair[1],
            reverse=True,
        )[:top_k]
        results = []
        for idx, score in ranked:
            results.append((self.chunks[int(idx)], float(score)))
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
