from pathlib import Path

import pytest

from app.rag import KnowledgeBase, QAEngine


@pytest.fixture(scope="session")
def qa_engine() -> QAEngine:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    kb = KnowledgeBase(data_dir)
    kb.load()
    return QAEngine(kb)


def test_search_returns_results(qa_engine: QAEngine) -> None:
    response = qa_engine.answer("How do I run MAFFT with iterative refinement?", top_k=2)
    assert response["sources"], "Expected at least one source"
    assert "mafft" in " ".join(response["sources"]).lower()


def test_empty_question_handled(qa_engine: QAEngine) -> None:
    response = qa_engine.answer("   ")
    assert "could not find" in response["answer"].lower()


def test_vsearch_query(qa_engine: QAEngine) -> None:
    response = qa_engine.answer("How do I run vsearch chimera detection?", top_k=2)
    assert response["matches"], "Expected retrieval results for vsearch"
    combined = " ".join(match["content"].lower() for match in response["matches"])
    assert "vsearch" in combined or "uchime" in combined
