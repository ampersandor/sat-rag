"""FastAPI application exposing a retrieval-based chatbot API."""
from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .rag import KnowledgeBase, QAEngine

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

app = FastAPI(title="Alignment Manual Chatbot", version="0.1.0")


class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question to answer.")
    top_k: int = Field(3, ge=1, le=10, description="Number of supporting passages to return.")


class Match(BaseModel):
    title: str
    source: str
    score: float
    content: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    matches: List[Match]


@app.on_event("startup")
async def load_knowledge_base() -> None:
    """Load manuals into memory when the service starts."""
    try:
        app.state.kb = KnowledgeBase(DATA_DIR)
        app.state.kb.load()
        app.state.qa = QAEngine(app.state.kb)
    except Exception as exc:  # pragma: no cover - startup failure should bubble up
        raise RuntimeError(f"Failed to initialise knowledge base: {exc}") from exc


@app.post("/query", response_model=QueryResponse)
async def query_manuals(request: QueryRequest) -> QueryResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    engine: QAEngine = app.state.qa
    result = engine.answer(request.question, top_k=request.top_k)
    return QueryResponse(**result)


@app.get("/healthz")
async def healthcheck() -> dict:
    """Simple health endpoint for monitoring."""
    status = "ready" if hasattr(app.state, "qa") else "initialising"
    return {"status": status}
