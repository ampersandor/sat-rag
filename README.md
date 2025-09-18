# Alignment Manual Chatbot API

This repository provides a FastAPI service that answers questions about multiple
sequence alignment tools using a lightweight retrieval pipeline. The knowledge
base ships with curated manuals for MAFFT, UCLUST/USEARCH and VSEARCH, enabling a
chatbot-style interface that responds with relevant excerpts.

## Features

- ✅ Retrieval-augmented question answering across custom documentation.
- ✅ FastAPI endpoint for integration with web front-ends (such as the companion React UI).
- ✅ TF–IDF vector search with multi-paragraph chunking.
- ✅ Simple health check endpoint for monitoring.
- ✅ Pytest coverage for the retrieval core.

## Project layout

```
app/
  main.py          # FastAPI application entry point
  rag.py           # Retrieval and QA engine implementation
scripts/
  fetch_docs.py    # (Optional) helper for downloading manuals from the web
data/
  *.txt            # Curated documentation used by the RAG engine
```

## Getting started

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API locally (reload enabled):

```bash
uvicorn app.main:app --reload
```

Example request:

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "How do I enable iterative refinement in MAFFT?"}'
```

Example response:

```json
{
  "question": "How do I enable iterative refinement in MAFFT?",
  "answer": "Here is what the manuals cover:\n- MAFFT Multiple Sequence Alignment Manual (summary): --linsi — L-INS-i algorithm; uses local pairwise alignment information and iterative refinement. High accuracy for a moderate number of sequences. --ginsi — G-INS-i algorithm; global homology assumption. Effective for sequences with uniform lengths.",
  "sources": ["mafft_manual.txt"],
  "matches": [
    {
      "title": "MAFFT Multiple Sequence Alignment Manual (summary)",
      "source": "mafft_manual.txt",
      "score": 0.42,
      "content": "..."
    }
  ]
}
```

Run tests:

```bash
pytest
```

The service can be paired with the existing React/Vite front-end by pointing its
API requests to the `/query` endpoint exposed here.
