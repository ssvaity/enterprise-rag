# Enterprise RAG — Approach 1 (Naive / Basic RAG)

Upload company documents, ask questions in natural language, and get answers
grounded only in those documents, with `[S#]` citations to file and page.

This is the Phase 1 MVP: fixed-size chunks, one dense embedding per chunk,
cosine top-K retrieval, and Claude answering from the retrieved chunks.

## Pipeline

```
Upload -> Parse (PDF/DOCX/TXT/MD/HTML) -> Clean -> Fixed-size chunks -> Embed -> Chroma
Question -> Embed -> Cosine top-K -> Grounded prompt with [S1..Sk] -> Claude -> Validate tags -> Answer + citations
```

| Piece      | Choice                                                                 |
|------------|------------------------------------------------------------------------|
| Chunking   | ~700 tokens, 15% overlap, never crosses a page (exact page citations)  |
| Embeddings | Chroma default (all-MiniLM-L6-v2, ONNX, runs locally, no key)          |
| Vector DB  | Chroma, persistent, cosine distance                                    |
| Retrieval  | top-K = 6                                                              |
| LLM        | OpenRouter (free models, e.g. `google/gemma-4-31b-it:free`) or Claude `claude-opus-5-5` |
| Citations  | `[S#]` tags; tags not in the retrieved set are dropped; an answer with no valid tags is replaced with "I couldn't find this…" |

## Run

```bash
python3 -m venv .venv.nosync && ln -s .venv.nosync .venv   # .nosync keeps iCloud from evicting it
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # then fill in OPENROUTER_API_KEY (free at openrouter.ai/keys)
.venv/bin/uvicorn app.main:app --reload --env-file .env
```

Open http://localhost:8000. The embedding model (~80 MB) downloads on the first upload.

## Choosing the LLM

| `RAG_LLM_PROVIDER` | Needs                | Model setting                                   |
|--------------------|----------------------|-------------------------------------------------|
| `openrouter`       | `OPENROUTER_API_KEY` | `RAG_OPENROUTER_MODEL` (any `:free` model, or `openrouter/free` to auto-pick one) |
| `anthropic`        | `ANTHROPIC_API_KEY`  | `RAG_MODEL` (default `claude-opus-5-5`)          |

Free OpenRouter models are rate limited (roughly 20 requests/minute, plus a
daily cap) and some providers log prompts, so don't send confidential documents
through them. Each answer reports which model produced it in its `model` field.

## API

| Method | Path                     | Purpose                                   |
|--------|--------------------------|-------------------------------------------|
| POST   | `/api/documents`         | Upload one or more files (`files` field)  |
| GET    | `/api/documents`         | List indexed documents                    |
| DELETE | `/api/documents/{doc_id}`| Remove a document and its vectors         |
| POST   | `/api/ask`               | `{"question": "...", "top_k": 6}`         |

Duplicate uploads are detected by content hash.

## Configuration

Environment variables (see `.env.example`): `RAG_MODEL`, `RAG_DATA_DIR`,
`RAG_CHUNK_TOKENS`, `RAG_CHUNK_OVERLAP`, `RAG_TOP_K`, `RAG_MAX_UPLOAD_MB`.

## Tests

```bash
.venv/bin/python -m pytest -q
```

Tests use real local embeddings and fake LLM clients, so no API key is needed.

## Known limits of Approach 1

These are what Approach 2 (hybrid BM25 + vector, reranking) fixes:

- Misses exact keyword matches (product codes, IDs, names)
- Fixed chunks can cut across sections
- Weak on multi-part questions
- No OCR for scanned files, no PPTX/XLSX yet, no auth or ACLs
