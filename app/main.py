"""FastAPI app: upload documents, list/delete them, and ask questions."""

import json
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import settings
from .ingest import SUPPORTED_EXTENSIONS, UnsupportedFileError, chunk_pages, content_hash, parse
from .llm import LLMError, get_llm
from .rag import RAGPipeline
from .store import VectorStore

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Enterprise RAG (Approach 1: Naive RAG)")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@lru_cache
def get_store() -> VectorStore:
    return VectorStore()


@lru_cache
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(get_store(), get_llm())


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "provider": settings.llm_provider}


@app.post("/api/documents")
async def upload(files: list[UploadFile], store: VectorStore = Depends(get_store)):
    results = []
    for f in files:
        data = await f.read()
        name = Path(f.filename or "upload").name
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            results.append({"filename": name, "status": "error", "detail": "File too large"})
            continue
        doc_id = content_hash(data)[:16]
        if store.has_document(doc_id):
            results.append({"filename": name, "doc_id": doc_id, "status": "duplicate"})
            continue
        try:
            pages = parse(name, data)
        except UnsupportedFileError as e:
            results.append({"filename": name, "status": "error", "detail": str(e)})
            continue
        except Exception as e:  # corrupt or encrypted files
            results.append({"filename": name, "status": "error", "detail": f"Could not parse: {e}"})
            continue
        chunks = chunk_pages(pages, settings.chunk_tokens, settings.chunk_overlap)
        if not chunks:
            results.append(
                {"filename": name, "status": "error", "detail": "No extractable text (scanned file?)"}
            )
            continue
        settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        (settings.uploads_dir / f"{doc_id}{Path(name).suffix.lower()}").write_bytes(data)
        store.add_document(doc_id, name, chunks)
        results.append({"filename": name, "doc_id": doc_id, "status": "indexed", "chunks": len(chunks)})
    return {"results": results}


@app.get("/api/documents")
def list_documents(store: VectorStore = Depends(get_store)):
    return {"documents": store.list_documents(), "supported": sorted(SUPPORTED_EXTENSIONS)}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str, store: VectorStore = Depends(get_store)):
    if not store.has_document(doc_id):
        raise HTTPException(404, "Document not found")
    store.delete_document(doc_id)
    for path in settings.uploads_dir.glob(f"{doc_id}.*"):
        path.unlink()
    return {"deleted": doc_id}


@app.post("/api/ask")
def ask(req: AskRequest):
    try:
        return get_pipeline().ask(req.question, req.top_k).to_dict()
    except LLMError as e:
        raise HTTPException(e.status, e.message)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/ask/stream")
def ask_stream(req: AskRequest):
    """Same as /api/ask, streamed as Server-Sent Events: "sources", then
    "token" events with raw text, then "done" with the validated answer. An LLM
    failure after streaming has started arrives as an "error" event."""
    try:
        pipeline = get_pipeline()
    except LLMError as e:
        raise HTTPException(e.status, e.message)

    def events():
        try:
            for event, data in pipeline.ask_stream(req.question, req.top_k):
                yield _sse(event, data)
        except LLMError as e:
            yield _sse("error", {"status": e.status, "detail": e.message})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        # Stop proxies (e.g. nginx) and browsers from buffering the stream.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
