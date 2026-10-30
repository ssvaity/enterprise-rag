"""Chroma vector store: one dense embedding per chunk, cosine similarity search.

Chroma's default embedding function (all-MiniLM-L6-v2 via ONNX) runs locally,
so no embedding API key is needed. The model downloads on first use.
"""

from dataclasses import dataclass
from datetime import datetime, timezone

import chromadb

from .config import settings
from .ingest import Chunk


@dataclass
class Hit:
    text: str
    doc_id: str
    filename: str
    page: int | None
    chunk_index: int
    score: float  # cosine similarity, higher is better


class VectorStore:
    def __init__(self, path: str | None = None, collection: str | None = None):
        if path is None:
            settings.chroma_dir.mkdir(parents=True, exist_ok=True)
            path = str(settings.chroma_dir)
        self._client = chromadb.PersistentClient(path=path)
        self._col = self._client.get_or_create_collection(
            collection or settings.collection, metadata={"hnsw:space": "cosine"}
        )

    def has_document(self, doc_id: str) -> bool:
        return bool(self._col.get(where={"doc_id": doc_id}, limit=1)["ids"])

    def add_document(self, doc_id: str, filename: str, chunks: list[Chunk]) -> None:
        uploaded_at = datetime.now(timezone.utc).isoformat()
        self._col.add(
            ids=[f"{doc_id}:{c.chunk_index}" for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    # Chroma metadata can't hold None; -1 means "no page".
                    "page": c.page if c.page is not None else -1,
                    "chunk_index": c.chunk_index,
                    "uploaded_at": uploaded_at,
                }
                for c in chunks
            ],
        )

    def delete_document(self, doc_id: str) -> None:
        self._col.delete(where={"doc_id": doc_id})

    def list_documents(self) -> list[dict]:
        docs: dict[str, dict] = {}
        for meta in self._col.get(include=["metadatas"])["metadatas"]:
            d = docs.setdefault(
                meta["doc_id"],
                {
                    "doc_id": meta["doc_id"],
                    "filename": meta["filename"],
                    "uploaded_at": meta["uploaded_at"],
                    "chunks": 0,
                },
            )
            d["chunks"] += 1
        return sorted(docs.values(), key=lambda d: d["uploaded_at"], reverse=True)

    def search(self, query: str, k: int) -> list[Hit]:
        if self._col.count() == 0:
            return []
        res = self._col.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append(
                Hit(
                    text=text,
                    doc_id=meta["doc_id"],
                    filename=meta["filename"],
                    page=meta["page"] if meta["page"] >= 0 else None,
                    chunk_index=meta["chunk_index"],
                    score=1.0 - dist,
                )
            )
        return hits
