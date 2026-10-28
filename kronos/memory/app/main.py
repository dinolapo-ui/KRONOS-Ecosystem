import logging
import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Path, Query
from kronos_common import configure_logging
from pydantic import BaseModel, Field

from .store import SessionStore
from .vectors import EmbeddingError, VectorStore

log = logging.getLogger("kronos.memory")
SESSION_RE = r"^[A-Za-z0-9_.-]{1,64}$"
COLLECTION_RE = r"^[a-z0-9][a-z0-9_-]{2,62}$"  # Chroma: 3-63 chars


class MessageIn(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1, max_length=50_000)
    agent: str | None = None


class DocIn(BaseModel):
    collection: str = Field(pattern=COLLECTION_RE)
    text: str = Field(min_length=1, max_length=500_000)
    id: str | None = Field(default=None, max_length=128)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class SearchIn(BaseModel):
    collection: str = Field(pattern=COLLECTION_RE)
    query: str = Field(min_length=1, max_length=5_000)
    k: int = Field(default=4, ge=1, le=20)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    os.makedirs(os.path.dirname(os.getenv("MEMORY_DB", "/data/sessions.db")), exist_ok=True)
    app.state.sessions = SessionStore(os.getenv("MEMORY_DB", "/data/sessions.db"))
    app.state.vectors = VectorStore(
        path=os.getenv("CHROMA_PATH", "/data/chroma"),
        ollama_host=os.getenv("OLLAMA_HOST", "http://ollama:11434"),
        embed_model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
    )
    log.info("memory service ready")
    yield
    app.state.sessions.close()
    app.state.vectors.close()


app = FastAPI(title="KRONOS Memory", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/sessions/{session_id}/messages", status_code=201)
def add_message(msg: MessageIn, session_id: str = Path(pattern=SESSION_RE)):
    return {"id": app.state.sessions.add(session_id, msg.role, msg.content, msg.agent)}


@app.get("/sessions/{session_id}/messages")
def get_history(session_id: str = Path(pattern=SESSION_RE),
                limit: int = Query(20, ge=1, le=200)):
    return app.state.sessions.history(session_id, limit)


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str = Path(pattern=SESSION_RE)):
    return {"deleted": app.state.sessions.delete_session(session_id)}


@app.post("/docs", status_code=201)
def add_doc(doc: DocIn):
    try:
        doc_id, chunks = app.state.vectors.add(doc.collection, doc.text, doc.id, doc.metadata)
    except EmbeddingError as exc:
        raise HTTPException(502, str(exc))
    return {"id": doc_id, "chunks": chunks}


@app.post("/docs/search")
def search_docs(q: SearchIn):
    try:
        return app.state.vectors.search(q.collection, q.query, q.k)
    except EmbeddingError as exc:
        raise HTTPException(502, str(exc))


@app.delete("/docs/{collection}/{doc_id}")
def delete_doc(collection: str = Path(pattern=COLLECTION_RE), doc_id: str = Path(max_length=128)):
    app.state.vectors.delete(collection, doc_id)
    return {"deleted": doc_id}
