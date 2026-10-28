"""Vector store: ChromaDB with embeddings computed via Ollama (no external downloads)."""
import uuid

import chromadb
import httpx


class EmbeddingError(RuntimeError):
    pass


def chunk_text(text: str, size: int = 1200, overlap: int = 150) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind("\n\n", start + size // 2, end)
            if cut == -1:
                cut = text.rfind(". ", start + size // 2, end)
            if cut != -1:
                end = cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)  # always advances
    return chunks


class VectorStore:
    def __init__(self, path: str, ollama_host: str, embed_model: str) -> None:
        self._client = chromadb.PersistentClient(path=path)
        self._http = httpx.Client(base_url=ollama_host.rstrip("/"), timeout=60.0)
        self._embed_model = embed_model

    def _embed(self, texts: list[str]) -> list[list[float]]:
        try:
            r = self._http.post("/api/embed", json={"model": self._embed_model, "input": texts})
            r.raise_for_status()
            return r.json()["embeddings"]
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise EmbeddingError(
                f"embedding failed (is '{self._embed_model}' pulled in Ollama?): {exc}"
            ) from exc

    def _collection(self, name: str):
        return self._client.get_or_create_collection(
            name, embedding_function=None, metadata={"hnsw:space": "cosine"}
        )

    def add(self, collection: str, text: str, doc_id: str | None, metadata: dict) -> tuple[str, int]:
        doc_id = doc_id or uuid.uuid4().hex
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("empty text")
        col = self._collection(collection)
        col.delete(where={"doc_id": doc_id})  # re-adding a doc replaces it
        col.add(
            ids=[f"{doc_id}:{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=self._embed(chunks),
            metadatas=[{**metadata, "doc_id": doc_id, "chunk": i} for i in range(len(chunks))],
        )
        return doc_id, len(chunks)

    def search(self, collection: str, query: str, k: int) -> list[dict]:
        col = self._collection(collection)
        n = min(k, col.count())
        if n == 0:
            return []
        res = col.query(query_embeddings=self._embed([query]), n_results=n)
        return [
            {"id": i, "text": d, "metadata": m, "distance": dist}
            for i, d, m, dist in zip(res["ids"][0], res["documents"][0],
                                     res["metadatas"][0], res["distances"][0])
        ]

    def delete(self, collection: str, doc_id: str) -> None:
        self._collection(collection).delete(where={"doc_id": doc_id})

    def close(self) -> None:
        self._http.close()
