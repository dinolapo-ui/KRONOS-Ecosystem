"""Async client for the memory service. Every call degrades gracefully:
memory being down must never take an agent or the orchestrator down."""
import logging
import os

import httpx

log = logging.getLogger("kronos.memory.client")


class MemoryClient:
    def __init__(self, base_url: str | None = None, timeout: float = 20.0) -> None:
        self.base_url = (base_url or os.getenv("MEMORY_URL", "http://memory:8000")).rstrip("/")
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def healthy(self) -> bool:
        try:
            (await self._http.get("/health", timeout=3.0)).raise_for_status()
            return True
        except httpx.HTTPError:
            return False

    async def history(self, session_id: str, limit: int = 20) -> list[dict]:
        try:
            r = await self._http.get(f"/sessions/{session_id}/messages", params={"limit": limit})
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("history unavailable: %s", exc)
            return []

    async def append(self, session_id: str, role: str, content: str,
                     agent: str | None = None) -> bool:
        try:
            r = await self._http.post(f"/sessions/{session_id}/messages",
                                      json={"role": role, "content": content, "agent": agent})
            r.raise_for_status()
            return True
        except httpx.HTTPError as exc:
            log.warning("append failed: %s", exc)
            return False

    async def add_doc(self, collection: str, text: str, doc_id: str | None = None,
                      metadata: dict | None = None) -> bool:
        try:
            r = await self._http.post("/docs", json={"collection": collection, "text": text,
                                                     "id": doc_id, "metadata": metadata or {}})
            r.raise_for_status()
            return True
        except httpx.HTTPError as exc:
            log.warning("add_doc failed: %s", exc)
            return False

    async def search(self, collection: str, query: str, k: int = 4) -> list[dict]:
        try:
            r = await self._http.post("/docs/search",
                                      json={"collection": collection, "query": query, "k": k})
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("search failed: %s", exc)
            return []

    async def aclose(self) -> None:
        await self._http.aclose()
