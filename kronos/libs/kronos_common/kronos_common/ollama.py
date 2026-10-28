"""Thin async Ollama client. All model access in KRONOS goes through this."""
import os

import httpx


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str | None = None, model: str | None = None,
                 timeout: float = 300.0) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_HOST", "http://ollama:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def chat(self, messages: list[dict], system: str | None = None,
                   model: str | None = None, json_mode: bool = False) -> str:
        payload: dict = {
            "model": model or self.model,
            "messages": ([{"role": "system", "content": system}] if system else []) + messages,
            "stream": False,
        }
        if json_mode:
            payload["format"] = "json"
        try:
            r = await self._http.post("/api/chat", json=payload)
            r.raise_for_status()
            return r.json()["message"]["content"]
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise OllamaError(f"Ollama request failed: {exc}") from exc

    async def healthy(self) -> bool:
        try:
            (await self._http.get("/api/tags", timeout=5.0)).raise_for_status()
            return True
        except httpx.HTTPError:
            return False

    async def has_model(self, name: str) -> bool:
        try:
            r = await self._http.get("/api/tags", timeout=5.0)
            r.raise_for_status()
            names = {m["name"] for m in r.json().get("models", [])}
        except (httpx.HTTPError, KeyError, ValueError):
            return False
        return (name if ":" in name else name + ":latest") in names

    async def aclose(self) -> None:
        await self._http.aclose()
