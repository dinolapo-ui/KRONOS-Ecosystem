"""Factory that turns a manifest + handler into a contract-compliant agent service."""
import logging
from contextlib import asynccontextmanager
from typing import Awaitable, Callable

from fastapi import FastAPI

from .logging import configure_logging
from .ollama import OllamaClient
from .schemas import Manifest, RunRequest, RunResponse, Status

log = logging.getLogger("kronos.agent")
Handler = Callable[[RunRequest, OllamaClient], Awaitable[RunResponse]]


def create_agent_app(manifest: Manifest, handler: Handler) -> FastAPI:
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.llm = OllamaClient()
        log.info("agent %s starting (model=%s)", manifest.name, app.state.llm.model)
        yield
        await app.state.llm.aclose()

    app = FastAPI(title=f"KRONOS/{manifest.name}", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"agent": manifest.name, "status": "ok",
                "llm": await app.state.llm.healthy()}

    @app.get("/manifest", response_model=Manifest)
    async def get_manifest():
        return manifest

    @app.post("/run", response_model=RunResponse)
    async def run(req: RunRequest):
        try:
            return await handler(req, app.state.llm)
        except Exception as exc:  # contract: agents never raise across the wire
            log.exception("agent %s failed", manifest.name)
            return RunResponse(agent=manifest.name, status=Status.ERROR, error=str(exc))

    return app
