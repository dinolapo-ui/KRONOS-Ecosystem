import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from kronos_common import (MemoryClient, OllamaClient, RunRequest, RunResponse,
                           Status, configure_logging)
from pydantic import BaseModel

from .config import Settings
from .registry import AgentRegistry
from .router import choose_agent

log = logging.getLogger("kronos.orchestrator")


class AskRequest(RunRequest):
    agent: str | None = None  # bypass routing


class AskResponse(RunResponse):
    routed_by: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = Settings.load()
    app.state.llm = OllamaClient()
    app.state.memory = MemoryClient()
    app.state.registry = AgentRegistry(settings.agents)
    found = await app.state.registry.refresh()
    log.info("KRONOS online, agents: %s", sorted(found))
    yield
    await app.state.registry.aclose()
    await app.state.memory.aclose()
    await app.state.llm.aclose()


app = FastAPI(title="KRONOS Orchestrator", lifespan=lifespan)


@app.get("/health")
async def health():
    llm: OllamaClient = app.state.llm
    embed = os.getenv("EMBED_MODEL", "nomic-embed-text")
    models = {llm.model: await llm.has_model(llm.model), embed: await llm.has_model(embed)}
    return {"status": "ok" if all(models.values()) else "degraded",
            "llm": await llm.healthy(), "models": models,
            "memory": await app.state.memory.healthy(),
            "agents": sorted(app.state.registry.manifests)}


@app.get("/v1/agents")
async def agents():
    return await app.state.registry.refresh()


@app.post("/v1/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    registry: AgentRegistry = app.state.registry
    if not registry.manifests:
        await registry.refresh()
    if not registry.manifests:
        raise HTTPException(503, "no agents available")

    task, method = req.task, "explicit"
    if req.agent:
        if req.agent not in registry.agents:
            raise HTTPException(404, f"unknown agent '{req.agent}'")
        name = req.agent
    else:
        name, task, method = await choose_agent(req.task, registry.manifests,
                                                app.state.llm)
        if name is None:
            raise HTTPException(422, "could not route task; use the 'agent' field or @name")

    memory: MemoryClient = app.state.memory
    context = dict(req.context)
    if req.session_id:
        context["history"] = await memory.history(req.session_id)

    result = await registry.run(name, RunRequest(task=task, session_id=req.session_id,
                                                 context=context))
    if req.session_id and result.status == Status.OK:
        await memory.append(req.session_id, "user", task)
        await memory.append(req.session_id, "assistant", result.output, agent=name)
    return AskResponse(**result.model_dump(), routed_by=method)
