from kronos_common import (Manifest, MemoryClient, RunRequest, RunResponse, Status,
                           build_messages, create_agent_app)

MANIFEST = Manifest(
    name="lex",
    description="Deep-dive research, historical analysis, narrative technical documentation",
    capabilities=["research", "history", "analysis", "documentation", "summarize", "sources"],
)

SYSTEM = (
    "You are LEX, a rigorous research analyst. Structure findings, separate fact from "
    "inference, flag weak evidence. When SOURCES are provided, ground your answer in them "
    "and cite them by number; if they do not cover the question, say so."
)

memory = MemoryClient()


async def handle(req: RunRequest, llm) -> RunResponse:
    hits = await memory.search("lex", req.task, k=4)  # [] if memory/embeddings are down
    system = SYSTEM
    if hits:
        sources = "\n\n".join(f"[{i}] {h['text']}" for i, h in enumerate(hits, 1))
        system += f"\n\nSOURCES:\n{sources}"
    text = await llm.chat(build_messages(req), system=system)
    return RunResponse(agent=MANIFEST.name, status=Status.OK, output=text,
                       data={"sources": [h["id"] for h in hits]})


app = create_agent_app(MANIFEST, handle)
