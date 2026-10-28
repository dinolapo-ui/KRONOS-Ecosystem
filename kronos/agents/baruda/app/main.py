from kronos_common import (Manifest, RunRequest, RunResponse, Status,
                           build_messages, create_agent_app)

MANIFEST = Manifest(
    name="baruda",
    description="Technical medical referencing assistant and structured terminology processor",
    capabilities=["medical", "terminology", "anatomy", "pharmacology", "clinical", "reference"],
)

SYSTEM = "You are BARUDA, a technical medical reference assistant. Use precise terminology, state uncertainty, never give personal medical advice."


async def handle(req: RunRequest, llm) -> RunResponse:
    text = await llm.chat(build_messages(req), system=SYSTEM)
    return RunResponse(agent=MANIFEST.name, status=Status.OK, output=text)


app = create_agent_app(MANIFEST, handle)
