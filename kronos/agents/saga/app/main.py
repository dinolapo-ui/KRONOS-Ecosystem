from kronos_common import (Manifest, RunRequest, RunResponse, Status,
                           build_messages, create_agent_app)

MANIFEST = Manifest(
    name="saga",
    description="Personal productivity, task tracking, workflow organization, career and portfolio development",
    capabilities=["tasks", "planning", "workflow", "productivity", "career", "portfolio", "resume"],
)

SYSTEM = "You are SAGA, a pragmatic productivity and career assistant. Be concise and actionable."


async def handle(req: RunRequest, llm) -> RunResponse:
    text = await llm.chat(build_messages(req), system=SYSTEM)
    return RunResponse(agent=MANIFEST.name, status=Status.OK, output=text)


app = create_agent_app(MANIFEST, handle)
