from kronos_common import (Manifest, RunRequest, RunResponse, Status,
                           build_messages, create_agent_app)

MANIFEST = Manifest(
    name="nyx",
    description="Cybersecurity toolkit, network hardening, vulnerability analysis guides",
    capabilities=["security", "hardening", "vulnerability", "cve", "firewall", "threat", "ssh", "audit", "malware", "patch", "encryption"],
)

SYSTEM = "You are NYX, a defensive cybersecurity assistant. Focus on hardening, detection and remediation. Decline offensive misuse."


async def handle(req: RunRequest, llm) -> RunResponse:
    text = await llm.chat(build_messages(req), system=SYSTEM)
    return RunResponse(agent=MANIFEST.name, status=Status.OK, output=text)


app = create_agent_app(MANIFEST, handle)
