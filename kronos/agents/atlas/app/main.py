import logging

from kronos_common import (Manifest, OllamaError, RunRequest, RunResponse, Status,
                           build_messages, create_agent_app)

from .tools import run_free_text, run_structured, wants_explanation

log = logging.getLogger("kronos.atlas")

MANIFEST = Manifest(
    name="atlas",
    description="Networking tools: subnet/VLSM calculator, Cisco IOS config templates, "
                "packet analysis helpers",
    capabilities=["network", "networking", "cisco", "ios", "subnet", "vlsm", "cidr", "mask",
                  "vlan", "trunk", "routing", "router", "switch", "ospf", "acl", "interface",
                  "packet", "gateway", "configure"],
)

SYSTEM = (
    "You are ATLAS, a networking engineer assistant. Give exact Cisco IOS syntax. "
    "Never do subnet arithmetic from memory: tell the user to ask with a CIDR such as "
    "'subnet 10.0.0.0/22 into four /24s' so the calculator tool runs."
)
EXPLAIN = (
    "Explain the tool result below in at most five short sentences: what it means and any "
    "caveats. Do not repeat the full table or config. Do not change, add or invent any values."
)


async def handle(req: RunRequest, llm) -> RunResponse:
    explain = bool(req.context.get("explain", False))
    structured = "tool" in req.context
    if structured:
        res = run_structured(str(req.context["tool"]), req.context.get("params") or {})
        if not res.ok:
            return RunResponse(agent=MANIFEST.name, status=Status.ERROR, error=res.text)
    else:
        res = await run_free_text(req.task, llm)
        explain = explain or wants_explanation(req.task)

    if res is None:  # not a tool request: plain conversation
        text = await llm.chat(build_messages(req), system=SYSTEM)
        return RunResponse(agent=MANIFEST.name, status=Status.OK, output=text)

    output = res.text
    if explain and res.ok:
        try:
            output += "\n\n" + await llm.chat(
                [{"role": "user", "content": f"Request: {req.task}\n\nResult:\n{res.text}"}],
                system=EXPLAIN)
        except OllamaError as exc:  # the deterministic result is still valid
            log.warning("explanation skipped: %s", exc)
    return RunResponse(agent=MANIFEST.name, status=Status.OK, output=output,
                       data={"tool": res.tool, "ok": res.ok, **res.data})


app = create_agent_app(MANIFEST, handle)
