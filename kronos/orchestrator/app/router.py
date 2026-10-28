"""Routing: explicit @agent -> clear keyword winner -> LLM classification -> best keyword."""
import json
import logging
import os
import re

from kronos_common import Manifest, OllamaClient, OllamaError

log = logging.getLogger("kronos.router")
_PREFIX = re.compile(r"^@(\w+)\s+(.*)$", re.DOTALL)

_SYSTEM = (
    "You route a user task to exactly one specialist agent. "
    'Reply ONLY with JSON: {"agent": "<name or none>"}.\nAgents:\n{catalog}'
)


def _match(word: str, cap: str) -> bool:
    if word == cap:
        return True
    if len(word) < 4 or len(cap) < 4:
        return False
    common = len(os.path.commonprefix([word, cap]))
    return common >= min(len(word), len(cap)) - 1  # harden ~ hardening, subnet ~ subnets


def _scores(task: str, manifests: dict[str, Manifest]) -> dict[str, int]:
    words = set(re.findall(r"[a-z0-9]+", task.lower()))
    return {
        name: sum(1 for cap in {c.lower() for c in m.capabilities}
                  if any(_match(w, cap) for w in words))
        for name, m in manifests.items()
    }


async def choose_agent(task: str, manifests: dict[str, Manifest],
                       llm: OllamaClient) -> tuple[str | None, str, str]:
    """Return (agent, cleaned_task, method)."""
    m = _PREFIX.match(task.strip())
    if m and m.group(1).lower() in manifests:
        return m.group(1).lower(), m.group(2), "explicit"

    ranked = sorted(_scores(task, manifests).items(), key=lambda kv: kv[1], reverse=True)
    if ranked and ranked[0][1] > 0 and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]):
        return ranked[0][0], task, "keyword"  # clear winner: no LLM call needed

    catalog = "\n".join(f"- {n}: {mf.description}" for n, mf in manifests.items())
    try:
        raw = await llm.chat([{"role": "user", "content": task}],
                             system=_SYSTEM.replace("{catalog}", catalog), json_mode=True)
        pick = str(json.loads(raw).get("agent", "")).lower()
        if pick in manifests:
            return pick, task, "llm"
    except (OllamaError, ValueError, AttributeError) as exc:
        log.warning("LLM routing failed: %s", exc)

    if ranked and ranked[0][1] > 0:
        return ranked[0][0], task, "keyword"
    return None, task, "none"
