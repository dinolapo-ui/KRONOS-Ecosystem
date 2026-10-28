import asyncio

from helpers import load_pkg, sub
from kronos_common import Manifest, OllamaError

load_pkg("orch_app", "orchestrator/app")
router = sub("orch_app", "router")

M = {
    "atlas": Manifest(name="atlas", description="net", capabilities=["subnet", "cisco", "vlan", "routing"]),
    "nyx": Manifest(name="nyx", description="sec", capabilities=["hardening", "ssh", "vulnerability"]),
    "lex": Manifest(name="lex", description="research", capabilities=["research", "history"]),
}


class LLM:
    def __init__(self, reply=None):
        self.reply, self.calls = reply, 0

    async def chat(self, *a, **k):
        self.calls += 1
        if self.reply is None:
            raise OllamaError("down")
        return self.reply


def pick(task, llm=None):
    return asyncio.run(router.choose_agent(task, M, llm or LLM()))


def test_explicit_prefix():
    assert pick("@nyx check this") == ("nyx", "check this", "explicit")


def test_keyword_winner_skips_llm():
    llm = LLM('{"agent": "lex"}')
    agent, _, how = pick("how do I harden SSH on a Linux server", llm)
    assert (agent, how, llm.calls) == ("nyx", "keyword", 0)
    assert pick("subnets and routing", llm)[0] == "atlas"


def test_llm_used_when_ambiguous_and_fallback_when_down():
    assert pick("tell me something", LLM('{"agent": "lex"}'))[0:3:2] == ("lex", "llm")
    assert pick("tell me something") == (None, "tell me something", "none")
    assert pick("history of cisco routing")[0] in {"atlas", "lex"}  # tie -> LLM down -> best keyword
