import asyncio

from kronos_common import MemoryClient, RunRequest, build_messages
from kronos_common.memory import MemoryClient as MC  # noqa: F401


def test_build_messages_filters_and_trims():
    req = RunRequest(task="now", context={"history": [
        {"role": "user", "content": "a"}, {"role": "system", "content": "no"},
        {"role": "assistant", "content": "b"}, {"role": "user", "content": ""}]})
    assert build_messages(req) == [{"role": "user", "content": "a"},
                                   {"role": "assistant", "content": "b"},
                                   {"role": "user", "content": "now"}]
    assert build_messages(RunRequest(task="x")) == [{"role": "user", "content": "x"}]


def test_memory_client_degrades_when_down():
    async def go():
        m = MemoryClient(base_url="http://127.0.0.1:1")
        assert await m.history("s") == []
        assert await m.append("s", "user", "x") is False
        assert await m.search("lex", "q") == []
        assert await m.healthy() is False
        await m.aclose()
    asyncio.run(go())
