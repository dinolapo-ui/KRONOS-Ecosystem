import logging

import httpx
from kronos_common import Manifest, RunRequest, RunResponse, Status

from .config import AgentConfig

log = logging.getLogger("kronos.registry")


class AgentRegistry:
    def __init__(self, agents: dict[str, AgentConfig]) -> None:
        self.agents = agents
        self._http = httpx.AsyncClient()
        self._manifests: dict[str, Manifest] = {}

    async def refresh(self) -> dict[str, Manifest]:
        """Fetch manifests from every reachable agent; unreachable ones are dropped."""
        found: dict[str, Manifest] = {}
        for name, cfg in self.agents.items():
            try:
                r = await self._http.get(f"{cfg.url}/manifest", timeout=5.0)
                r.raise_for_status()
                found[name] = Manifest.model_validate(r.json())
            except (httpx.HTTPError, ValueError) as exc:
                log.warning("agent %s unavailable: %s", name, exc)
        self._manifests = found
        return found

    @property
    def manifests(self) -> dict[str, Manifest]:
        return self._manifests

    async def run(self, name: str, req: RunRequest) -> RunResponse:
        cfg = self.agents[name]
        try:
            r = await self._http.post(f"{cfg.url}/run", json=req.model_dump(),
                                      timeout=cfg.timeout)
            r.raise_for_status()
            return RunResponse.model_validate(r.json())
        except (httpx.HTTPError, ValueError) as exc:
            log.error("agent %s call failed: %s", name, exc)
            return RunResponse(agent=name, status=Status.ERROR,
                               error=f"agent unreachable or invalid response: {type(exc).__name__}: {exc}")

    async def aclose(self) -> None:
        await self._http.aclose()
