import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AgentConfig:
    name: str
    url: str
    timeout: float = 120.0


@dataclass(frozen=True)
class Settings:
    agents: dict[str, AgentConfig]

    @classmethod
    def load(cls) -> "Settings":
        path = Path(os.getenv("KRONOS_AGENTS_FILE", "/srv/config/agents.yaml"))
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        agents = {
            name: AgentConfig(name=name, url=cfg["url"].rstrip("/"),
                              timeout=float(cfg.get("timeout", 120)))
            for name, cfg in (raw.get("agents") or {}).items()
        }
        if not agents:
            raise RuntimeError(f"No agents defined in {path}")
        return cls(agents=agents)
