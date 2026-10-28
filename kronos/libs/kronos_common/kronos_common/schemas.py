"""The KRONOS agent contract. Every agent and the orchestrator speak only this."""
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Status(str, Enum):
    OK = "ok"
    ERROR = "error"


class Manifest(BaseModel):
    name: str
    description: str
    version: str = "0.1.0"
    capabilities: list[str] = Field(default_factory=list)


class RunRequest(BaseModel):
    task: str = Field(min_length=1, max_length=20_000)
    session_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class RunResponse(BaseModel):
    agent: str
    status: Status
    output: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
