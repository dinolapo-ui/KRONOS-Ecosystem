from .schemas import RunRequest


def build_messages(req: RunRequest, max_history: int = 20) -> list[dict]:
    """Chat messages for the LLM: orchestrator-supplied history + the current task."""
    history = [
        {"role": h["role"], "content": h["content"]}
        for h in (req.context.get("history") or [])
        if h.get("role") in ("user", "assistant") and h.get("content")
    ][-max_history:]
    return history + [{"role": "user", "content": req.task}]
