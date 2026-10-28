"""Deterministic tool dispatch for ATLAS. Code decides which tool runs; the LLM is only
used to extract IOS parameters and (optionally) to explain the result."""
import re
from dataclasses import dataclass, field

from . import ios, subnet


@dataclass
class ToolResult:
    tool: str
    ok: bool
    text: str
    data: dict = field(default_factory=dict)


_CIDR = re.compile(r"\b(\d{1,3}(?:\.\d{1,3}){3})\s*/\s*(\d{1,2})\b")
_IPMASK = re.compile(r"\b(\d{1,3}(?:\.\d{1,3}){3})\s+(?:(?:net)?mask\s+)?(255(?:\.\d{1,3}){3})\b", re.I)
_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen".split())}
_WORD_RE = "|".join(w for w in _WORDS if w != "zero")


def _num(tok: str) -> int | None:
    tok = tok.lower()
    return int(tok) if tok.isdigit() else _WORDS.get(tok)


def _find_network(task: str) -> tuple[str | None, str]:
    for rx in (_CIDR, _IPMASK):
        m = rx.search(task)
        if m:
            return f"{m.group(1)}/{m.group(2)}", task[:m.start()] + " " + task[m.end():]
    return None, task


def wants_explanation(task: str) -> bool:
    return bool(re.search(r"\b(explain|why|how does|walk me through|what does)\b", task.lower()))


def _subnet_request(task: str) -> ToolResult | None:
    net, rest = _find_network(task)
    if not net:
        return None
    t = rest.lower()
    try:
        if "vlsm" in t or ("host" in t and re.search(r"\b(need|needs|require|requires|for)\b", t)):
            hosts = [int(x) for x in re.findall(
                r"\b(\d+)\s*(?:hosts?|users?|devices?|pcs?|computers?|nodes?)\b", t)]
            hosts = hosts or [int(x) for x in re.findall(r"\b\d+\b", t)]
            if hosts:
                r = subnet.vlsm(net, hosts)
                return ToolResult("subnet.vlsm", True, subnet.fmt_vlsm(r), r)
        if re.search(r"\b(split|divide|into|break|carve)\b", t):
            p = re.search(r"/\s*(\d{1,2})", rest)
            c = (re.search(rf"\binto\s+(\d+|{_WORD_RE})\b", t)
                 or re.search(rf"\b(\d+|{_WORD_RE})\s+(?:equal\s+)?(?:subnets?|networks?|parts)\b", t))
            prefix = int(p.group(1)) if p else None
            count = _num(c.group(1)) if c else None
            if prefix or count:
                r = subnet.split(net, prefix=prefix, count=count, limit=count if prefix else None)
                return ToolResult("subnet.split", True, subnet.fmt_split(r), r)
        d = subnet.describe(net)
        return ToolResult("subnet.describe", True, subnet.fmt_describe(d), d)
    except subnet.SubnetError as exc:
        return ToolResult("subnet", False, f"Cannot compute that: {exc}")


async def _ios_request(task: str, llm) -> ToolResult | None:
    key = ios.pick_template(task)
    if not key:
        return None
    t = ios.TEMPLATES[key]
    params = {**await ios.extract_params(llm, t, task), **ios.regex_hints(task, t)}
    ok, text, data = ios.generate(key, params)
    return ToolResult(f"ios.{key}", ok, text, data)


_SUBNET_WORDS = re.compile(r"\b(vlsm|split|divide|carve|into|hosts?)\b", re.I)


async def run_free_text(task: str, llm) -> ToolResult | None:
    # A config request that merely contains a CIDR ("svi ... 10.0.10.1/24") is an IOS job;
    # explicit subnet-planning words send it to the calculator instead.
    if ios.pick_template(task) and not _SUBNET_WORDS.search(task):
        return await _ios_request(task, llm)
    return _subnet_request(task) or await _ios_request(task, llm)


def run_structured(tool: str, params: dict) -> ToolResult:
    """Exact programmatic calls: context={"tool": "subnet.split", "params": {...}}."""
    try:
        if tool == "subnet.describe":
            d = subnet.describe(str(params["cidr"]))
            return ToolResult(tool, True, subnet.fmt_describe(d), d)
        if tool == "subnet.split":
            r = subnet.split(str(params["cidr"]), prefix=params.get("prefix"),
                             count=params.get("count"), limit=params.get("limit"))
            return ToolResult(tool, True, subnet.fmt_split(r), r)
        if tool == "subnet.vlsm":
            r = subnet.vlsm(str(params["cidr"]), [int(h) for h in params["hosts"]])
            return ToolResult(tool, True, subnet.fmt_vlsm(r), r)
        if tool.startswith("ios.") and tool[4:] in ios.TEMPLATES:
            ok, text, data = ios.generate(tool[4:], params)
            return ToolResult(tool, ok, text, data)
    except KeyError as exc:
        return ToolResult(tool, False, f"missing parameter: {exc}")
    except (subnet.SubnetError, ValueError, TypeError) as exc:
        return ToolResult(tool, False, str(exc))
    known = ["subnet.describe", "subnet.split", "subnet.vlsm"] + [f"ios.{k}" for k in ios.TEMPLATES]
    return ToolResult(tool, False, f"unknown tool '{tool}'. Available: {', '.join(known)}")
