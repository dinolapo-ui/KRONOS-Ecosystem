"""Cisco IOS config templates. The LLM may extract parameters from free text, but every
value is validated in code and the config itself is rendered deterministically."""
import ipaddress
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

from kronos_common import OllamaError

log = logging.getLogger("kronos.atlas.ios")


class ParamError(ValueError):
    pass


# ---------- validators ----------
_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,32}$")
_DOMAIN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{0,62}$")
_IFACE_WORDS = (r"gigabitethernet|fastethernet|tengigabitethernet|ethernet|port-channel|"
                r"loopback|gi|fa|te|eth|po|lo")
_IFACE = re.compile(rf"^(?:{_IFACE_WORDS})\d+(?:/\d+){{0,3}}$", re.I)
_VLANLIST = re.compile(r"^\d{1,4}(?:[,-]\d{1,4})*$")


def v_int(lo: int, hi: int) -> Callable[[Any], int]:
    def f(v: Any) -> int:
        try:
            n = int(str(v).strip())
        except ValueError:
            raise ParamError(f"must be a whole number between {lo} and {hi}")
        if not lo <= n <= hi:
            raise ParamError(f"must be between {lo} and {hi}")
        return n
    return f


def v_name(v: Any) -> str:
    s = str(v).strip()
    if not _NAME.match(s):
        raise ParamError("use letters, digits, '_', '.', '-' only (max 32 chars, no spaces)")
    return s


def v_domain(v: Any) -> str:
    s = str(v).strip()
    if not _DOMAIN.match(s):
        raise ParamError("not a valid domain name")
    return s


def v_iface(v: Any) -> str:
    s = re.sub(r"\s+", "", str(v).strip())
    if not _IFACE.match(s):
        raise ParamError("expected something like Gi0/1, Fa0/24 or GigabitEthernet1/0/1")
    return s


def v_ip(v: Any) -> str:
    return str(ipaddress.IPv4Address(str(v).strip()))


def v_mask(v: Any) -> str:
    s = str(v).strip().lstrip("/")
    try:
        return str(ipaddress.IPv4Network(f"0.0.0.0/{s}").netmask)
    except ValueError:
        raise ParamError("use a prefix like 24 or a mask like 255.255.255.0")


def v_cidr(v: Any) -> str:
    return str(ipaddress.IPv4Network(str(v).strip(), strict=False))


def v_addr(v: Any) -> str:
    s = str(v).strip().lower()
    if s == "any":
        return "any"
    net = ipaddress.IPv4Network(s, strict=False)
    if net.prefixlen == 32:
        return f"host {net.network_address}"
    return f"{net.network_address} {net.hostmask}"


def v_vlanlist(v: Any) -> str:
    s = re.sub(r"\s+", "", str(v))
    if not _VLANLIST.match(s):
        raise ParamError("use a list like 10,20,30-40")
    for part in re.split(r"[,-]", s):
        if not 1 <= int(part) <= 4094:
            raise ParamError("VLAN numbers must be 1-4094")
    return s


def v_choice(*opts: str) -> Callable[[Any], str]:
    def f(v: Any) -> str:
        s = str(v).strip().lower()
        if s not in opts:
            raise ParamError("must be one of: " + ", ".join(opts))
        return s
    return f


@dataclass
class Field:
    name: str
    parse: Callable[[Any], Any]
    help: str
    required: bool = True
    default: Any = None


@dataclass
class Template:
    key: str
    title: str
    fields: list[Field]
    example: str
    render: Callable[[dict], tuple[list[str], list[str]]]


# ---------- renderers: return (config lines, notes) ----------
def _vlan(p):
    return [f"vlan {p['vlan_id']}", f" name {p['name']}"], []


def _access(p):
    lines = [f"interface {p['interface']}"]
    if p.get("description"):
        lines.append(f" description {p['description']}")
    lines += [" switchport mode access", f" switchport access vlan {p['vlan_id']}",
              " spanning-tree portfast", " spanning-tree bpduguard enable", " no shutdown"]
    return lines, ["The VLAN must exist on the switch (create it with the VLAN template)."]


def _trunk(p):
    lines = [f"interface {p['interface']}"]
    if p.get("description"):
        lines.append(f" description {p['description']}")
    lines += [" switchport mode trunk", f" switchport trunk native vlan {p['native_vlan']}",
              f" switchport trunk allowed vlan {p['allowed_vlans']}", " no shutdown"]
    return lines, ["On platforms that support ISL and 802.1Q (e.g. Catalyst 3560/3750), add "
                   "'switchport trunk encapsulation dot1q' before 'switchport mode trunk'.",
                   "Native VLAN defaults to 999 (unused) rather than VLAN 1; the other end must match."]


def _svi(p):
    lines = [f"interface Vlan{p['vlan_id']}"]
    if p.get("description"):
        lines.append(f" description {p['description']}")
    lines += [f" ip address {p['ip']} {p['mask']}", " no shutdown"]
    return lines, ["Multilayer switches need 'ip routing' enabled for inter-VLAN routing."]


def _ospf(p):
    net = ipaddress.IPv4Network(p["network"])
    lines = [f"router ospf {p['process_id']}"]
    if p.get("router_id"):
        lines.append(f" router-id {p['router_id']}")
    lines.append(f" network {net.network_address} {net.hostmask} area {p['area']}")
    if p.get("passive_interface"):
        lines.append(f" passive-interface {p['passive_interface']}")
    return lines, ["Repeat the 'network' line for each connected network you want to advertise."]


def _acl(p):
    rule = f" {p['action']} {p['protocol']} {p['source']} {p['destination']}"
    if p.get("port") is not None:
        if p["protocol"] not in ("tcp", "udp"):
            raise ParamError("port: only valid with protocol tcp or udp")
        rule += f" eq {p['port']}"
    lines = [f"ip access-list extended {p['name']}", rule]
    notes = ["Every ACL ends with an implicit 'deny ip any any'; add explicit permits as needed.",
             "Rules are evaluated top to bottom; order matters."]
    if p.get("interface"):
        lines += [f"interface {p['interface']}", f" ip access-group {p['name']} {p['direction']}"]
    return lines, notes


def _ssh(p):
    lines = []
    if p.get("hostname"):
        lines.append(f"hostname {p['hostname']}")
    lines += [f"ip domain-name {p['domain']}", "crypto key generate rsa modulus 2048",
              "ip ssh version 2", "ip ssh time-out 60", "ip ssh authentication-retries 3",
              f"username {p['username']} privilege 15 secret <SET-A-STRONG-SECRET>",
              "line vty 0 15", " transport input ssh", " login local",
              " exec-timeout 10 0", " logging synchronous"]
    return lines, ["Replace <SET-A-STRONG-SECRET> yourself; never paste real passwords into a chat.",
                   "RSA key generation needs a non-default hostname and a domain name.",
                   "Consider an ACL on the VTY lines ('access-class') to limit management sources."]


_VLAN = Field("vlan_id", v_int(1, 4094), "VLAN number, 1-4094")
_IF = Field("interface", v_iface, "interface, e.g. Gi0/1")
_DESC = Field("description", v_name, "short description, no spaces (optional)", required=False)

TEMPLATES: dict[str, Template] = {t.key: t for t in [
    Template("vlan", "VLAN", [_VLAN, Field("name", v_name, "VLAN name, no spaces")],
             "create vlan 20 named sales", _vlan),
    Template("access_port", "Access port",
             [_IF, _VLAN, _DESC], "access port Gi0/5 in vlan 20", _access),
    Template("trunk_port", "Trunk port",
             [_IF, Field("allowed_vlans", v_vlanlist, "allowed VLANs, e.g. 10,20,30-40"),
              Field("native_vlan", v_int(1, 4094), "native VLAN", required=False, default=999),
              _DESC], "trunk on Gi0/1 allowed vlans 10,20,30", _trunk),
    Template("svi", "SVI (VLAN interface)",
             [_VLAN, Field("ip", v_ip, "IPv4 address"), Field("mask", v_mask, "prefix or mask"),
              _DESC], "svi for vlan 10 with ip 10.0.10.1/24", _svi),
    Template("ospf", "OSPF",
             [Field("network", v_cidr, "network to advertise, e.g. 10.0.0.0/24"),
              Field("area", v_int(0, 4294967295), "area number", required=False, default=0),
              Field("process_id", v_int(1, 65535), "process id", required=False, default=1),
              Field("router_id", v_ip, "router id (an IPv4 address)", required=False),
              Field("passive_interface", v_iface, "passive interface (optional)", required=False)],
             "configure ospf for 10.0.0.0/24 area 0 router id 1.1.1.1", _ospf),
    Template("acl", "Extended ACL",
             [Field("name", v_name, "ACL name"),
              Field("action", v_choice("permit", "deny"), "permit or deny"),
              Field("protocol", v_choice("ip", "tcp", "udp", "icmp"), "ip, tcp, udp or icmp"),
              Field("source", v_addr, "source: any, a host or a CIDR", required=False, default="any"),
              Field("destination", v_addr, "destination: any, a host or a CIDR",
                    required=False, default="any"),
              Field("port", v_int(1, 65535), "destination port (tcp/udp only)", required=False),
              Field("interface", v_iface, "interface to apply it on (optional)", required=False),
              Field("direction", v_choice("in", "out"), "in or out", required=False, default="in")],
             "acl BLOCK-TELNET deny tcp from 10.0.0.0/24 to any port 23", _acl),
    Template("ssh_hardening", "SSH hardening",
             [Field("domain", v_domain, "domain name, e.g. lab.local"),
              Field("hostname", v_name, "device hostname (optional)", required=False),
              Field("username", v_name, "admin username", required=False, default="admin")],
             "harden ssh on cisco router, domain lab.local, hostname R1", _ssh),
]}


# ---------- selection and extraction ----------
def pick_template(task: str) -> str | None:
    t = task.lower()
    verbs = re.search(r"\b(config|configure|configuration|template|generate|create|set\s?up|"
                      r"setup|build|write|give me|make|add|enable|apply|harden)\b", t)
    cisco = re.search(r"\b(cisco|ios)\b", t)
    if not (verbs or cisco):
        return None
    if re.match(r"\s*(what|why|explain|difference|define)\b", t) and not verbs:
        return None
    if "trunk" in t:
        return "trunk_port"
    if re.search(r"access[\s-]*port|switchport access|access mode", t):
        return "access_port"
    if re.search(r"\bsvi\b|interface vlan|vlan interface", t):
        return "svi"
    if "ospf" in t:
        return "ospf"
    if re.search(r"\bacl\b|access[\s-]*list", t):
        return "acl"
    if "ssh" in t and re.search(r"\b(cisco|ios|router|switch)\b", t):
        return "ssh_hardening"
    if re.search(r"\bvlan\b", t):
        return "vlan"
    return None


_IFACE_FIND = re.compile(rf"\b(?:{_IFACE_WORDS})\s?\d+(?:/\d+){{0,3}}\b", re.I)
_CIDR_FIND = re.compile(r"\b(\d{1,3}(?:\.\d{1,3}){3})\s*/\s*(\d{1,2})\b")


def regex_hints(task: str, t: Template) -> dict:
    names = {f.name for f in t.fields}
    out: dict = {}

    def grab(key: str, pattern: str, flags: int = re.I) -> None:
        if key in names:
            m = re.search(pattern, task, flags)
            if m:
                out[key] = m.group(1)

    grab("vlan_id", r"\bvlan\s*(?:id\s*)?#?(\d{1,4})\b")
    grab("name", r"\bnamed?\s+[\"']?([A-Za-z0-9_.-]+)")
    grab("native_vlan", r"\bnative\s+vlan\s+(\d{1,4})")
    grab("allowed_vlans", r"\ballow(?:ed)?\s+vlans?\s+([\d,\- ]+\d)")
    grab("area", r"\barea\s+(\d+)")
    grab("process_id", r"\bprocess(?:\s+id)?\s+(\d+)")
    grab("router_id", r"\brouter[\s-]*id\s+(\d{1,3}(?:\.\d{1,3}){3})")
    grab("domain", r"\bdomain(?:[\s-]*name)?\s+([A-Za-z0-9.-]+\.[A-Za-z]{2,})")
    grab("hostname", r"\bhostname\s+([A-Za-z0-9_.-]+)")
    grab("username", r"\buser(?:name)?\s+([A-Za-z0-9_.-]+)")
    if "interface" in names:
        m = _IFACE_FIND.search(task)
        if m:
            out["interface"] = m.group(0)
    m = _CIDR_FIND.search(task)
    if m:
        if {"ip", "mask"} <= names:
            out["ip"], out["mask"] = m.group(1), m.group(2)
        if "network" in names:
            out["network"] = f"{m.group(1)}/{m.group(2)}"
    return out


async def extract_params(llm, t: Template, task: str) -> dict:
    """Ask the model for parameters. Failures return {} so we fall back to regex + asking."""
    spec = "\n".join(f"- {f.name}: {f.help}" for f in t.fields)
    system = (f'Extract parameters for a Cisco IOS "{t.title}" config from the user\'s request.\n'
              "Reply ONLY with a JSON object using these keys. Omit any key the user did not "
              f"specify; never invent values.\n{spec}")
    try:
        raw = await llm.chat([{"role": "user", "content": task}], system=system, json_mode=True)
        data = json.loads(raw)
    except (OllamaError, ValueError) as exc:
        log.warning("parameter extraction failed: %s", exc)
        return {}
    if not isinstance(data, dict):
        return {}
    known = {f.name for f in t.fields}
    return {k: v for k, v in data.items()
            if k in known and isinstance(v, (str, int)) and str(v).strip()}


# ---------- binding and rendering ----------
def generate(key: str, raw: dict) -> tuple[bool, str, dict]:
    """Validate raw params and render. Returns (ok, text, data)."""
    t = TEMPLATES[key]
    params: dict = {}
    missing: list[Field] = []
    bad: list[str] = []
    for f in t.fields:
        v = raw.get(f.name)
        if v is None or str(v).strip() == "":
            if f.default is not None:
                params[f.name] = f.parse(f.default)
            elif f.required:
                missing.append(f)
            continue
        try:
            params[f.name] = f.parse(v)
        except ValueError as exc:  # ParamError and ipaddress errors
            bad.append(f"{f.name}: {exc}")
    if bad or missing:
        parts = []
        if bad:
            parts.append("Invalid value(s):\n" + "\n".join(f"- {b}" for b in bad))
        if missing:
            parts.append("I still need:\n" + "\n".join(f"- {f.name}: {f.help}" for f in missing))
        parts.append(f"Example request: \"{t.example}\"")
        return False, "\n\n".join(parts), {"needs": [f.name for f in missing], "invalid": bad}
    try:
        lines, notes = t.render(params)
    except ValueError as exc:
        return False, f"Invalid combination: {exc}", {"invalid": [str(exc)]}
    text = f"Cisco IOS - {t.title}\n```\nconfigure terminal\n" + "\n".join(lines) + "\nend\n```"
    if notes:
        text += "\n\nNotes:\n" + "\n".join(f"- {n}" for n in notes)
    used = ", ".join(f"{k}={v}" for k, v in params.items())
    text += f"\n\nParameters used: {used}\nSave with 'write memory' after verifying."
    return True, text, {"config": lines, "params": params}
