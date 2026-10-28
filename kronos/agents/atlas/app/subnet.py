"""Exact subnet math (stdlib ipaddress). No LLM involved, so results are always correct."""
import ipaddress
import itertools
import math


class SubnetError(ValueError):
    pass


MAX_RESULTS = 1024


def _net(text: str):
    try:
        return ipaddress.ip_network(text.strip(), strict=False)
    except ValueError as exc:
        raise SubnetError(str(exc)) from exc


def describe(text: str) -> dict:
    try:
        iface = ipaddress.ip_interface(text.strip())
    except ValueError as exc:
        raise SubnetError(str(exc)) from exc
    net = iface.network
    d: dict = {"network": str(net), "version": net.version, "prefix": net.prefixlen,
               "addresses": net.num_addresses}
    if net.version == 4:
        if net.prefixlen == 32:
            usable, first, last, bcast = 1, net.network_address, net.network_address, None
        elif net.prefixlen == 31:  # RFC 3021 point-to-point
            usable, first, last, bcast = 2, net.network_address, net.broadcast_address, None
        else:
            usable = net.num_addresses - 2
            first, last = net.network_address + 1, net.broadcast_address - 1
            bcast = net.broadcast_address
        d.update(netmask=str(net.netmask), wildcard=str(net.hostmask),
                 broadcast=str(bcast) if bcast else None,
                 first=str(first), last=str(last), usable=usable)
    else:
        d.update(first=str(net.network_address), last=str(net.broadcast_address))
    if iface.ip != net.network_address:
        d["address"] = str(iface.ip)
    return d


def split(text: str, prefix: int | None = None, count: int | None = None,
          limit: int | None = None) -> dict:
    net = _net(text)
    if prefix is None:
        if not count or count < 2:
            raise SubnetError("give a subnet count of at least 2, or a target prefix like /24")
        prefix = net.prefixlen + math.ceil(math.log2(count))
        limit = limit or count
    if prefix <= net.prefixlen or prefix > net.max_prefixlen:
        raise SubnetError(f"/{prefix} is not a valid smaller subnet of {net}")
    total = 2 ** (prefix - net.prefixlen)
    show = min(total, limit or total, MAX_RESULTS)
    subs = list(itertools.islice(net.subnets(new_prefix=prefix), show))
    return {"parent": str(net), "prefix": prefix, "total": total, "shown": len(subs),
            "subnets": [describe(str(s)) for s in subs]}


def vlsm(text: str, hosts: list[int]) -> dict:
    net = _net(text)
    if net.version != 4:
        raise SubnetError("VLSM helper supports IPv4 only")
    if not hosts or any(h < 1 for h in hosts):
        raise SubnetError("host counts must be positive whole numbers")
    end = int(net.broadcast_address)
    cursor = int(net.network_address)
    allocations = []
    for need in sorted(hosts, reverse=True):
        bits = max(2, math.ceil(math.log2(need + 2)))
        size = 2 ** bits
        cursor = -(-cursor // size) * size  # align up to the block size
        if cursor + size - 1 > end:
            raise SubnetError(f"not enough space in {net} for {need} hosts "
                              f"(needs a /{32 - bits} after earlier allocations)")
        sub = ipaddress.IPv4Network((cursor, 32 - bits))
        allocations.append({"need": need, **describe(str(sub))})
        cursor += size
    return {"parent": str(net), "allocations": allocations,
            "free_addresses": max(0, end - cursor + 1)}


def fmt_describe(d: dict) -> str:
    lines = []
    if "address" in d:
        lines.append(f"Address:      {d['address']}")
    lines.append(f"Network:      {d['network']}")
    if d["version"] == 4:
        lines.append(f"Netmask:      {d['netmask']}  (wildcard {d['wildcard']})")
        if d["broadcast"]:
            lines.append(f"Broadcast:    {d['broadcast']}")
        lines.append(f"Usable hosts: {d['usable']:,}  ({d['first']} - {d['last']})")
    else:
        lines.append(f"Range:        {d['first']} - {d['last']}")
        lines.append(f"Addresses:    {d['addresses']:,}")
    return "```\n" + "\n".join(lines) + "\n```"


def fmt_split(r: dict) -> str:
    first = r["subnets"][0]
    per = f"{first['usable']:,} usable hosts each" if first["version"] == 4 else ""
    head = f"{r['parent']} -> {r['total']} x /{r['prefix']}" + (f" ({per})" if per else "")
    lines = [head, ""]
    for i, s in enumerate(r["subnets"], 1):
        if s["version"] == 4:
            b = f"  bcast {s['broadcast']}" if s["broadcast"] else ""
            lines.append(f"{i:>3}. {s['network']:<18} {s['first']} - {s['last']}{b}")
        else:
            lines.append(f"{i:>3}. {s['network']}")
    if r["shown"] < r["total"]:
        lines.append(f"\n(showing {r['shown']} of {r['total']})")
    return "```\n" + "\n".join(lines) + "\n```"


def fmt_vlsm(r: dict) -> str:
    lines = [f"VLSM plan for {r['parent']}", ""]
    for a in r["allocations"]:
        lines.append(f"need {a['need']:>5} hosts -> {a['network']:<18} {a['first']} - {a['last']}"
                     f"  ({a['usable']:,} usable)")
    lines.append(f"\nFree space left: {r['free_addresses']:,} addresses")
    return "```\n" + "\n".join(lines) + "\n```"
