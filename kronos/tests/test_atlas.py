import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from helpers import load_pkg, sub

load_pkg("atlas_app", "agents/atlas/app")
subnet = sub("atlas_app", "subnet")
ios = sub("atlas_app", "ios")
tools = sub("atlas_app", "tools")


class FakeLLM:
    def __init__(self, reply="{}"):
        self.reply, self.calls = reply, 0

    async def chat(self, messages, system=None, model=None, json_mode=False):
        self.calls += 1
        return self.reply


def run(coro):
    return asyncio.run(coro)


# ---------- subnet math ----------
def test_describe_basic_and_edges():
    d = subnet.describe("10.0.0.0/22")
    assert (d["netmask"], d["broadcast"], d["usable"]) == ("255.255.252.0", "10.0.3.255", 1022)
    assert (d["first"], d["last"]) == ("10.0.0.1", "10.0.3.254")
    assert subnet.describe("10.0.0.5/32")["usable"] == 1
    assert subnet.describe("10.0.0.0/31")["usable"] == 2
    assert subnet.describe("192.168.1.77/24")["address"] == "192.168.1.77"
    assert subnet.describe("2001:db8::/32")["version"] == 6
    with pytest.raises(subnet.SubnetError):
        subnet.describe("10.0.0.0/33")


def test_split_by_prefix_and_count():
    r = subnet.split("10.0.0.0/22", prefix=24)
    assert [s["network"] for s in r["subnets"]] == [f"10.0.{i}.0/24" for i in range(4)]
    assert subnet.split("10.0.0.0/22", count=4)["prefix"] == 24
    r = subnet.split("10.0.0.0/24", count=3)  # not a power of two: /26 blocks, show 3
    assert r["prefix"] == 26 and r["shown"] == 3 and r["total"] == 4
    with pytest.raises(subnet.SubnetError):
        subnet.split("10.0.0.0/24", prefix=16)


def test_vlsm_is_aligned_and_non_overlapping():
    r = subnet.vlsm("192.168.0.0/24", [20, 100, 50])
    nets = [a["network"] for a in r["allocations"]]
    assert nets == ["192.168.0.0/25", "192.168.0.128/26", "192.168.0.192/27"]
    assert r["free_addresses"] == 32
    with pytest.raises(subnet.SubnetError):
        subnet.vlsm("192.168.0.0/24", [200, 100])


# ---------- IOS ----------
def test_ios_vlan_and_validation():
    ok, text, _ = ios.generate("vlan", {"vlan_id": 20, "name": "sales"})
    assert ok and "vlan 20\n name sales" in text
    ok, text, data = ios.generate("vlan", {"vlan_id": 5000, "name": "x"})
    assert not ok and "vlan_id" in text
    ok, text, _ = ios.generate("vlan", {"vlan_id": 20, "name": "sales\nreload"})
    assert not ok  # no command injection through names
    ok, text, data = ios.generate("vlan", {})
    assert not ok and set(data["needs"]) == {"vlan_id", "name"}


def test_ios_templates_render():
    ok, t, _ = ios.generate("svi", {"vlan_id": 10, "ip": "10.0.10.1", "mask": "24"})
    assert ok and "interface Vlan10" in t and "ip address 10.0.10.1 255.255.255.0" in t
    ok, t, _ = ios.generate("ospf", {"network": "10.0.0.0/24", "router_id": "1.1.1.1"})
    assert ok and "network 10.0.0.0 0.0.0.255 area 0" in t and "router ospf 1" in t
    ok, t, _ = ios.generate("trunk_port", {"interface": "Gi0/1", "allowed_vlans": "10,20,30-40"})
    assert ok and "native vlan 999" in t and "allowed vlan 10,20,30-40" in t
    ok, t, _ = ios.generate("acl", {"name": "NOTEL", "action": "deny", "protocol": "tcp",
                                    "source": "10.0.0.0/24", "port": 23, "interface": "Gi0/0"})
    assert ok and "deny tcp 10.0.0.0 0.0.0.255 any eq 23" in t and "access-group NOTEL in" in t
    ok, t, _ = ios.generate("acl", {"name": "X", "action": "permit", "protocol": "icmp", "port": 5})
    assert not ok
    ok, t, _ = ios.generate("ssh_hardening", {"domain": "lab.local"})
    assert ok and "transport input ssh" in t and "<SET-A-STRONG-SECRET>" in t


def test_pick_template():
    assert ios.pick_template("create vlan 20 named sales") == "vlan"
    assert ios.pick_template("configure trunk on Gi0/1") == "trunk_port"
    assert ios.pick_template("what is a VLAN?") is None
    assert ios.pick_template("how do I harden SSH on a Linux server") is None
    assert ios.pick_template("harden ssh on my cisco router") == "ssh_hardening"


# ---------- free-text dispatch ----------
def test_free_text_subnet_needs_no_llm():
    llm = FakeLLM()
    r = run(tools.run_free_text("subnet 10.0.0.0/22 into four /24s", llm))
    assert r.tool == "subnet.split" and "10.0.3.0/24" in r.text and llm.calls == 0
    r = run(tools.run_free_text("split 192.168.1.0/24 into 4 subnets", llm))
    assert r.data["prefix"] == 26
    r = run(tools.run_free_text("VLSM for 192.168.0.0/24: need 100 hosts, 50 hosts and 20 hosts", llm))
    assert r.tool == "subnet.vlsm" and "192.168.0.192/27" in r.text
    r = run(tools.run_free_text("what is the broadcast for 192.168.1.77 255.255.255.0", llm))
    assert r.tool == "subnet.describe" and "192.168.1.255" in r.text
    assert run(tools.run_free_text("what is OSPF?", llm)) is None


def test_free_text_ios_uses_regex_then_llm():
    llm = FakeLLM(json.dumps({"name": "sales", "vlan_id": 999}))  # model is wrong about the id
    r = run(tools.run_free_text("create vlan 20 named sales", llm))
    assert r.ok and "vlan 20\n name sales" in r.text  # regex hint wins over the model
    r = run(tools.run_free_text("configure access port Gi0/5", FakeLLM("not json")))
    assert not r.ok and "vlan_id" in r.text  # asks for what is missing


def test_config_requests_with_cidr_go_to_ios_not_calculator():
    r = run(tools.run_free_text("create svi for vlan 10 with ip 10.0.10.1/24", FakeLLM()))
    assert r.tool == "ios.svi" and r.ok and "ip address 10.0.10.1 255.255.255.0" in r.text
    r = run(tools.run_free_text("configure ospf for 10.0.0.0/24 area 0 router id 1.1.1.1", FakeLLM()))
    assert r.tool == "ios.ospf" and r.ok and "router-id 1.1.1.1" in r.text
    r = run(tools.run_free_text("split 10.0.0.0/24 into 4 subnets for vlans", FakeLLM()))
    assert r.tool == "subnet.split"


# ---------- through the HTTP contract ----------
def test_atlas_http_contract():
    app = sys_app()
    with TestClient(app) as c:
        j = c.post("/run", json={"task": "subnet 10.0.0.0/22 into four /24s"}).json()
        assert j["status"] == "ok" and j["data"]["tool"] == "subnet.split"
        j = c.post("/run", json={"task": "x", "context": {
            "tool": "subnet.vlsm", "params": {"cidr": "10.0.0.0/24", "hosts": [100, 50]}}}).json()
        assert j["status"] == "ok" and "10.0.0.128/26" in j["output"]
        j = c.post("/run", json={"task": "x", "context": {"tool": "ios.vlan",
                                                           "params": {"vlan_id": 0}}}).json()
        assert j["status"] == "error"
        j = c.post("/run", json={"task": "x", "context": {"tool": "nope"}}).json()
        assert j["status"] == "error" and "Available" in j["error"]


def sys_app():
    return sub("atlas_app", "main").app
