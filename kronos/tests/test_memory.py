import hashlib

import pytest
from fastapi.testclient import TestClient


def fake_embed(self, texts):
    out = []
    for t in texts:  # deterministic bag-of-words hash embedding, 64 dims
        v = [0.0] * 64
        for w in t.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
        out.append(v)
    return out


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_DB", str(tmp_path / "s.db"))
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    from app import vectors
    monkeypatch.setattr(vectors.VectorStore, "_embed", fake_embed)
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_session_history_order_and_limit(client):
    for i in range(5):
        client.post("/sessions/s1/messages", json={"role": "user", "content": f"m{i}"})
    got = client.get("/sessions/s1/messages", params={"limit": 3}).json()
    assert [m["content"] for m in got] == ["m2", "m3", "m4"]
    assert client.get("/sessions/other/messages").json() == []


def test_validation(client):
    assert client.post("/sessions/bad id/messages",
                       json={"role": "user", "content": "x"}).status_code == 422
    assert client.post("/sessions/s/messages",
                       json={"role": "root", "content": "x"}).status_code == 422
    assert client.post("/docs", json={"collection": "X", "text": "t"}).status_code == 422


def test_rag_roundtrip_and_replace(client):
    client.post("/docs", json={"collection": "lex", "id": "a", "text": "ospf area routing cisco"})
    client.post("/docs", json={"collection": "lex", "id": "b", "text": "roman empire aqueduct history"})
    hits = client.post("/docs/search", json={"collection": "lex", "query": "ospf routing", "k": 2}).json()
    assert hits[0]["metadata"]["doc_id"] == "a"
    client.post("/docs", json={"collection": "lex", "id": "a", "text": "totally different words"})
    n = client.post("/docs/search", json={"collection": "lex", "query": "ospf", "k": 10}).json()
    assert len([h for h in n if h["metadata"]["doc_id"] == "a"]) == 1
    client.delete("/docs/lex/a")
    assert client.post("/docs/search", json={"collection": "empty-col", "query": "x"}).json() == []


def test_long_doc_chunks(client):
    r = client.post("/docs", json={"collection": "lex", "text": ("word " * 100 + "\n\n") * 10}).json()
    assert r["chunks"] > 1
