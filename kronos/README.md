# KRONOS

Hub-and-spoke multi-agent system. `orchestrator` routes tasks to isolated agent
containers (saga, baruda, lex, nyx, atlas) over one HTTP contract
(`libs/kronos_common/schemas.py`). All LLM access goes through Ollama.

## Run (PowerShell, Docker Desktop)
```
copy .env.example .env
docker compose up --build -d
docker compose exec ollama ollama pull llama3.1:8b
docker compose exec ollama ollama pull nomic-embed-text
curl.exe -s localhost:8080/health
curl.exe -s -X POST localhost:8080/v1/ask -H "Content-Type: application/json" -d "{\"task\":\"@atlas subnet 10.0.0.0/22 into four /24s\"}"
```

## Contract (per agent)
GET /health, GET /manifest, POST /run -> RunResponse. Agents never raise across the wire.

## Layout
orchestrator/ routing + registry | agents/<name>/ one service each |
libs/kronos_common/ contract + Ollama client | native/ future C++ modules | config/agents.yaml

## Memory (sessions + RAG)
`memory/` is the only stateful service (volume `memory_data`): SQLite for conversation
history, ChromaDB for retrieval, embeddings via Ollama (`EMBED_MODEL`). Pass `session_id`
to `/v1/ask` and the orchestrator injects history into the agent and stores the exchange.
Memory outages degrade silently; nothing else depends on it being up.

Ingest a document for LEX (dev port 8081, PowerShell):
```
$b = @{collection="lex"; id="ospf-notes"; text=(Get-Content notes.txt -Raw)} | ConvertTo-Json
curl.exe -s -X POST localhost:8081/docs -H "Content-Type: application/json" -d $b
```
Tests: `pip install ./libs/kronos_common chromadb pytest` then `pytest tests`.

## ATLAS tools
ATLAS decides in code which tool runs, so subnet math and IOS syntax never depend on the
model. Free text works (`@atlas subnet 10.0.0.0/22 into four /24s`, `@atlas create svi for
vlan 10 with ip 10.0.10.1/24`), and so do exact structured calls (PowerShell):
```
$b = @{task="plan"; agent="atlas"; context=@{tool="subnet.vlsm"; params=@{cidr="192.168.0.0/24"; hosts=@(100,50,20)}}} | ConvertTo-Json -Depth 5
Invoke-RestMethod http://localhost:8080/v1/ask -Method Post -ContentType "application/json" -Body $b
```
Tools: `subnet.describe|split|vlsm`, `ios.vlan|access_port|trunk_port|svi|ospf|acl|ssh_hardening`.
Add `"explain": true` to the context (or ask "explain ...") for a short model commentary.
Missing parameters are asked for, never guessed.
