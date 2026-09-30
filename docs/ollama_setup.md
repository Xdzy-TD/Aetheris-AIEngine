# Offline LLM Integration (Ollama)

The controller's `AgenticPlanner` can optionally delegate dynamic tool-chain
planning to a local Ollama model instead of the deterministic fallback router
(`controller/planner.py::_plan_rule_based`). This is what makes the "reasons like
a real agent" differentiator (§7, Agentic Intelligence) work fully offline —
no external API calls, no cloud spend.

## Default model: Qwen2.5-7B

The client's model-preference list (`controller/ollama_client.py::_MODEL_PREFERENCE`)
now tries, in order:

1. `qwen2.5:7b` — **default**. ~4.7GB, runs comfortably on a laptop CPU/GPU
   with tool-calling support, which is what the controller's function-calling
   planner needs.
2. `qwen3.8:27b`, `llama3.1:8b`, `mistral:7b`, `llama3:8b`, `qwen2.5:3b` —
   `qwen3.8:27b` is a heavier fallback (27.3B params, ~18GB at Q4_K_M, vision
   + tools + thinking, 256K context) for machines with the RAM/VRAM to spare;
   the rest are lighter fallbacks below `qwen2.5:7b`.

`OllamaClient.resolve_model()` walks this list against whatever is actually
pulled on the target machine, so the system degrades gracefully rather than
hard-failing if the top preference isn't available.

## Setup

```bash
# 1. Install and start Ollama (https://ollama.com/download)
ollama serve

# 2. Pull the model
ollama pull qwen2.5:7b         # ~4.7GB, the default above

# Heavier machines:
ollama pull qwen3.8:27b        # ~18GB download, needs ~20GB free RAM/VRAM

# 3. Run Aetheris against it
python run.py demo --model qwen2.5:7b -q "What changed between these two images?"
python run.py serve --model qwen2.5:7b
```

Set `OLLAMA_BASE_URL` if Ollama isn't on `localhost:11434` (e.g. a
dedicated inference box on the same LAN as the operator laptop, matching
the on-prem deployment tier in §6 of the spec):

```bash
export OLLAMA_BASE_URL=http://192.168.1.50:11434
export AETHERIS_MODEL=qwen2.5:7b
python run.py serve
```

## Disabling the LLM planner

`--no-llm` (or `AETHERIS_NO_LLM=1`) forces the deterministic fallback router and
skips Ollama entirely — useful for CI, offline demos without a GPU box
handy, or air-gapped environments where even a local model daemon isn't
provisioned yet:

```bash
python run.py demo --no-llm
```

## Checking connectivity

```bash
python -c "
import asyncio
from controller.ollama_client import OllamaClient
print(asyncio.run(OllamaClient().status()))
"
```

Returns `reachable`, `resolved_model`, and the list of locally available
models — useful for a pre-flight check before a live demo.
