# Aetheris

Agentic Remote-Sensing Intelligence System — a satellite/aerial imagery
analysis service where an LLM-backed (with rule-based fallback) controller
inspects a query and dynamically composes a chain of specialist tools,
rather than running a fixed pipeline.

> **Note on this file:** the original `README.md` was lost in the archive
> this repository was restored from — see `RESTORATION_NOTES.md` — and has
> been rebuilt from `docs/architecture.md`, `docs/STATUS.md`,
> `pyproject.toml`, and the run/deploy scripts. Treat it as a best-effort
> reconstruction, not the original text.

## What it does

A query plus one or more images goes to the agentic controller
(`controller/planner.py`), which builds an `ExecutionPlan` — an ordered
sequence of specialist tool calls — validates it against a policy engine,
executes it, and returns an `ExecutionResult` with evidence, confidence, and
a tamper-evident audit trail. See `docs/architecture.md` for the full system
diagram and `docs/STATUS.md` for an honest accounting of what's a real
trained model versus a classical signal-processing baseline.

## Layout

```
controller/     agentic planner, policy engine, tool registry, audit log
specialists/    per-task analysis modules (VQA, grounding, change detection, SAR, geo-RAG, fusion)
confidence/     conformal prediction + self-consistency voting
preprocessing/  co-registration and SAR speckle-filter fallbacks
interfaces/     FastAPI (api/), Textual console (ui/), voice I/O (voice/)
pipeline/       the M01-M20 hardening layer: config, adapters, modules,
                schemas, services, the Flask GUI, and pipeline/tests/
benchmark/      JSONL-dataset benchmark harness
training/       LoRA fine-tuning and fusion-model fitting scripts
evaluation/     mask/bbox IoU and F1 metrics
tests/          top-level test suite (see also pipeline/tests/)
docs/           architecture and implementation-status docs
deploy/cloud_demo/  Dockerfile, docker-compose.yml, entrypoint.sh
models/         registry.json — what model backs each task, honestly labelled
```

## Running it

```bash
pip install -e ".[dev]"

# FastAPI service
python -m pipeline.run serve   # or: uvicorn interfaces.api.app:app

# Flask GUI (map + voice dictation)
python -m pipeline.run gui

# Test harness (lint, types, tests, smoke test, benchmark, audit verify)
./harness/run.sh
```

See `.env.example` for configuration (Ollama endpoint, API key, upload
limits, LoRA adapter paths, GUI SSL, etc.) and `docs/ollama_setup.md` for
setting up the local LLM planner.

[![Architecture diagram of xdzy-td/aetheris-aiengine](https://gitdiagram.com/xdzy-td/aetheris-aiengine/diagram.png)](https://gitdiagram.com/xdzy-td/aetheris-aiengine?utm_source=readme&utm_medium=picture)
