<div align="center">

🛰️ AETHERIS

Semantic Retrieval & Multi-Temporal Change Analysis of Satellite Imagery

Agentic Earth-Observation Intelligence Engine

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&size=17&pause=900&color=38BDF8&center=true&vCenter=true&width=760&lines=Natural-language+satellite+analysis;Optical+%2B+SAR+%2B+multi-temporal+workflows;Evidence-aware+%7C+Auditable+%7C+Local+AI" alt="AETHERIS animation">







<img src="https://skillicons.dev/icons?i=python,fastapi,flask,html,css,js,git,github,docker,linux&perline=10" alt="Tech stack icons">

</div>

🎯 About

AETHERIS is an agentic satellite-imagery analysis engine for Smart India Hackathon 2026 — PS 26227: Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery.

Instead of sending every request to one model, AETHERIS converts a natural-language question into an Earth-observation workflow, checks requirements, selects appropriate analysis tools, verifies evidence, and returns a traceable result.

Core flow:

Natural-language query → Observation contract → Quality/Policy checks → Routing → Specialist analysis → Evidence verification → Result + provenance

✨ Key Features

🧠 Agentic planning with local Ollama/Qwen

🔎 Semantic / Geo-RAG retrieval

🛰️ Optical imagery analysis

📡 SAR analysis and preprocessing

🕒 Multi-temporal change analysis

🔀 Optical + SAR evidence fusion

🎯 VQA / visual grounding paths

⚡ JEV Engine for fast typed routing/verification

🛡️ Policy validation before tool execution

🔐 SHA-256 artifact verification + hash-chained audit logs

📜 Evidence-aware result contracts

🌐 Web GUI + FastAPI API

📴 Local/offline AI through Ollama

🧱 Architecture

[![Architecture diagram](https://gitdiagram.com/diagram-badge.svg)](https://gitdiagram.com/xdzy-td/aetheris-aiengine?utm_source=readme&utm_medium=badge)

[![Architecture diagram of xdzy-td/aetheris-aiengine](https://gitdiagram.com/xdzy-td/aetheris-aiengine/diagram.png)](https://gitdiagram.com/xdzy-td/aetheris-aiengine?utm_source=readme&utm_medium=picture)

🚀 Quick Start

1. Clone

git clone https://github.com/Xdzy-TD/Aetheris-AIEngine.git
cd Aetheris-AIEngine

2. Create virtual environment

Windows PowerShell

python -m venv .venv
.\.venv\Scripts\Activate.ps1

Linux / macOS

python3 -m venv .venv
source .venv/bin/activate

3. Install

python -m pip install --upgrade pip
pip install -e .

For the complete optional stack:

pip install -e ".[all]"

For VLM support:

pip install -e ".[vlm]"

🤖 Ollama + Qwen 2.5 7B

AETHERIS uses a local Ollama server for LLM-based planning. The client defaults to qwen2.5:7b.

Install Ollama

Download Ollama from:

https://ollama.com/download

Verify:

ollama --version

Start Ollama

ollama serve

Pull Qwen

ollama pull qwen2.5:7b

Check:

ollama list

Test it:

ollama run qwen2.5:7b

Configure AETHERIS

The current Ollama client reads these variables:

OLLAMA_BASE_URL=http://localhost:11434
AETHERIS_MODEL=qwen2.5:7b

If you want to force LLM planning from the terminal:

PowerShell

$env:OLLAMA_BASE_URL="http://localhost:11434"
$env:AETHERIS_MODEL="qwen2.5:7b"

Linux / macOS

export OLLAMA_BASE_URL=http://localhost:11434
export AETHERIS_MODEL=qwen2.5:7b

Check Ollama:

curl http://localhost:11434/api/tags

▶️ Run AETHERIS

Web GUI

python -m pipeline.run gui

Open:

http://localhost:5000

Health check

python -m pipeline.run health

CLI pipeline

python -m pipeline.run pipeline

Custom question:

python -m pipeline.run pipeline -q "What land cover types are visible in this satellite image?"

Specify modality:

python -m pipeline.run pipeline -q "Identify changes between the observations." --modality optical

Show pipeline DAG

python -m pipeline.run dag

⚡ Recommended Demo Setup

Terminal 1 — Ollama

ollama serve

Terminal 2 — AETHERIS

cd Aetheris-AIEngine
# activate .venv first
ollama pull qwen2.5:7b
python -m pipeline.run health
python -m pipeline.run gui

Then open http://localhost:5000.

🌐 API

Start the FastAPI service:

uvicorn interfaces.api.app:app --host 0.0.0.0 --port 8000

API docs:

http://localhost:8000/docs

🧪 Testing

pytest -v

Lint:

ruff check .

📁 Project Structure

Aetheris-AIEngine/
├── controller/       # Planner, Ollama, JEV, policy, audit, tools
├── specialists/      # EO analysis specialists
├── preprocessing/    # Image preprocessing / registration
├── pipeline/         # M01–M20 pipeline + GUI + CLI
├── interfaces/       # FastAPI + interfaces
├── benchmark/        # Benchmark runner / dataset
├── evaluation/       # Evaluation utilities
├── training/         # Fine-tuning / calibration utilities
├── confidence/       # Confidence components
├── models/           # Model registry
├── tests/             # Test suite
├── docs/              # Architecture / setup / status
├── pyproject.toml
└── README.md

🛰️ Analysis Stack

Optical → preprocessing → VQA / grounding / change analysis

SAR → preprocessing → SAR evidence → change / fusion

Multi-temporal → acquisition comparison → change evidence → verification

Semantic retrieval → query understanding → geospatial retrieval → evidence

🔐 Evidence & Audit

AETHERIS tracks analysis artifacts and provenance through:

SHA-256 artifact verification

Hash-chained audit logging

Policy validation

Structured result contracts

Evidence verification

The system is designed to distinguish Verified, Qualified, and Abstain outcomes instead of treating every model response as equally reliable.

🛣️ Roadmap

Remote-sensing-adapted VLMs

Better open-vocabulary grounding

Bi-temporal Transformer / Change-VQA

Calibrated confidence

Indian EO dataset fine-tuning

SAM2 / RS-specific segmentation

Larger real-world evaluation benchmarks

These are roadmap items unless separately integrated and evaluated in the repository.

📜 License

Apache-2.0

<div align="center">

🛰️ AETHERIS

Semantic Retrieval • Multi-Temporal Change Analysis • Evidence-Aware Earth Observation

Hack & Slay • SIH 2026 • PS 26227

</div>
