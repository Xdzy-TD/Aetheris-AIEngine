<div align="center">

<a href="https://github.com/Xdzy-TD/Aetheris-AIEngine">
  <img src="https://capsule-render.vercel.app/api?type=waving&height=180&color=0:0f172a,100:2563eb&text=AETHERIS&fontColor=ffffff&fontSize=56&fontAlignY=38&desc=Semantic%20Retrieval%20%26%20Multi-Temporal%20Change%20Analysis&descAlignY=62&descSize=18" width="100%">
</a>

<br>

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&size=18&pause=900&color=38BDF8&center=true&vCenter=true&width=800&lines=Earth-Observation+Intelligence+Engine;Natural-Language+Satellite+Image+Analysis;Optical+%2B+SAR+%2B+Multi-Temporal+Workflows;Evidence-Aware+%7C+Auditable+%7C+Offline-First" alt="AETHERIS animated typing">

<br><br>








<br>

<img src="https://skillicons.dev/icons?i=python,fastapi,flask,html,css,js,git,github,docker,linux&perline=10" alt="AETHERIS technology icons">

</div>

🛰️ What is AETHERIS?

AETHERIS is an agentic Earth-observation intelligence engine built for the Smart India Hackathon 2026 — Problem Statement 26227: Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery.

Instead of treating satellite analysis as one fixed model call, AETHERIS turns a natural-language request into an Earth-observation workflow.

A query can be checked for data sufficiency, routed to appropriate specialist tools, executed over optical/SAR or multi-temporal imagery, verified through evidence, and returned with artifacts and provenance.

One Analyst • Many Tools

Natural-language query → observation contract → policy checks → dynamic routing → specialist analysis → evidence verification → auditable result

Core idea

┌─────────────────────┐
│ Natural-Language    │
│ Satellite Query     │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Query / Observation │
│ Contract            │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Quality + Policy    │
│ Gates               │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Agentic / JEV       │
│ Routing             │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Optical / SAR /     │
│ Temporal Specialists│
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Evidence + JEV      │
│ Verification        │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Verified / Qualified│
│ / Abstain + Audit   │
└─────────────────────┘

🎯 SIH Problem Statement

Problem Statement ID: 26227
Title: Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
Theme: Space Technology
Category: Software
Organization: Ministry of Defence (MoD)
Team: Hack & Slay

The SIH proposal describes the gap as fragmentation between visual question answering/captioning, visual grounding, change detection, sensor-specific SAR/EO analysis, and trust/provenance layers.

AETHERIS is designed to bring these workflows together behind a single natural-language analysis interface.

✨ Key Capabilities

Capability

What AETHERIS does

🧠 Agentic Planning

Uses a local Ollama LLM for dynamic tool-chain planning, with deterministic routing as fallback

🔎 Semantic Retrieval

Provides a Geo-RAG specialist path for semantic/geospatial retrieval

🛰️ Optical Analysis

Supports optical imagery workflows and spectral-statistics baselines

📡 SAR Analysis

Includes SAR preprocessing/bridge workflows such as speckle filtering

🕒 Multi-Temporal Analysis

Compares acquisitions and supports bi-temporal change workflows

🔀 Optical + SAR Fusion

Combines independently derived evidence channels

🎯 Visual Grounding

Supports text-conditioned/open-vocabulary grounding through the optional VLM path

👁️ VQA / Captioning

Supports satellite-image question answering/captioning workflows

🧪 Change Detection

Includes classical pixel-difference, embedding-based and learned evidence-fusion paths

🛡️ Policy Validation

Rejects invalid tool plans before execution

🔐 Evidence / Provenance

Stores artifacts with SHA-256 hashes and maintains a tamper-evident audit chain

📜 Evidence Certificates

Produces structured result/provenance information for supported workflows

⚡ JEV Engine

Joint Evidence Verification routing/verification layer for lower-token, fast decisions

🧪 Evaluation Harness

Benchmark/test infrastructure for routing, answerability and analysis behavior

🗺️ Web GUI

Flask interface with map visualization, pipeline status and analysis controls

🎙️ Voice Interface

Browser speech support plus optional offline Whisper/Piper paths

📴 Offline-First AI

Local Ollama inference avoids mandatory external LLM APIs

🧩 Architecture

flowchart TD
    A["User Query + Image(s)"] --> B["AETHERIS API / GUI"]
    B --> C["Query Compiler"]
    C --> D["Observation Contract"]
    D --> E["Data + Quality Gates"]
    E --> F["Policy Engine"]

    F --> G{"Planner"}
    G -->|Local LLM| H["Ollama + Qwen2.5:7b"]
    G -->|Fallback| I["Deterministic Router"]
    G -->|Fast routing| J["JEV Engine"]

    H --> K["Tool Registry"]
    I --> K
    J --> K

    K --> L["Optical / VQA"]
    K --> M["Grounding"]
    K --> N["Change Detection"]
    K --> O["SAR Bridge"]
    K --> P["Optical-SAR Fusion"]
    K --> Q["Geo-RAG"]
    K --> R["VLM Specialists"]

    L --> S["Evidence Layer"]
    M --> S
    N --> S
    O --> S
    P --> S
    Q --> S
    R --> S

    S --> T["Joint Evidence Verification"]
    T --> U["Artifacts + Provenance"]
    U --> V["Audit Chain / SHA-256"]
    T --> W["Verified / Qualified / Abstain"]

🤖 Local AI with Ollama + Qwen

AETHERIS is designed to work with a local Ollama server. The repository's Ollama client prefers:

qwen2.5:7b

other configured Qwen/Llama/Mistral fallbacks when available

The local LLM is used primarily for planning/routing. It does not directly execute tools; the controller validates the generated plan before execution.

This keeps the system suitable for offline/local deployments and reduces dependence on external API services.

1. Install Ollama

Windows

Install Ollama from:

https://ollama.com/download

Then open a new terminal and check:

ollama --version

Start the Ollama server if it is not already running:

ollama serve

Linux

curl -fsSL https://ollama.com/install.sh | sh

Then:

ollama serve

macOS

Install Ollama from:

https://ollama.com/download

Then verify:

ollama --version

2. Pull Qwen 2.5 7B

This is the default model expected by the AETHERIS Ollama client.

ollama pull qwen2.5:7b

Check installed models:

ollama list

Test Qwen directly:

ollama run qwen2.5:7b

Try:

Explain what multi-temporal satellite change detection means.

Exit the interactive model with:

/bye

3. Check Ollama from the terminal

Verify that the server is reachable:

curl http://localhost:11434/api/tags

Or from Python:

python -c "import asyncio; from controller.ollama_client import OllamaClient; print(asyncio.run(OllamaClient().health_check()))"

A successful result should report:

True

🛠️ Installation

Requirements

Python 3.11+

Git

Ollama

Qwen2.5 7B

Recommended: 8–16 GB RAM

Recommended: NVIDIA GPU for heavier VLM workloads

Optional: Rasterio/GDAL for GeoTIFF workflows

Optional: Torch/Transformers for VLM specialists

1. Clone the repository

git clone https://github.com/Xdzy-TD/Aetheris-AIEngine.git
cd Aetheris-AIEngine

If you are running the downloaded project instead:

cd AETHERIS2.0

2. Create a virtual environment

Windows PowerShell

python -m venv .venv
.\.venv\Scripts\Activate.ps1

Windows CMD

python -m venv .venv
.venv\Scripts\activate

Linux / macOS

python3 -m venv .venv
source .venv/bin/activate

3. Upgrade pip

python -m pip install --upgrade pip

4. Install AETHERIS

For the complete development setup:

pip install -e ".[all]"

For the base installation:

pip install -e .

For VLM support:

pip install -e ".[vlm]"

For Geo-RAG:

pip install -e ".[geo,geo_rag]"

For development/testing:

pip install -e ".[dev]"

Everything

pip install -e ".[all,vlm]"

⚙️ Environment Configuration

Create your local environment file:

cp .env.example .env

Windows PowerShell:

Copy-Item .env.example .env

For local Qwen inference, make sure the runtime configuration points to Ollama.

The Ollama client itself reads:

OLLAMA_BASE_URL=http://localhost:11434
AETHERIS_MODEL=qwen2.5:7b

Important: the current controller/ollama_client.py uses OLLAMA_BASE_URL and AETHERIS_MODEL. If your .env.example contains older AETHERIS_OLLAMA_URL naming, use the variable names above for the Ollama client.

Example:

OLLAMA_BASE_URL=http://localhost:11434
AETHERIS_MODEL=qwen2.5:7b
AETHERIS_USE_JEV=true
AETHERIS_NO_LLM=0
AETHERIS_GUI_HOST=0.0.0.0
AETHERIS_GUI_PORT=5000
AETHERIS_GUI_DEBUG=false

🚀 Running AETHERIS

Option A — Web GUI

Start the GUI:

python -m pipeline.run gui

Then open:

http://localhost:5000

The GUI provides the web interface for the AETHERIS pipeline, including analysis controls, status information, map visualization and supported voice features.

Option B — Health Check

Before a demo, run:

python -m pipeline.run health

This checks the M01 health layer and prints a JSON result.

Option C — CLI Satellite Analysis

Run the pipeline directly:

python -m pipeline.run pipeline

Use your own question:

python -m pipeline.run pipeline -q "What land cover types are visible in this satellite image?"

Specify modality:

python -m pipeline.run pipeline \
  -q "Identify significant changes between the observations." \
  --modality optical

Option D — Inspect the Pipeline DAG

python -m pipeline.run dag

This prints the current pipeline/DAG state.

🧠 Run with Qwen + Ollama

Start Ollama:

ollama serve

Make sure Qwen is installed:

ollama pull qwen2.5:7b

Set the model:

PowerShell

$env:AETHERIS_MODEL="qwen2.5:7b"
$env:OLLAMA_BASE_URL="http://localhost:11434"
$env:AETHERIS_NO_LLM="0"

Linux / macOS

export AETHERIS_MODEL=qwen2.5:7b
export OLLAMA_BASE_URL=http://localhost:11434
export AETHERIS_NO_LLM=0

Then launch:

python -m pipeline.run gui

🔥 Recommended Demo Startup

Open Terminal 1:

ollama serve

Open Terminal 2:

cd Aetheris-AIEngine

# activate environment
# Windows:
.venv\Scripts\activate

# Linux/macOS:
# source .venv/bin/activate

ollama pull qwen2.5:7b

python -m pipeline.run health
python -m pipeline.run gui

Open:

http://localhost:5000

Now AETHERIS can use the local Qwen model for planning when LLM routing is enabled.

🔁 LLM Planner vs JEV

AETHERIS contains two complementary planning paths:

                         USER QUERY
                             │
                  ┌──────────┴──────────┐
                  │                     │
             JEV Engine            Ollama + Qwen
             Fast routing          LLM planning
                  │                     │
                  └──────────┬──────────┘
                             ↓
                      Policy Validation
                             ↓
                      Tool Execution

JEV

The JEV layer is designed for fast typed routing/verification decisions and lower token usage.

Qwen + Ollama

Qwen provides the local language-model planning layer for more flexible natural-language tool-chain planning.

Safety boundary

The LLM plans; the controller/policy layer decides whether that plan is valid before execution.

🛰️ Supported Analysis Paths

Optical

Optical Image
     ↓
Metadata / Quality Gate
     ↓
Optical Specialist
     ↓
Grounding / VQA / Change Analysis

SAR

SAR Image
   ↓
SAR Bridge / Preprocessing
   ↓
Backscatter Analysis
   ↓
Change / Fusion Workflow

Multi-Temporal

T1 Image ───────┐
                ├──→ Registration → Difference → Evidence
T2 Image ───────┘

Optical + SAR

Optical ──→ Optical Evidence ──┐
                               ├──→ Evidence Fusion
SAR ──────→ SAR Evidence ──────┘

🔐 Evidence, Provenance & Auditability

AETHERIS is not designed to stop at a model confidence number.

The pipeline includes:

Artifact persistence

SHA-256 verification

Model/version provenance

Hash-chained audit logging

Policy validation

Evidence artifacts

Structured result contracts

Verification workflows

The audit chain is tamper-evident, not tamper-proof.

This gives an analysis result a traceable path from:

Query
  ↓
Plan
  ↓
Policy Decision
  ↓
Tool
  ↓
Model / Method
  ↓
Artifact
  ↓
Evidence
  ↓
Final Result

📁 Project Structure

Aetheris-AIEngine/
│
├── benchmark/                 # Benchmark runner + sample dataset
├── confidence/                # Confidence / conformal components
│
├── controller/                # Core orchestration
│   ├── llm_planner.py         # LLM-driven planning
│   ├── ollama_client.py       # Local Ollama integration
│   ├── planner.py             # Planning + fallback routing
│   ├── policy.py              # Plan validation
│   ├── jev_engine.py          # JEV verification/routing
│   ├── artifacts.py           # Artifact storage + SHA-256
│   ├── audit_log.py            # Hash-chain audit log
│   └── tool_registry.py       # Specialist registry
│
├── interfaces/
│   ├── api/                   # FastAPI service
│   └── voice/                 # STT / TTS
│
├── pipeline/
│   ├── gui/                   # Flask web GUI
│   ├── modules/               # M01–M20 pipeline modules
│   ├── services/              # Pipeline orchestration
│   ├── schemas/               # Typed contracts
│   └── run.py                 # CLI entry point
│
├── preprocessing/             # Registration / preprocessing
├── specialists/               # EO analysis specialists
├── training/                  # LoRA / calibration / fusion scripts
├── evaluation/                # Evaluation utilities
├── models/                    # Model registry
├── tests/                     # Test suite
├── docs/                      # Architecture / Ollama / status docs
│
├── .env.example
├── pyproject.toml
└── README.md

🌐 API

The primary service is FastAPI.

Run the API with Uvicorn:

uvicorn interfaces.api.app:app --host 0.0.0.0 --port 8000

Open API documentation:

http://localhost:8000/docs

Useful endpoints include:

GET  /v1/health
GET  /v1/tools
GET  /v1/models
GET  /v1/missions
GET  /v1/audit/verify
GET  /metrics
POST /v1/query
POST /v1/mission/{name}

🧪 Testing

Run the full test suite:

pytest

Run with verbose output:

pytest -v

Run the repository harness on Linux/macOS:

./harness/run.sh

Windows PowerShell:

.\harness\run.ps1

📊 Benchmarking

AETHERIS includes a benchmark harness under:

benchmark/

Run the benchmark using the repository's benchmark tooling where configured.

The shipped sample dataset is primarily a pipeline/refusal-path fixture. A real capability score requires a labelled dataset containing actual image paths and ground-truth answers.

🧠 Optional VLM Stack

The repository contains optional VLM specialists based on Hugging Face/Transformers.

Install:

pip install -e ".[vlm]"

These paths are separate from the default CPU/offline installation.

The repository status explicitly distinguishes pretrained VLM inference from an RS-adapted model. No remote-sensing-trained checkpoint should be claimed simply because the VLM dependency is installed.

🗺️ GeoTIFF / Geospatial Support

For geospatial workflows:

pip install -e ".[geo]"

Geo-RAG:

pip install -e ".[geo_rag]"

Geo-RAG adds:

Qdrant client support

Sentence-transformers

Semantic embedding/retrieval workflow

Georeferenced raster handling preserves metadata such as:

CRS

transform

resolution

bounds

band information

NoData

dtype

🐳 Docker

A cloud-demo deployment is included under:

deploy/cloud_demo/

Build:

docker compose -f deploy/cloud_demo/docker-compose.yml build

Run:

docker compose -f deploy/cloud_demo/docker-compose.yml up

🧭 Mission Workflows

AETHERIS includes mission-oriented workflows for use cases such as:

Flood / inundation

Urban expansion

Crop change

Disaster assessment

Water monitoring

Wildfire / burn severity

Landslide detection

Deforestation

Coastal erosion

Infrastructure development

Mining / industrial expansion

Road-network change

The repository also contains a custom mission generator for decomposing natural-language objectives into supported analysis calls.

⚠️ Current Implementation Reality

AETHERIS deliberately separates implemented functionality from research roadmap items.

The repository currently includes real implementations for the agentic controller, policy validation, GeoTIFF handling, artifact verification, audit logging, benchmark infrastructure, optional pretrained VLM paths, mission workflows, and learned evidence fusion.

However, some advanced research capabilities are not shipped as production-ready trained models.

In particular:

No RS-adapted VLM checkpoint ships with the repository.

The CLIP/CLIPSeg VLM paths are not automatically equivalent to an RS-specialized model.

The conformal calibration layer ships unfitted.

The learned fusion model ships unfitted.

A benchmark capability score should not be claimed from the metadata-only sample dataset.

SAM/SAM2 and other future RS-specialized models are roadmap items unless separately integrated and evaluated.

This distinction is intentional: AETHERIS reports what the code actually supports rather than presenting planned research as completed functionality.

🛣️ Research Roadmap

Future extensions can include:

RS-adapted VLM
      ↓
Better Grounding
      ↓
RS-specific Change-VQA
      ↓
Bi-temporal Transformer
      ↓
Calibrated Trust Layer
      ↓
Indian EO / ISRO-oriented Fine-tuning
      ↓
Larger Evaluation Benchmarks

The repository also identifies future research directions around:

LoRA / QLoRA on Indian EO data

SAM2 / RS text-prompt models

Bi-temporal transformer + Change-VQA

Calibrated confidence

zkML

Zero-knowledge proofs

Post-quantum cryptography

Distributed/tamper-proof audit anchoring

Secure multi-party computation

💡 Why AETHERIS?

Traditional satellite-analysis workflows often require an operator to manually select different tools for different questions.

AETHERIS instead aims to provide:

Ask naturally
     ↓
Understand the observation requirement
     ↓
Check whether the data can answer it
     ↓
Select the appropriate tools
     ↓
Analyze the imagery
     ↓
Cross-check evidence
     ↓
Return a qualified result
     ↓
Preserve provenance

The central design principle is:

Move from model confidence toward physical answerability and verifiable evidence.

🏗️ Development

Install development dependencies:

pip install -e ".[dev]"

Run formatting/linting:

ruff check .

Run tests:

pytest -v

Type checking:

mypy .

🧾 License

AETHERIS is released under the Apache License 2.0.

See LICENSE.

<div align="center">

🛰️ AETHERIS

Semantic Retrieval • Multi-Temporal Change Analysis • Evidence-Aware Earth Observation

<br>

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&size=15&pause=1200&color=64748B&center=true&vCenter=true&width=700&lines=Built+for+Earth+Observation.;Designed+for+Evidence.;Powered+by+Local+AI.;Optical+%2B+SAR+%2B+Temporal+Analysis." alt="AETHERIS footer animation">

<br><br>

Hack & Slay • Smart India Hackathon 2026 • PS 26227

</div>
