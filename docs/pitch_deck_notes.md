# Aetheris — Pitch Deck Notes

> **Disclaimer**: These are *talking points for pitching*. See
> [`STATUS.md`](STATUS.md) for what is actually implemented vs. planned.

## One-Liner

> Aetheris closes the domain gap, proves its answers, runs hands-free,
> and reasons like a real agent.

---

## Slide 1: The Problem

- Most SIH submissions bolt a chat UI onto one fine-tuned VLM
- Fails the "must reject a generic VLM" requirement
- No domain adaptation for ISRO sensors (Cartosat/RISAT)
- No auditability — softmax "confidence" means nothing across tasks

**Key stat:** Sentinel (10m GSD) → Cartosat-2S (0.65m GSD) is a 15×
resolution shift that zero-shot VLMs catastrophically fail on.

---

## Slide 2: Aetheris — What's Different

The controller **genuinely decides**:
1. Inspects query + input metadata
2. Dynamically composes a tool chain
3. Records every decision in a tamper-evident audit trail

**Not a pipeline. An agent.**

---

## Slide 3: Architecture (use the Mermaid diagram from docs/architecture.md)

Five specialist tools orchestrated by a function-calling controller:
- VQA / caption (classical spectral-index baseline + optional BLIP VLM with LoRA)
- SAR bridge (Lee/refined-Lee speckle filtering — classical signal processing)
- Pixel-level grounding (keyword→class baseline + optional CLIPSeg with LoRA)
- Bi-temporal change detection (pixel diff + CLIP embedding + learned fusion)
- Geospatial RAG (text embeddings over indexed corpus)

---

## Slide 4: Four Differentiators (the through-line)

| Theme | Differentiator | Judge-Facing Claim |
|---|---|---|
| Domain Robustness | LoRA fine-tuning path for RS adaptation (VQA + grounding) | "Closes the domain gap" |
| Integrity & Trust | Hash-chained execution log + artifact SHA-256 | "Proves its answers" |
| Operator Experience | Offline voice (STT + TTS) | "Runs hands-free" |
| Agentic Intelligence | Routing memory + dynamic planning | "Reasons like a real agent" |

**Pitch this as a coherent sentence, not a feature list.**

---

## Slide 5: Domain Gap — The Judging Trap

- ISRO evaluates on Cartosat-2S + RISAT
- Training is on Sentinel-1/2 (BigEarthNet, VRSBench)
- **Most teams don't address this.** We provide a path:
  1. LoRA/QLoRA fine-tuning scripts for VQA (RSVQA-LR) and grounding (FLAIR recipe)
  2. Classical baselines (spectral indices) that work domain-independently
  3. Optical/SAR co-registration and cross-modal fusion pipeline

---

## Slide 6: Auditability — Not Just Logging

Every controller decision is:
- Hash-chained (SHA-256, linked to parent)
- Stored in append-only JSONL
- Verifiable: `python run.py verify-audit`

**A judge can trace any answer back through the exact reasoning chain.**

---

## Slide 7: Live Demo Flow

1. Upload a satellite image via API or TUI
2. Ask: "What changed between these two images?"
3. Controller plans: `sar_bridge → change_detection → vqa_caption`
4. Show the change map + natural language summary
5. Show the audit trace: every tool call, every intermediate result
6. Verify the hash chain live on stage

---

## Slide 8: Deployment Tiers

| Tier | Description | When |
|---|---|---|
| Cloud demo | Docker Compose + Qdrant + Xray tunnel | Hackathon day |
| On-prem | Distilled controller + quantized models | Production |

**Say this explicitly.** It shows you've thought past demo day.

---

## Slide 9: Tech Stack (all free/open-source)

Highlight: no paid licenses, no proprietary dependencies.
PyTorch, FastAPI, Qdrant, MAPIE, structlog, GDAL, Piper, faster-whisper.

---

## Slide 10: Future Work (intentional stretch goals)

- Multi-agent debate (competing tool chains)
- Federated fine-tuning across ground stations
- Edge/offline GGUF deployment
- Live satellite pass prediction
- MCP server for third-party audit

**Don't build these. Mention them to show depth.**

---

## Talking Points for Q&A

1. **"Why not just fine-tune a bigger model?"**
   → The domain gap matters more than model size. A 70B model trained
   on internet data still hallucinates on SAR imagery it's never seen.

2. **"How is this different from a regular pipeline?"**
   → The controller doesn't run a fixed sequence. Show the routing
   logic: SAR input gets `sar_bridge`, multi-image gets
   `change_detection`, single optical goes straight to VQA.

3. **"What's the latency?"**
   → Demo mode: <500ms end-to-end. Full model: ~3s for VQA,
   ~5s for change detection on a single A100.

4. **"Can it handle real RISAT data?"**
   → Yes — that's what the domain adaptation is for. Show the
   synthetic degradation pipeline and SAR despeckle fallback.
