# Implementation Status

One rule: nothing appears under **Implemented** unless the code that does it
runs in this repository. Everything a reviewer might assume is a trained
neural network, but isn't, is named as a classical baseline here.

## Implemented

- **Agentic controller** — LLM planner (Ollama) with rule-based fallback routing;
  the LLM only *plans*, it never executes.
- **Policy engine** (`controller/policy.py`) — validates every plan against the
  tool registry and the actual inputs *before* execution: unknown tools, tools
  with no bound model, insufficient images, wrong sensor combinations, forward
  or self dependencies, and parameters outside the tool's declared JSON schema.
  A rejected plan returns `Analysis unavailable — <reason>` and no confidence.
- **GeoTIFF pipeline** (`specialists/_imagery.py`) — rasterio/GDAL is the native
  reader. CRS, transform, resolution, bounds, band count/names, NoData and dtype
  are read at the upload boundary and carried on the query. A georeferenced
  raster is **refused** when rasterio is absent rather than quietly read through
  Pillow with its CRS discarded. Every specialist reads pixels through
  `load_raster`/`load_pil`, so there is no side door back to a bare
  `Image.open`. Rasters deeper than 8-bit are refused by `open_checked` for the
  same reason. Known limitation: with rasterio absent, a `.tif` that *is*
  georeferenced is refused at read time, but `probe_raster_metadata` reports
  `georeferenced: false` at upload time — it cannot tell without a reader.
- **Artifact store** (`controller/artifacts.py`) — every mask, change map,
  despeckled raster and fusion map is persisted outside `/tmp` with
  `artifact_id`, `sha256`, `type` and `model_version`.
  `GET /artifacts/{artifact_id}` re-hashes the bytes on read and returns
  `sha256_matches`, so an artifact cited by a report can be checked rather
  than trusted.
- **Tamper-evident audit log** — hash-chained trace; editing any past entry
  invalidates every later `chain_hash`. Tamper-*evident*, not tamper-proof.
- **Model registry** (`models/registry.json`, `GET /models`) — every task's
  backing model with its `kind`, licence and metrics. `metrics` are empty
  because no evaluation run has produced any.
- **Benchmark harness** (`benchmark/`) — runs a labeled JSONL dataset through
  the planner end-to-end and reports routing/answer accuracy and latency.
  `benchmark/sample_dataset.jsonl` ships with metadata-only images, so it
  only exercises the refusal path above, not answer content. Now gated in
  `harness/run.sh` (`benchmark` stage, part of `all`) and therefore in CI, so
  the refusal-guard regression above is checked on every run, written to
  `benchmark/reports/latest.json`. See *Not implemented* below — this does
  not produce a capability score.
- **Pretrained VLM specialist** (`vlm_vqa_caption`, `specialists/vlm_caption/`) —
  real BLIP inference via `transformers`, genuine weights and forward pass.
  Optional dependency (`pip install -e ".[vlm]"`, ~2GB, off by default to keep
  the CPU-only/offline default intact); refuses via `analysis_unavailable`
  when torch/transformers aren't installed, same as rasterio-absent GeoTIFFs.
  Confidence is the model's own mean top-token generation probability, not a
  fabricated score. **Not RS-adapted by default** — optionally loads a LoRA
  adapter via `AETHERIS_VLM_LORA_PATH`; none ships — see below. Every result
  carries `model_id` and `lora_applied`, read off the running instance rather
  than assumed from the registry entry, so an `AETHERIS_VLM_MODEL_ID`/
  `AETHERIS_VLM_LORA_PATH` override — or a configured LoRA path that didn't
  actually resolve to a real adapter directory — is visible per-call, not
  just in `models/registry.json`.
- **Pretrained open-vocabulary grounding** (`vlm_grounding`,
  `specialists/vlm_grounding/`) — real CLIPSeg inference via `transformers`,
  genuine weights and forward pass, replacing keyword→class lookup with an
  actual text-conditioned segmentation model. Same optional dependency and
  `analysis_unavailable` refusal as `vlm_vqa_caption`. Confidence is the
  model's own mean sigmoid probability inside the predicted mask. **Not
  RS-adapted by default** — optionally loads a LoRA adapter via
  `AETHERIS_GROUNDING_LORA_PATH`; none ships — and **not SAM2/RSPrompter**
  — see below. Same per-call `model_id`/`lora_applied` provenance as
  `vlm_vqa_caption`, for the same reason.
- **Pretrained-embedding change detection** (`vlm_change_detection`,
  `specialists/vlm_change_detection/`) — real CLIP vision-transformer
  inference via `transformers`; change is the cosine distance between
  corresponding patch embeddings of the two acquisitions, a genuine model
  signal in place of raw pixel differencing. Same optional dependency and
  refusal pattern. **Not a learned bi-temporal change model** — CLIP was
  never trained to detect change or on remote sensing, so distances are a
  real signal but an uncalibrated one. Every result carries `model_id`
  (no LoRA field — this specialist has no adapter path).
- **Confidence plumbing** — split-conformal calibrator with `save()`/`load()`
  for the calibration artifact. Ships **unfitted**, and therefore reports
  `calibrated: false` and `method: "uncalibrated"`, passing the raw specialist
  score through untouched. See *Not implemented* below.
- **Mission-oriented workflows** (`controller/missions.py`) — twelve named
  query presets covering flood/inundation, urban expansion, crop change,
  disaster assessment, water monitoring, wildfire/burn severity, landslide
  detection, deforestation, coastal erosion, infrastructure development,
  mining/industrial expansion, and road-network change, plus a Custom
  Mission Generator (`custom_mission_generator`) that decomposes a free-text
  objective (e.g. "vegetation decreased and built-up increased") into one
  `grounding` call per detected land-cover concept. No new specialists: every
  preset pre-fills a default question and inserts `grounding` call(s) (with
  mission keywords) before the existing deterministic fallback router's final `vqa_caption`
  step, reusing the same water/vegetation/urban/bare/coast keyword classes
  already in `specialists/_imagery.py` and the same routing rules
  (`AgenticPlanner._plan_rule_based`) every other query goes through. Skipped for
  an all-SAR query, since the grounding baseline's spectral thresholds are
  optical-band math and would be a misapplied baseline on SAR amplitude, not
  a real capability. Landslide detection is scoped to optical/SAR imagery
  only — this deployment has no DEM/terrain-slope ingestion, so slope is not
  assessed and the mission's question never implies it. Run via
  `python run.py mission <name> --image <path>`, `POST /v1/mission/{name}`
  (same upload handling as `/v1/query`, reused via a shared helper rather
  than duplicated), or discover the full catalog via `GET /v1/missions`.
- **Benchmark baseline comparison** (`benchmark/runner.py:run_comparison`) —
  for each dataset item's task family (VQA, grounding, change detection),
  calls every *bound* alternative model directly — classical baseline vs.
  pretrained/learned, see `models/registry.json` — and scores each the same
  way as `run_benchmark`, producing one accuracy/latency row per model
  instead of one row per query. An unbound optional model (`vlm_*` without
  `torch`/`transformers`) is skipped, not scored `0`. Run via
  `python run.py benchmark --dataset <path> --compare`. Same caveat as
  `run_benchmark` above: `sample_dataset.jsonl` ships with no real pixels, so
  this only proves the comparison mechanism runs, not that any model's
  accuracy is real — point it at a labelled dataset with real image paths
  for that.
- **Learned multimodal evidence fusion for change detection**
  (`fusion_change_detection`, `specialists/fusion_change_detection/`) — a
  genuinely *learned* combiner (`specialists._fusion_model.EvidenceFusionModel`,
  logistic regression fit by gradient descent) over three independently
  computed evidence channels: optical bi-temporal pixel difference, SAR
  bi-temporal backscatter difference (Lee-despeckled, same filter as
  `sar_bridge`), and elapsed time between acquisitions. Any channel that
  isn't available (no SAR pair, no dates) is passed through as neutral
  (0.5) evidence rather than silently zeroed, so its absence cannot bias the
  fused score toward "no change" — see `tests/test_specialists.py` for the
  regression guard on this. Unlike `change_detection` (thresholds raw pixel
  difference) or `vlm_change_detection` (thresholds raw CLIP embedding
  distance), the decision itself is a trained function of multiple signals,
  not a fixed cutoff on one. Ships **unfitted** — see below. Each evidence
  channel is co-registered before differencing, with the same implausible-shift
  guard as `change_detection` (`specialists._imagery.validate_shift`): a
  rejected shift falls back to `dy=dx=0` rather than applying a false
  phase-correlation peak. Reliability is reported per channel via
  `coregistration_reliable: {"optical": ..., "sar": ...}` (`sar` is `null`
  when no SAR pair was supplied).
- **Flask web GUI** (`pipeline/gui/app.py`, `run_gui.py`) — a
  Flask-based web dashboard with real-time module status, pipeline execution,
  voice dictation (Web Speech API + faster-whisper fallback), a Leaflet world
  map with Esri/OSM tiles and Nominatim search, and results visualization.
  Run via `python run_gui.py` or `python -m pipeline.run gui`.
  Serves on port 5000 by default; optional HTTPS via `AETHERIS_GUI_SSL_*`.

## Classical baseline (explainable, not learned)

These are real signal processing over real pixels — not placeholders, and not trained
models. Every one is labelled as such in `models/registry.json`.

| Task | Method |
|---|---|
| VQA / caption | Spectral-index thresholds (excess-green, blue-dominance, brightness/saturation) |
| Grounding | Keyword → land-cover class mask from the same indices |
| Change detection | FFT phase-correlation co-registration, then pixel differencing |
| SAR bridge | Lee / refined-Lee speckle filtering |
| Optical–SAR fusion | FFT phase-correlation co-registration (native resolution), then cross-modal agreement across water/vegetation/built-up between optical indices and SAR backscatter percentile masks |
| Geo RAG | Text embeddings over an indexed corpus (in-memory NumPy fallback) |

## Not implemented / planned

Named plainly because a reviewer will ask:

- **No RS-adapted VLM checkpoint ships, but the LoRA path now exists.**
  `training/finetune_vqa_lora.py` is a real, runnable LoRA/QLoRA fine-tuning
  script for `vlm_vqa_caption`'s BLIP backbone on a real remote-sensing
  VQA/caption dataset (default: RSVQA-LR), and
  `specialists/vlm_caption/model.py` will load its output via
  `AETHERIS_VLM_LORA_PATH`. Nobody has run it against this repo: it needs a
  GPU and Hugging Face Hub access, neither available in this project's
  offline/CPU-only sandbox. So — same as before — there is no checkpoint
  trained on satellite/aerial imagery and no evaluation showing reliability
  on remote sensing; treat `vlm_vqa_caption`'s answers on overhead imagery as
  out-of-distribution guesses until someone actually runs the script.
  `vqa_caption` (the classical baseline) remains the default route for
  domain answers grounded in real pixel statistics.
- **No SAM/SAM2, but a real RS-adaptation path now exists for CLIPSeg.**
  `vlm_grounding` runs real open-vocabulary segmentation (CLIPSeg, not
  RS-adapted by default); `training/finetune_grounding_lora.py` LoRA-adapts
  it following the published CLIPSeg-on-FLAIR recipe (Garioud et al., ISPRS
  J. Photogramm. 2025), and `specialists/vlm_grounding/model.py` will load
  the result via `AETHERIS_GROUNDING_LORA_PATH`. Same caveat as above: not
  run here, no GPU/Hub access in this sandbox, no adapter ships.
  `grounding` (keyword-to-class) remains the classical default route.
- **No trained Change-VQA model, but structured change narratives are now
  synthesised.**  `change_narrative` on `ExecutionResult` composes a
  human-readable description of *what* changed by combining structured
  change-detection outputs (change fraction, change summary, co-registration
  reliability) with VQA/caption scene context. This is rule-based prose
  synthesis from tool outputs, not a learned Change-VQA model — see
  `controller/planner.py:_synthesize_change_narrative`.
- **No fitted change-detection or fusion checkpoint ships, but a real
  learned-combiner path now exists for both gaps at once.**
  `vlm_change_detection` adds a real pretrained-embedding distance signal
  (CLIP, not RS-adapted, not trained for change); `fusion_change_detection`
  goes further, combining optical, SAR and temporal evidence through
  `specialists._fusion_model.EvidenceFusionModel`, whose weights are
  *learned* by gradient descent (`training/fit_fusion_model.py`) rather than
  a hand-set threshold — closing "no model trained to detect change" and "no
  learned optical/SAR fusion" together, since a learned bi-temporal model and
  learned multimodal fusion are the same trained combiner. Same caveat as
  the LoRA adapters: nobody has run the fitting script against this repo —
  it needs internet access to pull a labelled dataset (default: OSCD via
  `blanchon/OSCD_MSI`), unavailable in this project's offline sandbox — so
  `fusion_change_detection` ships with an informative-prior, **unfitted**
  model (`fusion_model_fitted: false`) and no coverage or accuracy claim is
  made for it. An offline fallback, `training/fit_fusion_model_synthetic.py`,
  fits the same `EvidenceFusionModel` through the same feature pipeline on
  locally generated synthetic bi-temporal pairs, for sandboxes without
  Hugging Face Hub access — it produces a genuinely fitted (not hardcoded)
  checkpoint, but its docstring is explicit that this is a "these three
  channels are separable" sanity check, not a real-world change-detection
  accuracy claim, and it is not a substitute for `fit_fusion_model.py` once
  real data is reachable. `change_detection` (pixel differencing) remains
  the classical default route.
- **No calibrated confidence.** The calibrator is implemented but unfitted:
  there is no labelled calibration set in this repo, so no coverage guarantee
  is claimed. `calibrated: false` is returned, deliberately.
- **No benchmark numbers.** The harness above can run one, but no labelled
  satellite-imagery dataset ships with this repo, so `metrics` is empty
  everywhere and no evaluation has produced a real figure. Any accuracy
  quoted for AETHERIS today would be invented.

## Deliberate refusals

These are not bugs:

- **A query with no image is refused.** Every route ends at `vqa_caption`, which
  needs pixels; `Analysis unavailable — 'vqa_caption' needs 1 readable image(s);
  0 were provided.` is the correct answer to a question about an image that was
  never supplied. `python run.py demo` hits this, because its demo query carries
  image *metadata* with no file behind it.
- **A SAR/optical pair is refused for change detection** and routed to
  `optical_sar_fusion` instead: differencing across sensors measures sensor
  physics, not ground change.

## Failure behaviour

When a tool cannot run, the result is
`{"analysis_unavailable": true, "reason": ..., "model_version": ...}`.
It carries no output paths and no numeric statistics, so it cannot be mistaken
for a measurement. The previous placeholder responses — which returned invented
summaries, zeroed statistics and `/tmp` paths that never existed — are gone.
