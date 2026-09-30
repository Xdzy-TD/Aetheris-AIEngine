# Restoration notes

The uploaded ZIP had all 137 files dumped into a single flat folder. This
tree was reconstructed by reading every `import` statement, docstring, and
internal `Path(__file__)` calculation across all 137 files, cross-checked
against `pyproject.toml`'s `[tool.setuptools.packages.find]` list,
`docs/architecture.md`, `docs/STATUS.md`, the `Dockerfile`, and the run
scripts. A static check confirms every internal absolute import in the
restored tree now resolves to a real file **except** the ones listed below
under "Genuinely missing."

## Genuinely missing (not recoverable from the ZIP)

Flattening a nested tree into one folder silently overwrites any files that
share a basename — only the last one written survives. That happened here.
The following are referenced by working code (confirmed via
`interfaces/api/deps.py:_bind_specialists`, which lists the exact
module path + class name for every specialist) but do not exist anywhere in
the archive:

- **Root CLI entry point** (`run.py`, with `serve`/`demo`/`benchmark`/
  `verify-audit` subcommands, per `run.sh`/`run.bat`/`run.ps1`). Only
  `pipeline/run.py` (a *different* file, with `gui`/`health`/`pipeline`/`dag`
  subcommands) survived the collision. `pyproject.toml`'s `aetheris` script
  entry now points at `pipeline.run:main` so it references something real,
  but this is not the original entry point's behavior — you'll want to
  rewrite it.
- **`run_gui.py`** — the thin launcher `run_gui.bat` calls.
- **`pipeline/gui/app.py`** — the Flask app itself (collided with the
  FastAPI `app.py`, which won and is now at `interfaces/api/app.py`).
  `job_manager.py`, `templates/index.html`, and the `static/` assets it
  would use are all restored and waiting for it.
- **Eight of nine specialist `model.py` files** — only
  `specialists/change_detection/model.py` survived. Missing, each an empty
  package directory now:
  - `specialists/vqa_caption/model.py` — `VQACaptionSpecialist`
  - `specialists/vlm_caption/model.py` — `VLMCaptionSpecialist`
  - `specialists/sar_bridge/model.py` — `SARBridgeSpecialist`
  - `specialists/grounding/model.py` — `GroundingSpecialist`
  - `specialists/vlm_grounding/model.py` — `VLMGroundingSpecialist`
  - `specialists/vlm_change_detection/model.py` — `VLMChangeDetectionSpecialist`
  - `specialists/geo_rag/model.py` — `GeoRAGSpecialist` (its `indexer.py`
    helper did survive)
  - `specialists/optical_sar_fusion/model.py` — `OpticalSARFusionSpecialist`
  - `specialists/fusion_change_detection/model.py` — `FusionChangeDetectionSpecialist`
- **`evaluation/runner.py`** (`_score_item` etc.) — collided with
  `benchmark/runner.py`, which won.
- **`benchmark/README.md`** — referenced by `benchmark/__init__.py`'s own
  docstring; collided with the root `README.md`.

None of the above were invented — writing plausible-looking replacements
for missing business logic would be worse than leaving them out. The app
will still import and start (`_bind_specialists` catches and logs a warning
per unbound tool), but only `change_detection` will actually be bound.

## `README.md`

The real one is unrecoverable the same way — it lost a collision to
`.pytest_cache/README.md`, which is what the ZIP had at that path. It's
been rebuilt from `docs/architecture.md`, `docs/STATUS.md`, and the run
scripts; it's a reasonable reconstruction, not the original text.

## `pyproject.toml`

`[tool.setuptools.packages.find]` and `[tool.pytest.ini_options] testpaths`
already matched the real tree once restored — no change needed there.
`[project.scripts] aetheris = "run:main"` was the one real mismatch (see
"Genuinely missing" above); repointed to `pipeline.run:main` with a comment
explaining why.

## Lower-confidence placements

Everything above is pinned down by an import path, a docstring, or an
internal `Path(__file__)` computation. These are placed by strong
convention/pattern rather than a direct citation — worth a quick look:

- `run.bat` (root) and `run_gui.bat` (root) — dev convenience launchers, kept
  at the repo root since neither's own comments claim a subfolder.
- `docs/pitch_deck_notes.md`, `docs/ollama_setup.md` — grouped into `docs/`
  alongside `architecture.md`/`STATUS.md` by convention; nothing in the repo
  states their path directly.
- `pipeline/gui/job_manager.py` — its docstring only says "manages pipeline
  job lifecycle"; placed with the (missing) Flask GUI app it most plausibly
  belongs to.
- `logs/.gitkeep` — inferred from the Dockerfile's comment that `logs/` is a
  local dev directory intentionally excluded from the image.

## `deploy/cloud_demo/xray_config.json`

Not a stray/foreign file — `docker-compose.yml` in the same folder has a
commented-out `xray-tunnel` service that mounts exactly this file, for
optional LAN/internet tunneling of the demo. Restored alongside the
Dockerfile/compose file/entrypoint it belongs with. Flagging only because
it ships placeholder secrets (`REPLACE-WITH-UUID`, `REPLACE-WITH-PRIVATE-KEY`)
that must never be committed as real values, and because that service is
commented out for a reason — don't enable it on a box you don't want
reachable from the internet without understanding what you're exposing.

## One naming inconsistency left alone

`Dockerfile` and `deploy/cloud_demo/docker-compose.yml` both say
`security_pipeline_layer/` where every single Python import, `pipeline/run.py`'s
own docstring, and `pyproject.toml`'s package list all say `pipeline/`. This
looks like leftover wording from a rename that never reached the deploy
files (the project has at least one other admitted instance of this same
kind of drift, in `pyproject.toml`'s own dependency comments). Left
unchanged since it wasn't part of what was asked — but as written today,
building the Docker image will not copy the `pipeline/` package in at all,
which the app needs to boot.
