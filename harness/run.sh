#!/usr/bin/env bash
# Aetheris dev harness — single entry point for all local checks.
#
# Usage:
#   ./harness/run.sh              run everything (lint, types, tests, demo smoke test, audit verify)
#   ./harness/run.sh lint         ruff only
#   ./harness/run.sh types        mypy only
#   ./harness/run.sh test         pytest only
#   ./harness/run.sh smoke        run.py demo --no-llm, must exit 0 and produce a valid audit chain
#   ./harness/run.sh benchmark    run.py benchmark against sample_dataset.jsonl (refusal-guard only)
#   ./harness/run.sh audit        verify logs/execution_chain.jsonl
#
# Exits non-zero on the first failing stage so it's safe to use as a CI gate.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

STAGE="${1:-all}"
FAILED=0

_hr() { printf '%s\n' "------------------------------------------------------------"; }

run_lint() {
    _hr; echo "[harness] ruff lint"
    if command -v ruff >/dev/null 2>&1; then
        ruff check . || FAILED=1
    else
        echo "[harness] ruff not installed — skipping (pip install -e '.[dev]')"
    fi
}

run_types() {
    _hr; echo "[harness] mypy type-check"
    if command -v mypy >/dev/null 2>&1; then
        mypy controller specialists confidence interfaces || FAILED=1
    else
        echo "[harness] mypy not installed — skipping (pip install -e '.[dev]')"
    fi
}

run_test() {
    _hr; echo "[harness] pytest"
    python -m pytest -q || FAILED=1
}

run_smoke() {
    _hr; echo "[harness] smoke test: run.py demo --no-llm"
    python run.py demo --no-llm -q "What land cover types are visible in this satellite image?" || FAILED=1
}

run_benchmark() {
    _hr; echo "[harness] benchmark: sample_dataset.jsonl (refusal-guard regression, not a capability score)"
    python run.py benchmark --dataset benchmark/sample_dataset.jsonl \
        --out benchmark/reports/latest.json --no-llm || FAILED=1
}

run_audit() {
    _hr; echo "[harness] audit log verification"
    python run.py verify-audit || FAILED=1
}

case "$STAGE" in
    lint)      run_lint ;;
    types)     run_types ;;
    test)      run_test ;;
    smoke)     run_smoke ;;
    benchmark) run_benchmark ;;
    audit)     run_audit ;;
    all)
        run_lint
        run_types
        run_test
        run_smoke
        run_benchmark
        run_audit
        ;;
    *)
        echo "Unknown stage: $STAGE" >&2
        echo "Valid stages: lint, types, test, smoke, benchmark, audit, all" >&2
        exit 2
        ;;
esac

_hr
if [ "$FAILED" -ne 0 ]; then
    echo "[harness] FAILED — see stage output above"
    exit 1
fi
echo "[harness] all checks passed"
