"""
AETHERIS M01–M20 — Entry Point.

Usage:
    python -m pipeline.run gui          Launch the web GUI
    python -m pipeline.run health       Run M01 health check
    python -m pipeline.run pipeline     Execute pipeline (CLI)
    python -m pipeline.run dag          Show DAG structure
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path for old AETHERIS imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def cmd_gui(args: argparse.Namespace) -> None:
    """Launch the web GUI."""
    from pipeline.gui.app import main
    main()


def cmd_health(args: argparse.Namespace) -> None:
    """Run M01 health check."""
    from pipeline.modules.m01_baseline import run_health_check
    result = run_health_check()
    print(json.dumps(result.model_dump(mode="json"), indent=2))


def cmd_pipeline(args: argparse.Namespace) -> None:
    """Execute the pipeline from CLI."""
    from pipeline.services.pipeline import AetherisPipeline
    from pipeline.schemas.contracts import PipelineRequest

    pipeline = AetherisPipeline()
    request = PipelineRequest(
        question=args.question,
        modality=args.modality,
    )
    result = pipeline.execute(request)

    print("\n" + "=" * 60)
    print("  AETHERIS M01–M20 Pipeline Result")
    print("=" * 60)
    print(f"  Pipeline ID:  {result.pipeline_id}")
    print(f"  Status:       {result.status.value}")
    print(f"  Answer:       {result.final_answer or 'N/A'}")
    print(f"  Time:         {result.total_execution_time_s}s")
    print(f"  Modules:      {len(result.modules)}")
    print("-" * 60)
    for name, mod in result.modules.items():
        status_icon = "✓" if mod.status.value == "success" else "✗" if mod.status.value == "failure" else "○"
        print(f"  {status_icon} {name}: {mod.status.value} ({mod.execution_time_s:.4f}s)")
    print("=" * 60 + "\n")


def cmd_dag(args: argparse.Namespace) -> None:
    """Show DAG structure."""
    from pipeline.services.orchestrator import build_pipeline_dag
    dag = build_pipeline_dag()
    validation = dag.validate()
    state = dag.get_state()
    print(json.dumps(state, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="aetheris-m01-m20",
        description="AETHERIS M01–M20 Hackathon Layer",
    )
    subparsers = parser.add_subparsers(dest="command")

    # gui
    subparsers.add_parser("gui", help="Launch the web GUI")

    # health
    subparsers.add_parser("health", help="Run M01 health check")

    # pipeline
    pipe_parser = subparsers.add_parser("pipeline", help="Execute pipeline")
    pipe_parser.add_argument(
        "--question", "-q",
        default="What land cover types are visible in this satellite image?",
    )
    pipe_parser.add_argument("--modality", default="optical")

    # dag
    subparsers.add_parser("dag", help="Show DAG structure")

    args = parser.parse_args()

    if args.command is None:
        args.command = "gui"

    commands = {
        "gui": cmd_gui,
        "health": cmd_health,
        "pipeline": cmd_pipeline,
        "dag": cmd_dag,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
