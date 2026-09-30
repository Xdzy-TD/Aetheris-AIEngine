"""
SAT-OS API routes — FastAPI endpoints for the SAT-OS pipeline.

Mounted on the existing FastAPI app at /v1/sat/.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from interfaces.api.security import read_upload_limited, validate_image_upload
from pipeline.schemas.sat_schemas import FeedbackEntry
from pipeline.services.sat_pipeline import SATOSPipeline

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/sat", tags=["sat-os"])

# Singleton — same instance across requests
_pipeline: SATOSPipeline | None = None


def _get_pipeline() -> SATOSPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SATOSPipeline()
    return _pipeline


@router.post("/query")
async def sat_query(
    question: Annotated[str, Form(description="Natural-language observation question")],
    modality: Annotated[str, Form()] = "optical",
    modality2: Annotated[str | None, Form()] = None,
    image: UploadFile | None = File(None),
    image2: UploadFile | None = File(None),
) -> dict[str, Any]:
    """Full SAT-OS pipeline: query → contract → gate → observability →
    analysis → evidence → falsification → verification → certificate → report.
    """
    pipeline = _get_pipeline()
    file_data: bytes | None = None
    temp_paths: list[str] = []

    try:
        file_path = file2_path = None
        file_name = file2_name = ""

        for i, upload in enumerate((image, image2)):
            if upload is None:
                continue
            data = await read_upload_limited(upload)
            validate_image_upload(data, upload.filename or "upload")
            if i == 0:
                file_data = data
            suffix = Path(upload.filename or "image.tif").suffix
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="satos_")
            tmp.write(data)
            tmp.close()
            temp_paths.append(tmp.name)
            if i == 0:
                file_path, file_name = tmp.name, upload.filename or "upload"
            else:
                file2_path, file2_name = tmp.name, upload.filename or "upload2"

        result = pipeline.execute(
            question=question, file_path=file_path, file2_path=file2_path,
            file_name=file_name, file2_name=file2_name, file_data=file_data,
            modality=modality, modality2=modality2,
        )
        return result

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("sat_query_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        for p in temp_paths:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass


@router.get("/certificate/{run_id}")
async def get_certificate(run_id: str) -> dict[str, Any]:
    """Export the answerability certificate for a completed run."""
    pipeline = _get_pipeline()
    run = pipeline.get_run(run_id)
    if not run or not run.certificate:
        raise HTTPException(status_code=404, detail=f"No certificate for run {run_id}")
    return run.certificate.model_dump(mode="json")


@router.get("/report/{run_id}")
async def get_report(run_id: str) -> dict[str, str]:
    """Get the audit report text for a run."""
    from pipeline.modules.sat_report import generate_text_report
    pipeline = _get_pipeline()
    run = pipeline.get_run(run_id)
    if not run or not run.certificate:
        raise HTTPException(status_code=404, detail=f"No run record for {run_id}")
    return {"run_id": run_id, "report": generate_text_report(run.certificate)}


@router.get("/runs")
async def list_runs() -> list[dict[str, Any]]:
    """List all SAT-OS run records with their verification status."""
    return _get_pipeline().list_runs()


@router.get("/run/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    """Inspect a previous run's full reproducibility record."""
    pipeline = _get_pipeline()
    run = pipeline.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"No run record for {run_id}")
    return run.model_dump(mode="json")


@router.post("/feedback/{run_id}")
async def submit_feedback(
    run_id: str,
    category: Annotated[str, Form()],
    detail: Annotated[str, Form()] = "",
) -> dict[str, Any]:
    """Submit feedback/correction for a completed run."""
    pipeline = _get_pipeline()
    entry = FeedbackEntry(run_id=run_id, category=category, detail=detail)
    saved = pipeline.submit_feedback(entry)
    return {"feedback_id": saved.feedback_id, "run_id": run_id}


@router.get("/feedback/{run_id}")
async def get_feedback(run_id: str) -> list[dict[str, Any]]:
    """Get all feedback for a run."""
    return [e.model_dump(mode="json") for e in _get_pipeline().get_feedback(run_id)]


@router.get("/compare/{run_id_a}/{run_id_b}")
async def compare_runs(run_id_a: str, run_id_b: str) -> dict[str, Any]:
    """Compare two runs to see if the system improved."""
    return _get_pipeline().compare_runs(run_id_a, run_id_b)
