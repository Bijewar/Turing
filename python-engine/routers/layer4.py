"""
routers/layer4.py — Layer 4 Report Generation Endpoints

Exposes the /report/{run_id} endpoints that generate, retrieve, and stream
the final human-readable report from the Top 3 cross-domain bridges produced by Layer 3.
"""
import asyncio
import json
import logging
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from database import crud
from database.database import get_db
from schemas.layer3 import Step14Response
from schemas.report import FinalReport
from services.layer4.report_builder import build_report
from routers.runs import RUNS_STORE

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/report/{run_id}",
    response_model=FinalReport,
    summary="Layer 4 — Generate final cross-domain bridge report"
)
async def generate_report(
    run_id: str,
    bridges: Step14Response,
    db: AsyncSession = Depends(get_db)
):
    """
    Accepts the Step 14 Top 3 ranked bridges and generates a full
    human-readable FinalReport including problem statement, executive
    summary, recommended experiment, and contradiction warnings.

    Validates run_id as a UUID, verifies run exists in database, returns cached
    report if already generated, and persists new reports.
    """
    try:
        run_uuid = uuid.UUID(run_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="run_id must be a valid UUID")

    run = await crud.get_run(db, str(run_uuid))
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    # Idempotency: return existing cached report if already generated
    existing_report = run.get_final_report()
    if existing_report:
        try:
            return FinalReport(**existing_report)
        except Exception:
            pass

    try:
        report = await build_report(run_id=str(run_uuid), step14_response=bridges)
        report_dict = report.model_dump(mode="json")
        await crud.update_run_report(db, str(run_uuid), report_dict)
        return report
    except Exception as exc:
        logger.error(f"build_report failed for run {run_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Report generation failed.")


@router.get(
    "/report/{run_id}",
    response_model=FinalReport,
    summary="Layer 4 — Retrieve generated report for a run"
)
async def get_report(
    run_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieves the generated FinalReport for the specified run_id.
    Returns 400 if run_id is not a valid UUID.
    Returns 404 if the run does not exist or if no report has been generated.
    Returns 400 if the run is still in progress or has failed.
    """
    try:
        run_uuid = uuid.UUID(run_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="run_id must be a valid UUID")

    run = await crud.get_run(db, str(run_uuid))
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    if run.status in ("RUNNING", "PENDING"):
        raise HTTPException(
            status_code=400,
            detail=f"Run '{run_id}' is still in progress (status: {run.status})"
        )

    if run.status == "FAILED":
        raise HTTPException(
            status_code=400,
            detail=f"Run '{run_id}' failed: {run.error_message or 'Unknown error'}"
        )

    report_dict = run.get_final_report()
    if not report_dict:
        raise HTTPException(status_code=404, detail=f"No report found for run '{run_id}'")

    try:
        return FinalReport(**report_dict)
    except Exception as exc:
        logger.error(f"Failed to parse stored report for run {run_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to parse stored report.")


@router.get(
    "/report/{run_id}/stream",
    summary="Layer 4 — Stream report generation updates via SSE"
)
async def get_layer4_stream(
    run_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Streams Layer 4 report generation updates via Server-Sent Events (SSE).
    Handles client disconnects, sends periodic heartbeats, enforces a max-wait
    deadline timeout, and emits terminal events upon completion or error.
    """
    try:
        uuid.UUID(run_id)
    except (ValueError, AttributeError):
        async def invalid_id_generator():
            yield f"event: error\ndata: {json.dumps({'error': 'Invalid run_id UUID format'})}\n\n"
        return StreamingResponse(
            invalid_id_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    async def event_generator():
        start_time = time.monotonic()
        max_wait_timeout = 300.0  # 5 minutes deadline
        last_heartbeat = time.monotonic()

        while True:
            if await request.is_disconnected():
                logger.info(f"Layer 4 SSE client disconnected for run {run_id}")
                break

            if time.monotonic() - start_time > max_wait_timeout:
                logger.warning(f"Layer 4 SSE stream timed out after {max_wait_timeout}s for run {run_id}")
                yield f"event: error\ndata: {json.dumps({'error': 'Stream timed out waiting for report generation'})}\n\n"
                break

            # Check in-memory store
            if run_id in RUNS_STORE and "layer4Data" in RUNS_STORE[run_id]:
                data = RUNS_STORE[run_id]["layer4Data"]
                yield f"data: {json.dumps(data)}\n\n"
                if data.get("isComplete"):
                    break
            else:
                # Cold-start fallback: check database
                run = await crud.get_run(db, run_id)
                if not run:
                    yield f"event: error\ndata: {json.dumps({'error': f'Run {run_id} not found'})}\n\n"
                    break

                if run.status == "COMPLETE" and run.get_final_report():
                    report = run.get_final_report()
                    data = {
                        "isComplete": True,
                        "theMechanism": report.get("the_mechanism", ""),
                        "theExperiment": report.get("the_experiment", ""),
                        "whoSolvedThis": report.get("who_solved_this", ""),
                        "warningsAndConflicts": report.get("warnings_and_conflicts", []),
                        "next3Actions": report.get("next_3_actions", []),
                        "confidenceDisclaimer": report.get("confidence_disclaimer", ""),
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    break
                elif run.status == "FAILED":
                    yield f"event: error\ndata: {json.dumps({'error': 'Run failed during execution'})}\n\n"
                    break
                else:
                    yield f"data: {json.dumps({'isComplete': False, 'theMechanism': 'Waiting for prior layers...', 'theExperiment': '', 'whoSolvedThis': '', 'warningsAndConflicts': []})}\n\n"

            # Heartbeat check
            if time.monotonic() - last_heartbeat >= 15.0:
                yield ": heartbeat\n\n"
                last_heartbeat = time.monotonic()

            await asyncio.sleep(1.0)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
