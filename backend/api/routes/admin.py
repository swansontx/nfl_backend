"""Admin endpoints"""
import uuid
from datetime import datetime
from typing import Dict
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session

from backend.api.schemas import (
    RecomputeRequest,
    RecomputeResponse,
    JobStatusResponse
)
from backend.api.dependencies import get_db, verify_admin_token
from backend.models.prop_models import PropModelRunner
from backend.config.logging_config import get_logger
from backend.database.models import Game

logger = get_logger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])

# In-memory job storage (in production, use Redis or database)
_jobs: Dict[str, Dict] = {}


async def _run_projection_job(
    job_id: str,
    game_id: str,
    markets: list[str] = None
):
    """Background task to run projection generation"""
    try:
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["started_at"] = datetime.utcnow()

        logger.info("projection_job_started", job_id=job_id, game_id=game_id)

        # Run model
        runner = PropModelRunner()
        projections = runner.generate_projections(
            game_id=game_id,
            markets=markets,
            save_to_db=True
        )

        # Update job
        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["completed_at"] = datetime.utcnow()
        _jobs[job_id]["result"] = {
            "projections_generated": len(projections),
            "markets": list(set(p.market for p in projections))
        }

        logger.info("projection_job_completed", job_id=job_id, count=len(projections))

    except Exception as e:
        logger.error("projection_job_failed", job_id=job_id, error=str(e))
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["completed_at"] = datetime.utcnow()
        _jobs[job_id]["error"] = str(e)


@router.post("/recompute", response_model=RecomputeResponse)
async def recompute_projections(
    request: RecomputeRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _auth: bool = Depends(verify_admin_token)
):
    """
    Trigger recomputation of projections for a game

    Requires admin authentication via Bearer token.

    Args:
        request: Recompute request with game_id and options

    Returns:
        Job status with job_id for tracking
    """
    logger.info("recompute_request", game_id=request.game_id)

    # Verify game exists
    game = db.query(Game).filter(Game.game_id == request.game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail=f"Game not found: {request.game_id}")

    # Create job
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "game_id": request.game_id,
        "markets": request.markets,
        "created_at": datetime.utcnow(),
        "started_at": None,
        "completed_at": None,
        "error": None,
        "result": None
    }

    # Queue background task
    background_tasks.add_task(
        _run_projection_job,
        job_id=job_id,
        game_id=request.game_id,
        markets=request.markets
    )

    return RecomputeResponse(
        status="queued",
        game_id=request.game_id,
        job_id=job_id,
        message=f"Projection computation queued for game {request.game_id}"
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    _auth: bool = Depends(verify_admin_token)
):
    """
    Get status of a background job

    Requires admin authentication.

    Args:
        job_id: Job identifier from recompute response

    Returns:
        Job status and results (if completed)
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = _jobs[job_id]

    # Calculate progress
    progress = None
    if job["status"] == "running":
        progress = 0.5  # Simple progress indicator
    elif job["status"] == "completed":
        progress = 1.0

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        game_id=job.get("game_id"),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        progress=progress,
        error=job.get("error"),
        result=job.get("result")
    )


@router.delete("/jobs/{job_id}")
async def delete_job(
    job_id: str,
    _auth: bool = Depends(verify_admin_token)
):
    """
    Delete a job from job storage

    Requires admin authentication.

    Args:
        job_id: Job identifier

    Returns:
        Success message
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job_status = _jobs[job_id]["status"]
    if job_status == "running":
        raise HTTPException(
            status_code=400,
            detail="Cannot delete running job"
        )

    del _jobs[job_id]

    return {"message": f"Job {job_id} deleted", "status": "success"}


@router.get("/jobs")
async def list_jobs(
    status: str = None,
    _auth: bool = Depends(verify_admin_token)
):
    """
    List all jobs (optionally filtered by status)

    Requires admin authentication.

    Args:
        status: Filter by status (queued/running/completed/failed)

    Returns:
        List of jobs
    """
    jobs = list(_jobs.values())

    if status:
        jobs = [j for j in jobs if j["status"] == status]

    # Sort by created_at descending
    jobs.sort(key=lambda x: x["created_at"], reverse=True)

    return {
        "total": len(jobs),
        "jobs": jobs
    }
