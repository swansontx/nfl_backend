"""API request/response schemas"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ProjectionResponse(BaseModel):
    """Single projection response"""
    player_id: str
    player_name: str
    team: str
    position: str
    market: str

    # Projection values
    mu: float = Field(..., description="Expected value")
    sigma: Optional[float] = Field(None, description="Standard deviation")
    prob_over_line: Optional[float] = Field(None, description="Probability over line")

    # Metadata
    confidence: float = Field(..., description="Model confidence 0-1")
    usage_norm: Optional[float] = Field(None, description="Usage share")
    tier: Optional[str] = Field(None, description="Core/Mid/Lotto")
    score: Optional[float] = Field(None, description="Opportunity score")
    model_version: str = Field(default="v1")

    class Config:
        from_attributes = True


class GameProjectionsResponse(BaseModel):
    """Response for game projections endpoint"""
    game_id: str
    home_team: str
    away_team: str
    game_date: datetime
    total_projections: int
    projections: List[ProjectionResponse]


class PlayerProjectionsResponse(BaseModel):
    """Response for player projections endpoint"""
    player_id: str
    player_name: str
    team: str
    position: str
    season: int
    total_projections: int
    projections: List[ProjectionResponse]


class RecomputeRequest(BaseModel):
    """Request to recompute projections"""
    game_id: str
    markets: Optional[List[str]] = Field(None, description="Specific markets to compute")
    force: bool = Field(False, description="Force recomputation even if exists")


class RecomputeResponse(BaseModel):
    """Response for recompute request"""
    status: str  # queued, running, completed, failed
    game_id: str
    job_id: str
    message: Optional[str] = None


class JobStatusResponse(BaseModel):
    """Response for job status"""
    job_id: str
    status: str  # queued, running, completed, failed
    game_id: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: Optional[float] = Field(None, description="Progress 0-1")
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: datetime
    services: Dict[str, str]  # service_name -> status


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None
