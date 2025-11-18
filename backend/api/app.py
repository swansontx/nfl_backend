"""Main FastAPI application"""
from datetime import datetime
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import time
import uuid

from backend.config import settings
from backend.config.logging_config import setup_logging, get_logger
from backend.api.schemas import HealthResponse, ErrorResponse
from backend.api.routes import projections, admin

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Create app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="NFL Props Backend API for projection generation and retrieval",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests for tracing"""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    # Log request
    start_time = time.time()
    logger.info(
        "request_started",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        client=request.client.host if request.client else None
    )

    # Process request
    response = await call_next(request)

    # Log response
    duration = time.time() - start_time
    logger.info(
        "request_completed",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=int(duration * 1000)
    )

    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id

    return response


# Exception handlers
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions"""
    request_id = getattr(request.state, "request_id", None)

    logger.error(
        "unhandled_exception",
        request_id=request_id,
        error=str(exc),
        type=type(exc).__name__,
        path=request.url.path
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if settings.debug else "An unexpected error occurred",
            request_id=request_id
        ).dict()
    )


# Routes
@app.get("/", tags=["root"])
async def root():
    """Root endpoint"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs_url": "/docs",
        "health_url": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health():
    """
    Health check endpoint

    Returns application health and service status
    """
    # Check service health
    services = {
        "api": "healthy",
        "database": "unknown",  # Could ping database
        "redis": "unknown"  # Could ping redis
    }

    # Try to check database
    try:
        from backend.database.session import engine
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        services["database"] = "healthy"
    except Exception as e:
        logger.warning("database_health_check_failed", error=str(e))
        services["database"] = "unhealthy"

    # Try to check Redis
    try:
        from backend.roster_injury import RosterInjuryService
        service = RosterInjuryService(use_cache=True)
        if service.redis_client:
            service.redis_client.ping()
            services["redis"] = "healthy"
        else:
            services["redis"] = "disabled"
    except Exception as e:
        logger.warning("redis_health_check_failed", error=str(e))
        services["redis"] = "unhealthy"

    overall_status = "healthy" if all(
        s in ["healthy", "disabled", "unknown"] for s in services.values()
    ) else "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        timestamp=datetime.utcnow(),
        services=services
    )


# Include routers
app.include_router(projections.router)
app.include_router(admin.router)


# Startup event
@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info(
        "application_starting",
        app_name=settings.app_name,
        version=settings.app_version,
        debug=settings.debug
    )

    # Ensure directories exist
    settings.ensure_directories()

    logger.info("application_started")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("application_shutting_down")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower()
    )
