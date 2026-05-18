"""
main.py — Turing API Entry Point

Initialises the FastAPI application, registers all routers,
configures CORS middleware, database, and starts the Uvicorn server.
"""
import os
import uuid
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from dotenv import load_dotenv

# Load .env before any other imports that might check env vars
load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Single source of truth for the API version.
# Set APP_VERSION in .env or the deployment environment on each release.
# ---------------------------------------------------------------------------
APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")

from database.database import init_db, engine
from routers import layer1, layer2, layer3, layer4, runs


# ===========================================================================
# Middlewares
# ===========================================================================

class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Request-Id", str(uuid.uuid4()))
        response = await call_next(request)
        response.headers["X-Request-Id"] = req_id
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_upload_mb: int = 50):
        super().__init__(app)
        self.max_upload_bytes = max_upload_mb * 1024 * 1024

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_upload_bytes:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Request size exceeds {self.max_upload_bytes // (1024*1024)}MB limit."}
            )
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests_per_minute: int = 120):
        super().__init__(app)
        self.max_requests = max_requests_per_minute
        self.requests = {}

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        # Clean old timestamps
        timestamps = self.requests.get(client_ip, [])
        timestamps = [t for t in timestamps if now - t < 60]
        if len(timestamps) >= self.max_requests:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please try again later."}
            )
        timestamps.append(now)
        self.requests[client_ip] = timestamps
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise resources on startup, tear down on shutdown."""
    await init_db()
    try:
        yield
    finally:
        await engine.dispose()


app = FastAPI(
    title="Turing - Causal Nexus API",
    description="Causal Graph Discovery Engine Backend",
    version=APP_VERSION,
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Global Exception Handler
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred.", "status": 500}
    )

# ---------------------------------------------------------------------------
# Middleware Registration
# ---------------------------------------------------------------------------
app.add_middleware(RequestIdMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestSizeLimitMiddleware, max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "50")))
app.add_middleware(RateLimitMiddleware)

cors_origins_env = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
allowed_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept", "Origin", "X-Requested-With"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(layer1.router, prefix="/api/layer1", tags=["Layer 1 — Data Ingestion"])
app.include_router(layer2.router, prefix="/api/layer2", tags=["Layer 2 — Agent Simulation"])
app.include_router(layer3.router, prefix="/api/layer3", tags=["Layer 3 — Cross-Domain Search"])
app.include_router(layer4.router, prefix="/api/layer4", tags=["Layer 4 — Report Generation"])
app.include_router(runs.router,   prefix="/api/runs",   tags=["Runs — Session Management"])


@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "service": "turing-python-engine", "version": APP_VERSION}


@app.get("/health", tags=["Health"])
async def health_check():
    from sqlalchemy import text
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"status": "healthy", "version": APP_VERSION}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": "Database unreachable"}
        )


# ---------------------------------------------------------------------------
# Local development entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )