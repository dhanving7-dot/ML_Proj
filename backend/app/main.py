"""FastAPI application: wiring, middleware and error handling.

Run with:  uvicorn app.main:app --reload   (from the backend/ directory)
Docs:      http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from . import API_VERSION, db
from .config import get_settings
from .errors import AppError
from .orm import RequestLog
from .routers import observability, predictions, registry, system
from .services.dataset_service import ensure_default_dataset
from .services.events import record_event
from .services.model_service import model_service

log = logging.getLogger("fraudguard")

UNLOGGED_PREFIXES = ("/api/health", "/docs", "/redoc", "/openapi.json")


def startup() -> None:
    settings = get_settings()
    db.configure(settings.resolved_database_url)
    db.init_db()
    with db.session_scope() as s:
        ensure_default_dataset(s, settings)
    try:
        with db.session_scope() as s:
            model_service.load_active(s)
    except Exception:  # keep serving health/docs; /predict returns 503 until a valid model exists
        log.exception("Active model could not be loaded")


@asynccontextmanager
async def lifespan(_: FastAPI):  # type: ignore[no-untyped-def]
    await run_in_threadpool(startup)
    yield


app = FastAPI(
    title="FraudGuard API",
    version=API_VERSION,
    description=(
        "Credit-card fraud detection with Logistic Regression, Isolation Forest and a validated hybrid risk score: "
        "leakage-safe training, model registry, monitoring, retraining and an M2-M6 machine-learning laboratory. "
        "Risk scores are statistical estimates and support, but do not replace, human investigation."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def _write_request_log(method: str, path: str, status: int, latency: float) -> None:
    try:
        with db.session_scope() as s:
            s.add(RequestLog(method=method, path=path[:200], status_code=status, latency_ms=round(latency, 3)))
    except Exception:
        log.exception("Failed to write request log")


@app.middleware("http")
async def request_logging(request: Request, call_next):  # type: ignore[no-untyped-def]
    start = time.perf_counter()
    response = await call_next(request)
    path = request.url.path
    if request.method != "OPTIONS" and path.startswith("/api") and not path.startswith(UNLOGGED_PREFIXES):
        latency = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{latency:.2f}"
        await run_in_threadpool(_write_request_log, request.method, path, response.status_code, latency)
    return response


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """422 with field locations and messages only. The rejected input is not echoed back: it may be a
    non-finite number that JSON cannot represent, and echoing inputs is unnecessary exposure."""
    errors = [{"loc": [str(p) for p in e.get("loc", ())], "msg": str(e.get("msg", "Invalid value")), "type": str(e.get("type", ""))} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


def _record_error(error_id: str, path: str, exc: Exception) -> None:
    try:
        with db.session_scope() as s:
            record_event(s, "server_error", "critical", f"Unhandled {type(exc).__name__} on {path}", {"error_id": error_id})
    except Exception:
        pass


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    error_id = uuid.uuid4().hex[:12]
    log.exception("Unhandled error %s on %s", error_id, request.url.path)
    await run_in_threadpool(_record_error, error_id, request.url.path, exc)
    # Never expose stack traces or filesystem paths to clients.
    return JSONResponse(status_code=500, content={"detail": "Internal server error", "error_id": error_id})


app.include_router(system.router)
app.include_router(predictions.router)
app.include_router(registry.router)
app.include_router(observability.router)

_STATIC = os.environ.get("FG_STATIC_DIR")
if _STATIC and (Path(_STATIC) / "index.html").is_file():
    # Single-origin deployment: serve the built React app; unknown non-API paths fall back to index.html.
    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    _STATIC_ROOT = Path(_STATIC).resolve()

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        if path.startswith("api/"):
            raise HTTPException(status_code=404)
        candidate = (_STATIC_ROOT / path).resolve()
        if path and candidate.is_file() and _STATIC_ROOT in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(_STATIC_ROOT / "index.html")

else:

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, Any]:
        return {"name": "FraudGuard API", "version": API_VERSION, "docs": "/docs"}
