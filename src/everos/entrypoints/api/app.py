"""FastAPI application factory.

Wires CORS + the project's middleware stack + global exception handler +
lifespan, and registers the public routes (``/health``, ``/metrics``).
"""

from __future__ import annotations

import hmac
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from everos.core.lifespan import (
    LifespanProvider,
    MetricsLifespanProvider,
    build_lifespan,
)
from everos.core.middleware import (
    DEFAULT_CORS_ALLOW_CREDENTIALS,
    DEFAULT_CORS_ALLOW_HEADERS,
    DEFAULT_CORS_ALLOW_METHODS,
    DEFAULT_CORS_ORIGINS,
    ProfileMiddleware,
    PrometheusMiddleware,
    global_exception_handler,
)
from everos.core.observability.logging import get_logger

from .lifespans import (
    CascadeLifespanProvider,
    LanceDBLifespanProvider,
    LLMLifespanProvider,
    OmeLifespanProvider,
    SqliteLifespanProvider,
)
from .routes import (
    admin,
    get,
    health,
    memorize,
    metrics,
    search,
)

logger = get_logger(__name__)


def _docs_enabled() -> bool:
    """Enable docs endpoints (/docs, /redoc, /openapi.json) only in dev."""
    return os.environ.get("ENV", "prod").upper() == "DEV"


def _load_api_token() -> str:
    """Load optional bearer auth, failing closed for an explicit token file."""
    token = os.environ.get("EVEROS_API_TOKEN", "").strip()
    if token:
        return token
    token_file = os.environ.get("EVEROS_API_TOKEN_FILE", "").strip()
    if not token_file:
        return ""
    try:
        with open(token_file, encoding="utf-8") as handle:
            token = handle.read().strip()
    except OSError as exc:
        raise RuntimeError(f"cannot read EVEROS_API_TOKEN_FILE: {exc}") from exc
    if not token:
        raise RuntimeError("EVEROS_API_TOKEN_FILE is empty")
    return token


def create_app(
    *,
    cors_origins: list[str] | None = None,
    cors_allow_credentials: bool = DEFAULT_CORS_ALLOW_CREDENTIALS,
    cors_allow_methods: list[str] | None = None,
    cors_allow_headers: list[str] | None = None,
    lifespan_providers: list[LifespanProvider] | None = None,
) -> FastAPI:
    """Build the FastAPI application instance.

    Args:
        cors_origins: Allowed CORS origins (default: ``["*"]``).
        cors_allow_credentials: Whether to allow credentials (default: True).
        cors_allow_methods: Allowed CORS methods (default: ``["*"]``).
        cors_allow_headers: Allowed CORS headers (default: ``["*"]``).
        lifespan_providers: Optional list of LifespanProvider; defaults to
            ``[MetricsLifespanProvider(), SqliteLifespanProvider(),
            LanceDBLifespanProvider(), CascadeLifespanProvider(),
            OmeLifespanProvider()]``.

    Returns:
        FastAPI: Configured application instance.
    """
    enable_docs = _docs_enabled()

    if lifespan_providers is None:
        lifespan_providers = [
            MetricsLifespanProvider(),
            LLMLifespanProvider(),
            SqliteLifespanProvider(),
            LanceDBLifespanProvider(),
            CascadeLifespanProvider(),
            OmeLifespanProvider(),
        ]

    app = FastAPI(
        title="everos",
        version="0.1.0",
        description="md-first memory extraction framework",
        lifespan=build_lifespan(lifespan_providers),
        docs_url="/docs" if enable_docs else None,
        redoc_url="/redoc" if enable_docs else None,
        openapi_url="/openapi.json" if enable_docs else None,
    )

    # Exception handlers: HTTPException, validation errors, plus a fallback.
    app.add_exception_handler(HTTPException, global_exception_handler)
    app.add_exception_handler(RequestValidationError, global_exception_handler)
    app.add_exception_handler(Exception, global_exception_handler)

    # Middleware order: earlier `add_middleware` calls become inner, later ones outer.
    # CORS innermost (matches base_app.py legacy pattern).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or DEFAULT_CORS_ORIGINS,
        allow_credentials=cors_allow_credentials,
        allow_methods=cors_allow_methods or DEFAULT_CORS_ALLOW_METHODS,
        allow_headers=cors_allow_headers or DEFAULT_CORS_ALLOW_HEADERS,
    )
    app.add_middleware(PrometheusMiddleware)
    app.add_middleware(ProfileMiddleware)

    api_token = _load_api_token()

    @app.middleware("http")
    async def require_api_token(request: Request, call_next):
        """Require a configured bearer token on every non-health request."""
        if request.url.path == "/health" or request.method == "OPTIONS":
            return await call_next(request)
        is_admin_request = request.url.path.startswith("/api/v1/admin/")
        if is_admin_request and not api_token:
            return JSONResponse(
                status_code=503,
                content={"detail": "Admin API requires EVEROS_API_TOKEN"},
            )
        if not api_token:
            return await call_next(request)
        supplied = request.headers.get("Authorization", "")
        expected = f"Bearer {api_token}"
        if not hmac.compare_digest(supplied, expected):
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
        return await call_next(request)

    # Routes.
    app.include_router(health.router)
    app.include_router(metrics.router)
    app.include_router(memorize.router)
    app.include_router(search.router)
    app.include_router(get.router)
    app.include_router(admin.router)

    logger.info("app_created", docs_enabled=enable_docs)
    return app
