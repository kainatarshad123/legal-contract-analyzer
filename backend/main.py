"""Application entry point for the Legal Contract Analyzer API."""

import logging

from dotenv import load_dotenv

# Load environment variables before importing routers/services.
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from db.database import init_db
from routers.comparison import router as comparison_router
from routers.contracts import router as contracts_router
from routers.exports import router as exports_router
from routers.qa import router as qa_router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

init_db()

app = FastAPI(title="Legal Contract ML Agent SaaS")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(contracts_router)
app.include_router(qa_router)
app.include_router(exports_router)
app.include_router(comparison_router)


@app.exception_handler(Exception)
async def handle_unexpected_exception(
    request: Request,
    error: Exception,
) -> JSONResponse:
    """Log unexpected errors and return a safe API response."""

    logger.exception(
        "Unhandled error while processing %s %s",
        request.method,
        request.url.path,
        exc_info=error,
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected server error occurred.",
        },
    )


@app.get("/")
def home() -> dict[str, str]:
    return {
        "message": (
            "Legal Contract ML Agent SaaS backend is running "
            "with Gemini support."
        )
    }