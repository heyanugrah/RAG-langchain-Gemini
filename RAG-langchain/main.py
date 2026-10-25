import logging
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from config import settings
from rag_engine import rag_service


# Logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# Initialize RAG when the application starts
@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Initializing RAG service...")

    try:
        await run_in_threadpool(rag_service.initialize)
        app.state.rag_ready = True

        logger.info("RAG service is ready.")

    except Exception:
        app.state.rag_ready = False
        logger.exception("RAG initialization failed.")
        raise

    yield

    logger.info("Shutting down RAG application.")


app = FastAPI(
    title="Banking RAG API",
    description="Banking question-answering API using LangChain, ChromaDB, and Gemini.",
    version="1.0.0",
    lifespan=lifespan
)


# Request model
class AskRequest(BaseModel):

    question: str = Field(
        min_length=3,
        max_length=1000,
        description="Question about the banking documents"
    )


# Response models
class SourceDocument(BaseModel):
    source: str
    content: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]


class HealthResponse(BaseModel):
    status: str
    rag_ready: bool


# Root endpoint
@app.get("/")
def root():
    return {
        "message": "Banking RAG API",
        "docs": "/docs",
        "health": "/health"
    }


# Health endpoint
@app.get("/health", response_model=HealthResponse)
def health_check():

    ready = getattr(app.state, "rag_ready", False)

    return HealthResponse(
        status="ok" if ready else "not_ready",
        rag_ready=ready
    )


# Ask endpoint
@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):

    if not getattr(app.state, "rag_ready", False):
        raise HTTPException(
            status_code=503,
            detail="RAG service is not ready."
        )

    try:
        result = rag_service.ask(request.question)

        return AskResponse(**result)

    except Exception:
        logger.exception("Failed to process RAG question.")

        raise HTTPException(
            status_code=502,
            detail="Unable to generate an answer. Please try again."
        )