"""Simple FastAPI wrapper exposing the AgentOrchestrator over HTTP.

Start with:

    uvicorn app.api:app --host 0.0.0.0 --port 8000

The orchestrator is created at startup and shutdown is handled on application
shutdown to ensure MCP clients are closed gracefully.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any
import logging

from app.config.settings import Settings
from app.services.agent_orchestrator import AgentOrchestrator
from app.observability.logging import configure_logging


app = FastAPI(title="LangChain DeepAgents MCP MVP")


class RunRequest(BaseModel):
    query: str


@app.on_event("startup")
async def startup_event() -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    orchestrator = AgentOrchestrator(settings)
    await orchestrator.initialize()
    app.state.orchestrator = orchestrator


@app.on_event("shutdown")
async def shutdown_event() -> None:
    orchestrator: Any = getattr(app.state, "orchestrator", None)
    if orchestrator is not None:
        try:
            await orchestrator.shutdown()
        except Exception:
            logging.getLogger(__name__).exception("Error shutting down orchestrator")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/run")
async def run(req: RunRequest) -> dict:
    orchestrator: Any = getattr(app.state, "orchestrator", None)
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    result = await orchestrator.run(req.query)
    return {"result": result}
