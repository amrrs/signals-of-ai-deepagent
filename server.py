"""FastAPI backend for Signals of AI.

Serves the spy HUD front-end and streams a live surveillance feed over
Server-Sent Events (SSE) while the deep agent works.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

from spy_agent.agent import (  # noqa: E402  (load_dotenv must run first)
    DEFAULT_MODEL,
    build_mission_prompt,
    build_spy_agent,
    stream_mission,
)

app = FastAPI(title="Signals of AI", version="1.0.0")

STATIC_DIR = Path(__file__).parent / "static"

# A single shared agent instance (stateless across requests).
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_spy_agent()
    return _agent


class Mission(BaseModel):
    target: str
    client: str | None = None
    focus: str | None = None
    nebius_token_factory_api_key: str | None = None


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": DEFAULT_MODEL}


@app.post("/api/deploy")
async def deploy(mission: Mission):
    """Run a mission and stream the live feed as SSE."""
    if not mission.target or not mission.target.strip():
        async def err():
            yield _sse({"type": "error", "text": "No target specified."})
        return StreamingResponse(err(), media_type="text/event-stream")

    prompt = build_mission_prompt(mission.target, mission.client, mission.focus)

    async def event_stream():
        yield _sse({"type": "status", "text": f"Establishing secure link · model {DEFAULT_MODEL}"})
        yield _sse({"type": "status", "text": f"Target acquired: {mission.target.strip()}"})
        try:
            runtime_key = (mission.nebius_token_factory_api_key or "").strip()
            agent = build_spy_agent(api_key=runtime_key) if runtime_key else get_agent()
        except Exception as exc:  # noqa: BLE001
            yield _sse({"type": "error", "text": f"Agent init failed: {exc}"})
            return

        try:
            async for event in stream_mission(agent, prompt):
                yield _sse(event)
                await asyncio.sleep(0)  # let the event loop flush
        except Exception as exc:  # noqa: BLE001
            yield _sse({"type": "error", "text": f"Mission aborted: {exc}"})
        yield _sse({"type": "done"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
