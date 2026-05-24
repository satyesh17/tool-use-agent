"""FastAPI HTTP wrapper around the tool-use agent."""
import sys
from pathlib import Path

# Add src/ to Python path explicitly — bypasses editable-install issues
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi import FastAPI
from pydantic import BaseModel
from tool_use_agent.agent import ToolUseAgent


app = FastAPI(title="Tool-Use Agent API", version="0.1.0")
agent = ToolUseAgent()


class QueryRequest(BaseModel):
    """Request body: a single user query for the agent."""
    query: str


class QueryResponse(BaseModel):
    """Response body: the agent's final answer."""
    answer: str


@app.post("/agent/run", response_model=QueryResponse)
def run_agent(req: QueryRequest) -> QueryResponse:
    """Run the agent on a user query, return the final answer."""
    answer = agent.run(req.query)
    return QueryResponse(answer=answer)


@app.get("/health")
def health_check() -> dict:
    """Simple health check endpoint."""
    return {"status": "ok"}
