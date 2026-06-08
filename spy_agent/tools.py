"""Field tools available to the Competitor Spy agent.

These mirror the classic deepagents research tools (`tavily_search` and
`think_tool`) but are tuned for competitive-intelligence gathering.
"""

from __future__ import annotations

import os
from typing import Literal

from langchain_core.tools import tool
from tavily import TavilyClient

_tavily_client: TavilyClient | None = None


def _client() -> TavilyClient:
    """Lazily build a single Tavily client from the environment."""
    global _tavily_client
    if _tavily_client is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY is not set in the environment / .env")
        _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client


@tool
def tavily_search(
    query: str,
    max_results: int = 4,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
) -> str:
    """Run a live web search to gather intelligence on a target.

    Use this to surface a competitor's products, pricing, funding, hiring,
    launches, news, reviews and positioning. Issue focused queries (one angle
    at a time) rather than one broad query.

    Args:
        query: The search query (e.g. "Acme Corp pricing 2026").
        max_results: Number of results to return (1-10).
        topic: "general", "news" (recent events), or "finance" (funding/markets).
        include_raw_content: Set True to pull fuller page text for deep dives.

    Returns:
        A formatted string of search hits (title, url, snippet).
    """
    max_results = max(1, min(int(max_results), 10))
    response = _client().search(
        query,
        max_results=max_results,
        topic=topic,
        include_raw_content=include_raw_content,
    )

    results = response.get("results", []) or []
    if not results:
        return f"No intel found for query: {query!r}"

    # Keep each snippet compact. Tavily can return very long pages; feeding the
    # full text back to the model bloats the context (50k+ tokens) and causes
    # slow generations / request timeouts.
    snippet_cap = 900 if not include_raw_content else 1800

    lines: list[str] = [f"INTEL FOR: {query!r} ({len(results)} sources)\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        content = (r.get("content") or "").strip()
        if include_raw_content and r.get("raw_content"):
            content = (r["raw_content"] or content).strip()
        if len(content) > snippet_cap:
            content = content[:snippet_cap] + "…"
        lines.append(f"[{i}] {title}\nURL: {url}\n{content}")
        lines.append("")
    return "\n".join(lines).strip()


@tool
def think_tool(reflection: str) -> str:
    """A private scratchpad for strategic reasoning between searches.

    Use this after gathering intel to decide whether you have enough to write
    a section of the dossier, what gaps remain, and what to investigate next.
    This does NOT search the web; it only records your reasoning.

    Args:
        reflection: Your analysis of findings so far and the next move.

    Returns:
        Confirmation that the reflection was logged.
    """
    return f"Reflection logged:\n{reflection}"
