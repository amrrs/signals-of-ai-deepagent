"""Competitor Spy: a deepagents-powered competitive-intelligence operative."""

from __future__ import annotations

import os
from typing import Any, AsyncIterator

from deepagents import create_deep_agent
from langchain_nebius import ChatNebius

from .tools import tavily_search, think_tool

# NOTE: We default to an *instruct* (non-thinking) model. Thinking models such
# as "moonshotai/Kimi-K2.5" sometimes emit their final answer into a separate
# reasoning channel that langchain-nebius drops, leaving `content` empty. The
# instruct model reliably returns the dossier in `content`. Override via the
# SPY_MODEL env var if you want to experiment with another model.
DEFAULT_MODEL = os.getenv("SPY_MODEL", "nvidia/Nemotron-3-Ultra-550b-a55b")

SPY_SYSTEM_PROMPT = """You are CIPHER, an elite competitive-intelligence operative.
Your mission: gather open-source intelligence (OSINT) on a TARGET company and
produce a sharp, decision-ready intelligence dossier for your client.

## Tradecraft (how you operate)
1. Start by calling `think_tool` to lay out your investigation plan: what angles
   you need to cover (products, pricing, funding, customers, hiring, recent
   moves, weaknesses).
2. Use `tavily_search` to gather evidence. Run FOCUSED queries, one angle at a
   time (e.g. "<target> pricing", "<target> funding 2026", "<target> reviews
   complaints", "<target> new product launch"). Prefer the "news" topic for
   recent moves and "finance" for funding.
3. After every 1-2 searches, call `think_tool` to assess what you learned, what
   is still missing, and decide the next query. Do NOT search blindly.
4. Aim for breadth then depth. Run enough searches to be confident (typically
   5-9), then stop and write the dossier. Don't over-search.

## Rules of engagement
- Only use publicly available information. Never fabricate facts, numbers, or
  quotes. If something is unknown, say "UNCONFIRMED" rather than guessing.
- Always attribute claims to sources (include the URLs you used).
- Be specific and quantitative where possible (prices, dates, headcount, $).

## Final report format (return ONLY this as your final message, in Markdown)
Produce a dossier titled with the target name, using these sections:

# 🎯 INTELLIGENCE DOSSIER: <Target>

**Classification:** CONFIDENTIAL · **Compiled by:** CIPHER

## 1. Executive Summary
3-5 punchy bullets a busy exec can read in 20 seconds.

## 2. Company Snapshot
Founded, HQ, size, funding/ownership, core business. Mark UNCONFIRMED items.

## 3. Products & Pricing
Key offerings, tiers, and prices (with figures where found).

## 4. Strengths
What they do well / their moat.

## 5. Weaknesses & Vulnerabilities
Gaps, complaints, churn signals, negative reviews.

## 6. Recent Moves (last ~12 months)
Launches, funding, hires, partnerships, pivots — with dates.

## 7. Market Positioning
Who they target, how they differentiate, main rivals.

## 8. Threats & Opportunities (for the client)
Where the target is dangerous, and where they are exposed.

## 9. Recommended Counter-Moves
3-5 concrete, actionable recommendations for the client.

## 10. Sources
Numbered list of the URLs you relied on.

Keep it tight, evidence-driven, and free of fluff.
"""


def build_spy_agent(model_name: str = DEFAULT_MODEL, api_key: str | None = None):
    """Create the compiled Competitor Spy deep agent."""
    api_key = api_key or os.getenv("NEBIUS_TOKEN_FACTORY_API_KEY")
    if not api_key:
        raise RuntimeError("Set NEBIUS_TOKEN_FACTORY_API_KEY in your .env")

    model = ChatNebius(
        model=model_name,
        api_key=api_key,
        temperature=0.3,
        max_tokens=8000,
        timeout=180,
        max_retries=2,
    )

    return create_deep_agent(
        model=model,
        tools=[tavily_search, think_tool],
        system_prompt=SPY_SYSTEM_PROMPT,
    )


def build_mission_prompt(
    target: str,
    client: str | None = None,
    focus: str | None = None,
) -> str:
    """Turn the form inputs into a mission briefing for the agent."""
    lines = [f"TARGET COMPANY: {target.strip()}"]
    if client and client.strip():
        lines.append(f"CLIENT (you work for): {client.strip()}")
        lines.append(
            "Frame threats, opportunities and counter-moves from the client's "
            "perspective versus this target."
        )
    if focus and focus.strip():
        lines.append(f"PRIORITY FOCUS AREAS: {focus.strip()}")
    lines.append(
        "\nConduct the investigation using your tradecraft, then deliver the "
        "full intelligence dossier in the required format."
    )
    return "\n".join(lines)


def _summarize_tool_call(name: str, args: dict[str, Any]) -> str:
    if name == "tavily_search":
        q = args.get("query", "")
        topic = args.get("topic", "general")
        return f'Running search [{topic}]: "{q}"'
    if name == "think_tool":
        r = str(args.get("reflection", "")).strip()
        return r if len(r) <= 400 else r[:400] + "…"
    if name == "write_todos":
        todos = args.get("todos", [])
        return f"Updating mission checklist ({len(todos)} items)"
    return f"Calling {name}"


async def stream_mission(
    agent,
    mission: str,
) -> AsyncIterator[dict[str, Any]]:
    """Stream the agent run as a sequence of spy-feed events.

    Yields dicts shaped like:
      {"type": "plan"|"search"|"intel"|"status"|"final"|"error", ...}
    """
    final_text = ""
    reasoning_fallback = ""
    seen_tool_calls: set[str] = set()

    try:
        async for chunk in agent.astream(
            {"messages": [{"role": "user", "content": mission}]},
            stream_mode="updates",
            config={"recursion_limit": 100},
        ):
            for node, update in chunk.items():
                messages = (update or {}).get("messages", []) if isinstance(update, dict) else []
                for msg in messages:
                    msg_type = getattr(msg, "type", None)

                    # AI tool calls -> plan / search events
                    tool_calls = getattr(msg, "tool_calls", None) or []
                    for tc in tool_calls:
                        tc_id = tc.get("id") or ""
                        if tc_id and tc_id in seen_tool_calls:
                            continue
                        if tc_id:
                            seen_tool_calls.add(tc_id)
                        name = tc.get("name", "tool")
                        args = tc.get("args", {}) or {}
                        text = _summarize_tool_call(name, args)
                        kind = {
                            "tavily_search": "search",
                            "think_tool": "plan",
                            "write_todos": "status",
                        }.get(name, "status")
                        yield {"type": kind, "tool": name, "text": text}

                    # Tool results -> intel events
                    if msg_type == "tool":
                        name = getattr(msg, "name", "tool")
                        content = msg.content
                        if isinstance(content, list):
                            content = " ".join(
                                str(c.get("text", c)) if isinstance(c, dict) else str(c)
                                for c in content
                            )
                        content = str(content)
                        if name == "tavily_search":
                            preview = content[:600] + ("…" if len(content) > 600 else "")
                            yield {"type": "intel", "tool": name, "text": preview}

                    # Final AI text answer
                    if msg_type == "ai":
                        content = msg.content
                        if isinstance(content, list):
                            content = "".join(
                                c.get("text", "") if isinstance(c, dict) else str(c)
                                for c in content
                            )
                        if isinstance(content, str) and content.strip():
                            final_text = content

                        # Fallback: some thinking models put the answer in a
                        # reasoning channel and leave `content` empty.
                        kwargs = getattr(msg, "additional_kwargs", {}) or {}
                        reasoning = (
                            kwargs.get("reasoning_content")
                            or kwargs.get("reasoning")
                            or ""
                        )
                        if isinstance(reasoning, str) and reasoning.strip():
                            reasoning_fallback = reasoning

        if not final_text.strip() and reasoning_fallback.strip():
            final_text = reasoning_fallback

        yield {"type": "final", "text": final_text}
    except Exception as exc:  # noqa: BLE001 - surface to the UI
        yield {"type": "error", "text": f"{type(exc).__name__}: {exc}"}
