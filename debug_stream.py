import asyncio
from dotenv import load_dotenv
load_dotenv()
from spy_agent.agent import build_spy_agent, build_mission_prompt

async def main():
    agent = build_spy_agent()
    mission = build_mission_prompt("Linear", None, "pricing")
    last_msgs = []
    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": mission}]},
        stream_mode="updates",
        config={"recursion_limit": 100},
    ):
        for node, update in chunk.items():
            msgs = (update or {}).get("messages", []) if isinstance(update, dict) else []
            for m in msgs:
                tc = getattr(m, "tool_calls", None) or []
                c = m.content
                ctype = type(c).__name__
                clen = len(c) if isinstance(c, (str, list)) else 0
                print(f"node={node:18} type={getattr(m,'type',None):6} tool_calls={len(tc)} content={ctype}({clen})")
                last_msgs.append(m)
    print("\n=== LAST AI MESSAGE CONTENT ===")
    for m in reversed(last_msgs):
        if getattr(m, "type", None) == "ai":
            print("content repr:", repr(m.content)[:500])
            print("tool_calls:", [t.get("name") for t in (getattr(m,"tool_calls",None) or [])])
            print("additional_kwargs:", repr(getattr(m, "additional_kwargs", {}))[:800])
            print("response_metadata:", repr(getattr(m, "response_metadata", {}))[:800])
            break

asyncio.run(main())
