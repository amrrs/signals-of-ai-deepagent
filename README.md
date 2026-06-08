# Signals of AI

Signals of AI is a live competitive-intelligence demo that turns a target company into a streamed OSINT investigation and a concise executive dossier.

The app is powered by Nebius Token Factory for runtime model access, Tavily for live web intelligence, and LangChain DeepAgents for multi-step research and reasoning.

## What It Does

- Accepts a target company, optional client context, and optional focus areas.
- Streams each investigation step to a cyber-style surveillance terminal.
- Uses Tavily to search current public web sources for products, pricing, funding, news, positioning, and risks.
- Uses a LangChain DeepAgent with Nebius-hosted models to plan, search, reason, and compile the final dossier.
- Supports a per-run Nebius Token Factory key from the UI, with `.env` fallback for local demos.

## Tech Stack

- `FastAPI` backend
- Server-Sent Events for live agent streaming
- Static HTML/CSS/JavaScript front-end
- `langchain-nebius` for Nebius model access
- `deepagents` for agent orchestration
- `tavily-python` for web search

## Project Structure

```text
.
├── server.py              # FastAPI app and streaming API
├── static/
│   └── index.html         # Main Signals of AI interface
├── spy_agent/
│   ├── agent.py           # DeepAgent setup, prompt, and stream adapter
│   └── tools.py           # Tavily search and reasoning tools
└── pyproject.toml         # Python dependencies
```

## Setup

This project requires Python 3.12 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Create a `.env` file in the project root:

```bash
NEBIUS_TOKEN_FACTORY_API_KEY=your_nebius_token_factory_key
TAVILY_API_KEY=your_tavily_api_key
```

## Run Locally

```bash
source .venv/bin/activate
python server.py
```

Open the app at:

```text
http://127.0.0.1:8000
```

## How To Use

1. Enter a target company, such as `Notion`, `Stripe`, or `OpenAI`.
2. Optionally enter your company as the client to frame the recommendations.
3. Optionally add focus areas, such as `pricing`, `AI features`, or `enterprise GTM`.
4. Leave the Nebius Token Factory key blank to use `.env`, or paste a runtime key for that mission.
5. Click `Deploy Agent` and watch the live feed while the dossier is generated.

## Environment Variables

- `NEBIUS_TOKEN_FACTORY_API_KEY`: Nebius Token Factory key for model access.
- `TAVILY_API_KEY`: Required for live web search.
- `SPY_MODEL`: Optional model override. Defaults to `nvidia/Nemotron-3-Ultra-550b-a55b`.

## API

`GET /`

Serves the main Signals of AI interface.

`GET /api/health`

Returns backend status and the configured model name.

`POST /api/deploy`

Starts a mission and streams Server-Sent Events.

Example payload:

```json
{
  "target": "Stripe",
  "client": "Adyen",
  "focus": "pricing, enterprise sales, AI payments",
  "nebius_token_factory_api_key": ""
}
```

## Notes

Signals of AI only uses public web information surfaced through Tavily. The generated dossier should be treated as an AI-assisted research brief: verify critical claims and source URLs before making business decisions.
