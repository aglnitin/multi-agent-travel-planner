# Multi-Agent Travel Planner

A production-grade multi-agent travel planning bot built with **LangGraph + LangChain**, ported from an n8n visual workflow. Send a natural-language travel request via Telegram (or CLI) and get back a curated itinerary with real flights, hotels, and activities.

## Demo

```
You: I want to fly from Delhi to Bangkok on July 10 2026, return July 17, 2 adults, mid-range

Agent:
───────────────────────────────────────
✈️ TRIP OVERVIEW
  • Route: Delhi → Bangkok
  • Dates: 2026-07-10 to 2026-07-17 (7 nights)
  • Travelers: 2  •  Style: Mid-range

✈️ FLIGHTS (Top Picks)
  1. IndiGo  DEL→BKK  15:45–21:45  Non-stop  ₹65,704
  2. Air India         11:00–16:55  Non-stop  ₹71,173

🏨 HOTELS (Top Picks)
  1. Prince Palace Bangkok  ★★★★  ₹2,909/night  ₹20,360 total
  2. Away Bangkok Riverside ★★★★½ ₹4,032/night  ₹28,222 total

🗓️ DAILY ITINERARY
  Day 1 — Jul 10: Arrival → Asiatique Waterfront
  Day 2 — Jul 11: Grand Palace, Wat Phra Kaew, Wat Arun
  ...

💡 TIPS
  1. Carry Thai Baht cash for local markets.
  2. Use BTS Skytrain to skip traffic.
───────────────────────────────────────
```

## Architecture

```
User Message (Telegram / CLI)
         │
         ▼
  ┌─────────────┐
  │ Coordinator │  Parses input → structured TripRequest (Pydantic)
  └──────┬──────┘
         │
    error? ──► Return friendly error message
         │
         ▼
  ┌──────────────────────────────────────────┐
  │           Orchestrator                   │
  │  ┌──────────┐  ┌────────┐  ┌──────────┐ │
  │  │ Flights  │  │ Hotels │  │Activities│ │  ← parallel (asyncio.gather)
  │  │  Agent   │  │  Agent │  │  Agent   │ │
  │  │ SerpAPI  │  │ SerpAPI│  │Serp+Tavly│ │
  │  └──────────┘  └────────┘  └──────────┘ │
  └──────────────────┬───────────────────────┘
                     │
                     ▼
             ┌───────────────┐
             │   Validator   │  Curates top picks, formats final itinerary
             └───────┬───────┘
                     │
              Final Response
```

**Stack:** LangGraph (workflow) · LangChain (LLM + tools) · OpenAI gpt-4o-mini · SerpAPI · Tavily · python-telegram-bot

## Project Structure

```
├── src/
│   ├── graph.py              # LangGraph StateGraph — nodes, edges, routing, memory
│   ├── state.py              # TravelState TypedDict
│   ├── config.py             # Environment variable loader
│   ├── bot.py                # Telegram bot handler
│   ├── agents/
│   │   ├── coordinator.py    # Structured output agent (input validation)
│   │   ├── orchestrator.py   # Parallel sub-agent runner
│   │   └── validator.py      # Curate + format final itinerary
│   └── tools/
│       ├── flights.py        # SerpAPI Google Flights @tool
│       ├── hotels.py         # SerpAPI Google Hotels @tool
│       └── activities.py     # SerpAPI Local + Tavily @tool
├── main.py                   # Entry point (--cli or Telegram mode)
├── walkthrough.html          # Visual code walkthrough (open in browser)
├── INTERVIEW_GUIDE.md        # EM interview deep-dive
├── requirements.txt
└── .env.example
```

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/aglnitin/multi-agent-travel-planner.git
cd multi-agent-travel-planner
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up API keys

```bash
cp .env.example .env
```

Edit `.env` with your keys:

| Key | Where to get it |
|-----|----------------|
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) |
| `SERPAPI_API_KEY` | [serpapi.com](https://serpapi.com) |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com) |
| `TELEGRAM_BOT_TOKEN` | [@BotFather](https://t.me/BotFather) on Telegram |

### 3. Run

**CLI mode** (no Telegram needed — great for testing):
```bash
python main.py --cli
```

**Telegram bot mode:**
```bash
python main.py
```

## How It Works

### LangChain vs LangGraph

- **LangChain** = the AI toolkit: `ChatOpenAI`, `@tool`, `with_structured_output`, `create_react_agent`
- **LangGraph** = the workflow engine: `StateGraph`, conditional routing, `MemorySaver` per-user conversation

They are not alternatives — LangGraph orchestrates *when* each LangChain component runs.

### Key Patterns

| Pattern | Where | What it does |
|---------|-------|-------------|
| Structured output | Coordinator | Forces LLM to return validated Pydantic JSON — no freeform text |
| ReAct loop | Sub-agents | Reason → call tool → observe result → answer |
| Parallel fan-out | Orchestrator | `asyncio.gather` cuts latency from ~15s to ~5s |
| Conditional routing | graph.py | `add_conditional_edges` skips API calls if input is incomplete |
| Conversation memory | graph.py | `MemorySaver` keyed by Telegram `chat_id` — follow-up messages work |

### Agent Flow (per message)

1. **Coordinator** — Calls OpenAI with structured output schema. If origin/destination/dates are missing → returns error message immediately (no API calls wasted).
2. **Orchestrator** — Spins up 3 ReAct agents in parallel. Each agent reasons about airport codes / hotel classes, calls the relevant SerpAPI endpoint, and returns formatted results.
3. **Validator** — Receives all raw data, selects top 2-3 flights and hotels, distributes activities across days, and returns the final formatted itinerary.

## n8n → Python Mapping

| n8n Node | Python equivalent |
|----------|------------------|
| `CoOrdinator` agent | `coordinator_node` + `TripRequestSchema` (Pydantic) |
| `If` (error routing) | `add_conditional_edges` + `route_after_coordinator` |
| `Planner` agent | Removed — logic absorbed into `orchestrate_node` |
| `Travel Agent (Orchestrator)` | `orchestrate_node` with `asyncio.gather` |
| `Flights AI Agent` (agentTool) | `_run_agent(... [search_flights])` |
| `Hotels AI Agent` (agentTool) | `_run_agent(... [search_hotels])` |
| `Activities AI Agent` (agentTool) | `_run_agent(... [search_local, search_tavily])` |
| `Validator and Summarizer` | `validate_node` |
| `memoryBufferWindow` | `MemorySaver(thread_id=chat_id)` |
| `Telegram Trigger` + Send nodes | `src/bot.py` |

## Resources

- [walkthrough.html](walkthrough.html) — Open in browser for a visual step-by-step code trace
- [INTERVIEW_GUIDE.md](INTERVIEW_GUIDE.md) — Architecture decisions, scaling strategy, EM interview Q&A
- [LangGraph docs](https://langchain-ai.github.io/langgraph/)
- [SerpAPI docs](https://serpapi.com/search-api)
- [Tavily docs](https://docs.tavily.com)
