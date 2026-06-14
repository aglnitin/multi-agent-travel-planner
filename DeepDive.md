# Multi-Agent Travel Planner — DeepDive

---

## 1. What We Built and Why

### The Problem
Build a travel planning assistant that accepts a natural-language request ("I want to fly from Delhi to Bangkok June 20–27 for 2 people, budget travel") and returns a curated, validated itinerary with real flights, hotels, and activities.

---

## 2. Framework Decision: LangGraph vs CrewAI vs RAG

### Why LangGraph?
LangGraph is a **state-machine orchestration library** built on LangChain. It models agentic workflows as directed graphs where nodes are functions and edges carry typed state.

**Key reasons for this use case:**
1. **Explicit conditional routing** — the `If` node in n8n (route error vs. orchestrate) maps naturally to `add_conditional_edges`. CrewAI has no equivalent.
2. **Fine-grained state control** — we need to pass `TripRequest` typed state between 4 distinct phases. LangGraph's `TypedDict` state is checkpointed and typed.
3. **Parallel execution** — `asyncio.gather` inside an orchestrate node mirrors n8n's parallel tool-agent execution.
4. **Built-in memory** — `MemorySaver` (or `RedisCheckpointSaver` in prod) handles multi-turn conversation per session, replacing n8n's `memoryBufferWindow`.
5. **Production observability** — native LangSmith integration for tracing every token and tool call.

### Why NOT CrewAI?
- CrewAI uses a **role + task** abstraction — great for "agent A writes, agent B reviews" patterns.
- It lacks first-class conditional routing between agents.
- Less control over state schemas; harder to validate structured output between agents.
- Better suited for document-processing pipelines, not request → API → validate workflows.

### Why NOT RAG?
RAG (Retrieval-Augmented Generation) adds a vector database lookup step. It's appropriate when the agent needs to reference a knowledge base (e.g., "what are the visa rules for Thailand?"). This workflow retrieves **live data** from SerpAPI and Tavily — not static documents. RAG would add latency and cost with no benefit here.

**Verdict: LangGraph wins for this use case because the workflow is a stateful pipeline with conditional branching, not a task-delegation pattern.**

---

## 3. Architecture Deep Dive

### Agent Roles (Separation of Concerns)

```
User Message
     │
     ▼
┌────────────┐
│ Coordinator │  → Structured output (Pydantic schema)
│            │    → Validates input completeness
│            │    → Handles greetings / off-topic
└────┬───────┘
     │
  error? ──► Return error_msg directly to user
     │
     ▼
┌─────────────────────────────────────────┐
│              Orchestrator               │
│   ┌──────────┐ ┌───────┐ ┌──────────┐ │
│   │ Flights  │ │Hotels │ │Activities│ │  ← Parallel asyncio.gather
│   │  Agent   │ │ Agent │ │  Agent   │ │
│   │(SerpAPI) │ │(Serp) │ │(Serp +   │ │
│   └──────────┘ └───────┘ │ Tavily)  │ │
│                           └──────────┘ │
└─────────────────┬───────────────────────┘
                  │
                  ▼
          ┌───────────────┐
          │   Validator   │  → Curates options (top 2-3 flights, top 2-3 hotels)
          │  & Summarizer │  → Formats final response
          └───────┬───────┘
                  │
            Final Output (Telegram / CLI)
```

### LangGraph State Flow

```python
class TravelState(TypedDict):
    messages: Annotated[list, add_messages]   # conversation history (MemorySaver)
    session_id: str                            # Telegram chat_id
    user_input: str                            # raw user message
    trip_request: Optional[TripRequest]        # structured data from Coordinator
    orchestrator_output: Optional[str]         # raw combined agent output
    final_output: Optional[str]                # clean itinerary for the user
```

State is **immutable per step** — each node returns a partial dict that LangGraph merges. This makes debugging deterministic: you can replay any step from a checkpoint.

### Key Design Patterns Used

| Pattern | Where | Why |
|---|---|---|
| Structured Output | Coordinator | Forces LLM to return validated Pydantic schema, not freeform text |
| ReAct (Reason + Act) | Flights/Hotels/Activities agents | Agents decide which tool to call based on reasoning steps |
| Parallel Fan-Out | Orchestrate node | 3 API calls run concurrently via `asyncio.gather`, cutting latency by ~3x |
| Checkpoint / Memory | MemorySaver | Per-session conversation state persists across turns |
| Validator Pattern | Validator node | Separates raw data retrieval from quality control / formatting |
| Error Boundary | Coordinator → conditional edge | Fail fast before expensive API calls; return user-friendly message |

---

## 4. Data Flow Example

**Input:** "I want to fly Mumbai to Paris, leaving July 10 returning July 17, 2 adults, prefer 4-star hotels"

**Step 1 — Coordinator output:**
```json
{
  "origin": "Mumbai",
  "destination": "Paris",
  "departure_date": "2026-07-10",
  "return_date": "2026-07-17",
  "number_of_travelers": 2,
  "hotel_preferences": {"star_rating_min": 4},
  "error": false,
  "error_msg": ""
}
```

**Step 2 — Orchestrate (parallel):**
- Flights agent → calls `search_flights("BOM", "CDG", "2026-07-10", "2026-07-17", 2)`
- Hotels agent → calls `search_hotels("Paris", "2026-07-10", "2026-07-17", 2, "4")`
- Activities agent → calls `search_local_sightseeing(...)` + `search_tavily(...)`

**Step 3 — Validator:**
- Selects top 2-3 flight options, top 2-3 hotels
- Assigns activities across 7 days
- Returns formatted itinerary

---

## 5. External API Integration

### SerpAPI
- **Google Flights** (`engine=google_flights`): real-time flight data with prices, stops, duration
- **Google Hotels** (`engine=google_hotels`): real hotel listings with ratings and prices
- **Google Local** (`engine=google_local`): local businesses and attractions with ratings

**Rate limits:** 100 searches/month on free tier. Production needs a paid plan.

### Tavily AI Search
- AI-optimized web search returning structured JSON with an `answer` field.
- Used by Activities agent to supplement SerpAPI's local results with web context.
- Particularly useful for "what to do in X" type queries where local search is sparse.

---

## 6. Memory and Multi-Turn Conversations

In n8n, `memoryBufferWindow` was keyed by `message.chat.id` and attached to the Coordinator agent. In LangGraph:

```python
memory = MemorySaver()                        # in-process, for dev
graph.compile(checkpointer=memory)            # attach to graph

config = {"configurable": {"thread_id": str(chat_id)}}
graph.ainvoke(state, config=config)           # each chat gets isolated memory
```

This means the Coordinator can understand follow-up messages:
- Turn 1: "I want to go to Bali"
- Turn 2: "actually make it 3 people" ← Coordinator sees the full conversation history

**In production, swap `MemorySaver` for `RedisCheckpointSaver`** to survive restarts and share state across multiple pods.

---

## 7. Scalability Considerations

### Current Architecture Bottlenecks
1. **In-memory checkpointer** — lost on restart; single process
2. **SerpAPI rate limits** — free tier exhausted quickly under load
3. **No response caching** — same flight search repeated for identical queries

### Production Scaling Path

**Tier 1 (100 users/day):** Current implementation is fine.

**Tier 2 (10k users/day):**
```
Telegram → FastAPI webhook → Celery task queue
                                     │
                              LangGraph workers (async)
                                     │
                              Redis (MemorySaver + cache)
                                     │
                    SerpAPI / Tavily (with response cache, TTL=1h)
```

**Key changes:**
- Replace `MemorySaver` → `langgraph.checkpoint.redis.RedisSaver`
- Add response caching: same flight search within 1 hour returns cached result
- Horizontal scaling: multiple LangGraph worker processes
- Rate limit handling: exponential backoff + circuit breaker on SerpAPI calls

**Tier 3 (100k users/day):**
- Separate microservices for Flights, Hotels, Activities
- Dedicated API gateway with request deduplication
- SerpAPI Enterprise plan or direct airline/hotel APIs
- Add LangSmith for distributed tracing

---

## 8. Cost Optimization

### Token Usage Per Request (approximate, gpt-4o-mini)
| Agent | Input tokens | Output tokens | Cost (USD) |
|---|---|---|---|
| Coordinator | ~500 | ~200 | ~$0.00015 |
| Flights Agent (2 turns) | ~800 | ~400 | ~$0.00024 |
| Hotels Agent (2 turns) | ~800 | ~400 | ~$0.00024 |
| Activities Agent (3 turns) | ~1200 | ~600 | ~$0.00036 |
| Validator | ~2000 | ~800 | ~$0.00056 |
| **Total per itinerary** | **~5300** | **~2400** | **~$0.0016** |

SerpAPI cost: ~$0.01/search × 4 searches = $0.04/request.

**Cost reduction strategies:**
1. **Cache SerpAPI results** for identical origin/destination/date combos (TTL=1h). Most expensive part.
2. **Compress orchestrator output** before passing to Validator — strip raw JSON, keep only named fields.
3. **Model tiering**: Coordinator can use `gpt-4o-mini`. Validator may benefit from `gpt-4o` for quality but only run it on completed itineraries.
4. **Prompt caching**: Anthropic Claude has prompt caching; for OpenAI, minimize repetitive system prompts.

---

## 9. Error Handling Strategy

### Failure Modes and Mitigations

| Failure | Detection | Mitigation |
|---|---|---|
| SerpAPI timeout | `requests.Timeout` exception in tool | Return partial results; agent notes "flight data unavailable" |
| SerpAPI quota exceeded | HTTP 429 | Exponential backoff + fallback message |
| LLM returns invalid JSON (Coordinator) | Pydantic validation error | Retry with `with_structured_output` retry logic |
| Orchestrator sub-agent failure | `return_exceptions=True` in `asyncio.gather` | Remaining agents continue; error surfaced in output |
| Telegram API down | `telegram.error.NetworkError` | Retry queue via `python-telegram-bot`'s built-in retry |

### The n8n "retryOnFail" Pattern
In n8n, the Coordinator had `retryOnFail: true, maxTries: 2`. In Python, replicate with:
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4))
async def call_structured_llm(messages):
    return await structured_llm.ainvoke(messages)
```

---

## 10. Observability and Debugging

### LangSmith Integration (recommended for production)
```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__..."
os.environ["LANGCHAIN_PROJECT"] = "travel-agent"
```
Every graph run, tool call, LLM invocation, and token count is automatically traced. You can replay failed executions and see exactly where hallucinations or errors occurred.

### Structured Logging
```python
logger.info("chat_id=%s node=coordinator error=%s", chat_id, trip["error"])
logger.info("chat_id=%s node=orchestrate latency_ms=%d", chat_id, elapsed_ms)
```

### Metrics to Track
- P95 latency per node (Orchestrate is the slowest: ~5-8s with parallel API calls)
- SerpAPI quota consumption per day
- Coordinator error rate (% of messages that lack required fields)
- Validator LLM output token count (proxy for itinerary quality/verbosity)

---

## 11. Testing Strategy

### Unit Tests — Coordinator
```python
# Mock the LLM, verify structured output parsing
async def test_coordinator_extracts_dates():
    state = {"user_input": "Mumbai to Goa, June 5-10 2027, 2 people", ...}
    result = await coordinator_node(state)
    assert result["trip_request"]["departure_date"] == "2027-06-05"
    assert result["trip_request"]["error"] == False
```

### Unit Tests — Tools
```python
# Mock SerpAPI response, verify tool output format
def test_search_flights_returns_correct_structure(mock_serpapi):
    result = json.loads(search_flights.invoke({...}))
    assert "flights" in result
    assert all("price_INR" in f for f in result["flights"])
```

### Integration Tests — Full Graph
```python
async def test_full_graph_happy_path():
    result = await travel_graph.ainvoke(
        {"user_input": "Delhi to Bangkok, July 1-7 2027"},
        config={"configurable": {"thread_id": "test-1"}}
    )
    assert result["final_output"] is not None
    assert "FLIGHTS" in result["final_output"]
    assert "HOTELS" in result["final_output"]
```

### Prompt Regression Tests
Store golden-path inputs → expected output shape. Run after any system prompt change to detect regressions in output structure or quality.

---

## 12. Common Questions

### Q: "Why LangGraph over just using LangChain agents directly?"
**A:** LangChain agents (e.g., `AgentExecutor`) are stateless and don't support conditional branching between agents. LangGraph gives us an explicit state machine where we can (a) route based on intermediate results (error vs. no error), (b) checkpoint state at each step for fault tolerance, and (c) compose multiple agents with typed data contracts between them. For production workflows with multiple failure modes and multi-turn conversations, that explicitness is critical.

### Q: "How do you ensure the LLM doesn't hallucinate flight prices?"
**A:** The agents are grounded — they can only return data from the tools. The Flights/Hotels/Activities agents use ReAct (reason + act), meaning the LLM calls `search_flights` and then formats the returned JSON. It cannot invent prices or airlines because its output step only has tool results to work with. The Validator is the only free-form generation step, and it's explicitly instructed to only use data from the orchestrator's output.

### Q: "What happens if SerpAPI goes down during a user request?"
**A:** `asyncio.gather(return_exceptions=True)` catches individual agent failures without failing the whole request. If Flights fails but Hotels and Activities succeed, the Validator gets partial data and notes that flight information is temporarily unavailable. We also add retry logic with exponential backoff for transient failures. For persistent outages, a circuit breaker disables the API and we surface a graceful degradation message.

### Q: "How would you add RAG to this system?"
**A:** A natural extension is a "Destination Knowledge Base" — a vector store of visa requirements, safety advisories, seasonal tips, local customs. We'd add a fourth parallel agent: `knowledge_agent` that does a similarity search against this store. The Validator then incorporates this context into the "Tips" section. We'd use LangChain's `Chroma` or `Pinecone` vector store with periodic ingestion from government travel sites and travel blogs. The key decision is whether this runs as a fifth parallel agent (fast, adds latency of one vector search) or as a standalone Retriever tool available to the Activities agent.

### Q: "How would you handle 10x the current load?"
**A:** Three changes: (1) Move from in-process `MemorySaver` to `RedisSaver` so state persists across restarts and is shared across multiple worker processes. (2) Add a response cache (Redis, TTL=1h) keyed by `{origin}:{destination}:{departure}:{return}` — flight and hotel prices don't change minute-to-minute. (3) Deploy multiple async worker processes behind a message queue (Celery or RQ). The Telegram webhook pushes requests to the queue; workers consume them. This decouples request receipt from processing latency.

### Q: "How do you version-control your prompts?"
**A:** Prompts are strings in Python source files — they go through the same PR review and git history as code. For A/B testing, we parameterize the prompt via config and use LangSmith experiments to compare quality metrics (output token count, structured format compliance, user satisfaction proxy via Telegram reactions). We never change a production prompt without a corresponding LangSmith baseline comparison.

### Q: "What's your strategy for reducing LLM costs as the system scales?"
**A:** In order of impact: (1) **Cache SerpAPI responses** — this is 90% of the per-request cost. (2) **Compress the orchestrator's output** before sending it to the Validator — strip raw JSON, keep only the most relevant fields. This cuts Validator input tokens by ~60%. (3) **Route simple error cases earlier** — the Coordinator catches greetings and incomplete requests without touching SerpAPI at all. (4) **Batch similar requests** — if 5 users ask for Delhi→Goa on the same date within 1 minute, one SerpAPI call serves all.

### Q: "What's the biggest risk in this architecture?"
**A:** The Validator LLM step is a quality bottleneck — it receives all raw data and must curate, format, and validate without hallucinating. If the orchestrator returns low-quality data (e.g., SerpAPI returns sparse hotel results), the Validator can either over-fill from its training data or under-deliver. Mitigation: add explicit tool-call grounding even in the Validator (e.g., require it to quote hotel names verbatim from the input), and add output format validation (regex check for expected sections before sending to the user).

---

## 13. Architecture Decision Log

| Decision | Alternatives Considered | Rationale |
|---|---|---|
| LangGraph for orchestration | CrewAI, custom async pipeline | LangGraph's state machine + conditional edges map 1:1 to the n8n IF node pattern |
| `asyncio.gather` for parallel agents | Sequential calls, LangGraph's `Send` API | Parallel cuts P95 latency from ~15s to ~5s; `Send` adds complexity without benefit here |
| `gpt-4o-mini` for all nodes | `gpt-4o` for Validator, smaller for Coordinator | Cost: `gpt-4o-mini` is 15x cheaper; quality gap is acceptable for structured tasks |
| Direct SerpAPI HTTP calls | `langchain_community.SerpAPIWrapper` | LangChain wrapper only supports `google` engine; we need `google_flights`, `google_hotels`, `google_local` separately |
| Pydantic structured output for Coordinator | Regex parsing, JSON mode without schema | Pydantic gives type safety + automatic validation; catches missing fields at schema level |
| MemorySaver (in-process) | Redis, SQLite | Dev simplicity; swap to Redis before production with one line change |

---

*Generated from the n8n workflow: "Multi-Agent - Travel Chat final (V4)" + "Travel Agent (Orchestrator)"*
