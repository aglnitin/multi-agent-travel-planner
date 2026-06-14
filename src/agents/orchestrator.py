import asyncio
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from src.config import OPENAI_API_KEY, OPENAI_MODEL
from src.state import TravelState, TripRequest
from src.tools.flights import search_flights
from src.tools.hotels import search_hotels
from src.tools.activities import search_local_sightseeing, search_tavily


FLIGHTS_SYSTEM = """You are a Flights Search Agent.

Input: Trip request with origin, destination, dates, traveler count, and optional flight preferences.

INSTRUCTIONS:
1. Call the search_flights tool ONCE.
2. Use the city names to determine the correct IATA airport codes.
3. Present 4-6 round-trip flight options.
4. For each option include: airline, flight number, departure/arrival times, duration, stops, price in INR.
5. Highlight the best value option.
6. Do not guess — use only data returned by the tool.
No explanations or preamble."""


HOTELS_SYSTEM = """You are a Hotels Search Agent.

Input: Trip request with destination, check-in/out dates, travelers, and optional hotel preferences.

INSTRUCTIONS:
1. Call the search_hotels tool ONCE.
2. Present 5-7 hotel options.
3. For each option include: name, star rating, review score, neighborhood, price per night (INR), total stay cost (INR), key amenities.
4. Prefer 4-star+ when no budget constraint is given.
5. Do not guess — use only data returned by the tool.
No explanations or preamble."""


ACTIVITIES_SYSTEM = """You are an Activities & Sightseeing Agent.

Input: Trip request with destination, trip duration, and user preferences.

INSTRUCTIONS:
1. Call search_local_sightseeing AND search_tavily — both, once each.
2. Consolidate results into 8-12 diverse activity options.
3. Include a mix of iconic attractions and unique/local experiences.
4. For each activity include: name, type, rating, why it's worth visiting, estimated duration, and any entry fees if known.
5. Group loosely by area or theme (the Validator will assign days).
Do not guess — use only data returned by the tools.
No explanations or preamble."""


def _make_trip_prompt(trip: TripRequest) -> str:
    """Format the structured trip request as a prompt string for sub-agents."""
    return json.dumps({k: v for k, v in trip.items() if v is not None and k not in ("error", "error_msg")}, indent=2)


async def _run_agent(model_name: str, system_prompt: str, tools: list, trip_prompt: str) -> str:
    """Create a ReAct agent and invoke it asynchronously."""
    llm = ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY, temperature=0)
    agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt)
    result = await agent.ainvoke({"messages": [HumanMessage(content=trip_prompt)]})
    return result["messages"][-1].content


async def orchestrate_node(state: TravelState) -> dict:
    """
    Runs Flights, Hotels, and Activities sub-agents in parallel,
    then combines their outputs into a single structured summary for the Validator.
    """
    trip = state["trip_request"]
    trip_prompt = _make_trip_prompt(trip)

    flights_task = _run_agent(OPENAI_MODEL, FLIGHTS_SYSTEM, [search_flights], trip_prompt)
    hotels_task = _run_agent(OPENAI_MODEL, HOTELS_SYSTEM, [search_hotels], trip_prompt)
    activities_task = _run_agent(
        OPENAI_MODEL, ACTIVITIES_SYSTEM, [search_local_sightseeing, search_tavily], trip_prompt
    )

    # Parallel execution — same pattern as n8n running all 3 agent tools concurrently
    flights_result, hotels_result, activities_result = await asyncio.gather(
        flights_task, hotels_task, activities_task, return_exceptions=True
    )

    def safe(result, label: str) -> str:
        if isinstance(result, Exception):
            return f"[{label} agent failed: {result}]"
        return result

    combined = (
        f"=== FLIGHTS ===\n{safe(flights_result, 'Flights')}\n\n"
        f"=== HOTELS ===\n{safe(hotels_result, 'Hotels')}\n\n"
        f"=== ACTIVITIES ===\n{safe(activities_result, 'Activities')}"
    )

    return {"orchestrator_output": combined}
