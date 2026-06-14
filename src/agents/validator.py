from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import OPENAI_API_KEY, OPENAI_MODEL
from src.state import TravelState


VALIDATOR_SYSTEM = """You are the Trip Plan Validator and Summarizer.

INPUT: Raw outputs from Flights, Hotels, and Activities agents combined with the original trip request.

IF INPUT CONTAINS AN ERROR: Rephrase it politely and clearly.

YOUR RESPONSIBILITIES:

1. VALIDATE
   - Ensure travel dates are consistent throughout.
   - Ensure hotel stay covers the full trip duration.
   - Ensure activities fit within the number of days.
   - Silently correct small conflicts.

2. CURATE
   Flights: Select the 2-3 best options. Prefer non-stop or minimal layovers. Balance price vs duration.
   Hotels: Select top 2-3. Prefer 4-star+ (or best available). Prioritize central location and value.
   Activities: Select 5-8 experiences. Distribute by day. Balance iconic and local. Avoid overloading any day.

3. PRESENT IN THIS STRUCTURE (always use this format):
───────────────────────────────────────
✈️ TRIP OVERVIEW
  • Route: [Origin] → [Destination]
  • Dates: [Departure] to [Return] ([N] nights)
  • Travelers: [N]
  • Style: [travel_style if provided]

✈️ FLIGHTS (Top Picks)
  [2-3 options with airline, times, stops, price in INR]

🏨 HOTELS (Top Picks)
  [2-3 options with name, stars, price/night, total cost, neighborhood]

🗓️ DAILY ITINERARY
  Day 1 — [Date]: [2-3 activities]
  Day 2 — [Date]: [2-3 activities]
  ...

💡 TIPS
  [2-3 practical travel tips for this destination]
───────────────────────────────────────

Keep the response readable, structured, and professional.
Do NOT mention internal agents, workflow steps, or system details.
All prices in INR."""


async def validate_node(state: TravelState) -> dict:
    """
    Validates, curates, and formats the orchestrator's combined output
    into a clean, user-facing travel itinerary.
    """
    llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0.3)

    trip = state.get("trip_request", {})
    raw_output = state.get("orchestrator_output", "")

    context = f"TRIP REQUEST:\n{trip}\n\nRAW AGENT OUTPUTS:\n{raw_output}"

    messages = [
        SystemMessage(content=VALIDATOR_SYSTEM),
        HumanMessage(content=context),
    ]

    response = await llm.ainvoke(messages)
    return {"final_output": response.content}
