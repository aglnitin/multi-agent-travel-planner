import json
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import OPENAI_API_KEY, OPENAI_MODEL
from src.state import TravelState, TripRequest


COORDINATOR_SYSTEM = """You are the Trip Request Coordinator.

YOUR JOB: Extract structured travel details from the user message and return valid JSON.

Current date/time: {current_time}

RULES:
- Extract origin, destination, departure_date, return_date.
- If ANY of these four are missing → set error: true with a friendly error_msg listing missing fields.
- If the message is a greeting, small talk, or has NO travel request → set error: true with a warm message greeting back and asking for trip details.
- Default number_of_travelers to 1 if not mentioned.
- Extract preferences, travel_style, budget if present.
- Use city names (not airport codes) for origin and destination.
- Dates MUST be in YYYY-MM-DD format.
- Dates must not be in the past.
- ALWAYS return a single valid JSON object. Never reply with plain text.

Return JSON with these fields:
{{
  "origin": "city name or null",
  "destination": "city name or null",
  "departure_date": "YYYY-MM-DD or null",
  "return_date": "YYYY-MM-DD or null",
  "error": true/false,
  "error_msg": "message if error, empty string if no error",
  "number_of_travelers": 1,
  "budget": null or number,
  "travel_style": null or "budget"/"mid-range"/"luxury",
  "preferences": [],
  "hotel_preferences": {{}},
  "flight_preferences": {{}}
}}"""


class TripRequestSchema(BaseModel):
    origin: Optional[str] = Field(None)
    destination: Optional[str] = Field(None)
    departure_date: Optional[str] = Field(None, description="YYYY-MM-DD")
    return_date: Optional[str] = Field(None, description="YYYY-MM-DD")
    error: bool = Field(...)
    error_msg: str = Field(...)
    number_of_travelers: int = Field(default=1)
    budget: Optional[float] = None
    travel_style: Optional[str] = None
    preferences: Optional[List[str]] = None
    hotel_preferences: Optional[dict] = None
    flight_preferences: Optional[dict] = None


async def coordinator_node(state: TravelState) -> dict:
    """
    Parses raw user input into a structured TripRequest.
    Sets error=True and error_msg if input is incomplete or not a travel request.
    """
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0,
    )
    structured_llm = llm.with_structured_output(TripRequestSchema, method="json_mode")

    system_prompt = COORDINATOR_SYSTEM.format(
        current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["user_input"]),
    ]

    try:
        result: TripRequestSchema = await structured_llm.ainvoke(messages)
        trip_request: TripRequest = result.model_dump()
    except Exception as exc:
        trip_request = {
            "error": True,
            "error_msg": f"Sorry, I encountered an error parsing your request. Please try again. ({exc})",
        }

    return {
        "trip_request": trip_request,
        "final_output": trip_request.get("error_msg", "") if trip_request.get("error") else None,
    }


def route_after_coordinator(state: TravelState) -> str:
    """Conditional edge: route to error response or orchestrator."""
    trip = state.get("trip_request") or {}
    if trip.get("error", True):
        return "error_end"
    return "orchestrate"
