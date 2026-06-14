from typing import TypedDict, Optional, List, Annotated
from langgraph.graph.message import add_messages


class TripRequest(TypedDict, total=False):
    origin: str
    destination: str
    departure_date: str        # YYYY-MM-DD
    return_date: str           # YYYY-MM-DD
    error: bool
    error_msg: str
    number_of_travelers: int
    budget: Optional[float]
    travel_style: Optional[str]   # budget | mid-range | luxury
    preferences: Optional[List[str]]
    hotel_preferences: Optional[dict]
    flight_preferences: Optional[dict]


class TravelState(TypedDict):
    messages: Annotated[list, add_messages]
    session_id: str
    user_input: str
    trip_request: Optional[TripRequest]
    orchestrator_output: Optional[str]
    final_output: Optional[str]
