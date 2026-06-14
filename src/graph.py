from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.state import TravelState
from src.agents.coordinator import coordinator_node, route_after_coordinator
from src.agents.orchestrator import orchestrate_node
from src.agents.validator import validate_node


async def error_end_node(state: TravelState) -> dict:
    """Pass-through node that surfaces the error message as final_output."""
    return {"final_output": state.get("trip_request", {}).get("error_msg", "Something went wrong.")}


def build_graph():
    """
    Constructs the LangGraph StateGraph:

    START
      └─► coordinator        (parse user input → TripRequest)
              │
        ┌─────┴──────┐
    error?         no error
        │               │
    error_end       orchestrate    (flights + hotels + activities in parallel)
        │               │
       END          validate       (curate & format)
                        │
                       END
    """
    graph = StateGraph(TravelState)

    graph.add_node("coordinator", coordinator_node)
    graph.add_node("error_end", error_end_node)
    graph.add_node("orchestrate", orchestrate_node)
    graph.add_node("validate", validate_node)

    graph.add_edge(START, "coordinator")

    graph.add_conditional_edges(
        "coordinator",
        route_after_coordinator,
        {
            "error_end": "error_end",
            "orchestrate": "orchestrate",
        },
    )

    graph.add_edge("error_end", END)
    graph.add_edge("orchestrate", "validate")
    graph.add_edge("validate", END)

    # MemorySaver enables per-session conversation memory keyed by thread_id
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# Singleton — import this in bot.py and main.py
travel_graph = build_graph()
