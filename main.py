"""
Entry point for the Multi-Agent Travel Planner.

Modes:
  python main.py              → starts the Telegram bot
  python main.py --cli        → interactive CLI mode (no Telegram required)
"""
import asyncio
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


async def run_cli() -> None:
    """Interactive CLI mode for testing the travel graph without Telegram."""
    from src.graph import travel_graph

    session_id = "cli-session"
    config = {"configurable": {"thread_id": session_id}}
    print("\n🌍 Multi-Agent Travel Planner (CLI mode)")
    print("Type your travel request or 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not user_input:
            continue

        initial_state = {
            "messages": [],
            "session_id": session_id,
            "user_input": user_input,
            "trip_request": None,
            "orchestrator_output": None,
            "final_output": None,
        }

        print("\nAgent: Thinking...\n")
        try:
            result = await travel_graph.ainvoke(initial_state, config=config)
            reply = result.get("final_output") or "I could not process your request."
        except Exception as exc:
            reply = f"Error: {exc}"

        print(f"Agent:\n{reply}\n")
        print("-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Agent Travel Planner")
    parser.add_argument("--cli", action="store_true", help="Run in interactive CLI mode")
    args = parser.parse_args()

    if args.cli:
        asyncio.run(run_cli())
    else:
        from src.bot import run_telegram_bot
        run_telegram_bot()


if __name__ == "__main__":
    main()
