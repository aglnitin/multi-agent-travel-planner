"""
Telegram bot integration — mirrors the n8n Telegram Trigger + Send nodes.

Each Telegram chat gets its own LangGraph thread (session_id = chat_id),
which maps to n8n's Simple Memory keyed by message.chat.id.
"""
import asyncio
import logging
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from src.config import TELEGRAM_BOT_TOKEN
from src.graph import travel_graph

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Receives a Telegram message, runs it through the LangGraph travel workflow,
    and sends back the final response.
    """
    chat_id = update.message.chat_id
    user_text = update.message.text or ""

    logger.info("Received message from chat_id=%s: %s", chat_id, user_text[:80])

    # Send a typing indicator while processing
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    initial_state = {
        "messages": [],
        "session_id": str(chat_id),
        "user_input": user_text,
        "trip_request": None,
        "orchestrator_output": None,
        "final_output": None,
    }

    # thread_id = chat_id enables per-conversation memory (MemorySaver)
    config = {"configurable": {"thread_id": str(chat_id)}}

    try:
        result = await travel_graph.ainvoke(initial_state, config=config)
        reply = result.get("final_output") or "I'm sorry, I couldn't process your request."
    except Exception as exc:
        logger.exception("Graph execution failed for chat_id=%s", chat_id)
        reply = f"An unexpected error occurred. Please try again. ({exc})"

    # Telegram messages are capped at 4096 chars; split if needed
    for chunk in _split_message(reply, 4096):
        await update.message.reply_text(chunk, parse_mode=None)


def _split_message(text: str, max_len: int):
    """Split long messages into chunks that fit within Telegram's limit."""
    for i in range(0, len(text), max_len):
        yield text[i: i + max_len]


def run_telegram_bot() -> None:
    """Start the Telegram bot with long-polling."""
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN is not set in .env")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Telegram bot is running. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)
