import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
