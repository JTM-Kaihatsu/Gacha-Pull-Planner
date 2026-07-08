
"""config.py
Environment‑variable helper to keep secrets out of code.
"""
import os
from dotenv import load_dotenv

load_dotenv()

def get_openai_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing. Set it in .env or host vars.")
    return key

def get_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o")

# Origins allowed to call the API (CORS). Override with the ALLOWED_ORIGINS env
# var (comma-separated). The default covers local dev plus the deployed demo
# frontend, so CORS works out of the box without extra host configuration.
DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:5173,"
    "https://gacha-pull-planner.vercel.app"
)


def get_allowed_origins() -> list:
    origins = os.getenv("ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS)
    return [o.strip() for o in origins.split(",") if o.strip()]
