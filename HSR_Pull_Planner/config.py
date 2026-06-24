
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

def get_allowed_origins() -> list:
    origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173")
    return [o.strip() for o in origins.split(",")]
