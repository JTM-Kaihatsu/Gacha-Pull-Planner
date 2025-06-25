
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
