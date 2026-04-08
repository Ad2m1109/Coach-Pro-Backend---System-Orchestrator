"""
Global Configuration for the Football Coach Backend
"""

import os

from env_loader import load_backend_env

load_backend_env()

# LLM configuration
# Prefer environment variables so production doesn't rely on hardcoded ngrok URLs.
REMOTE_LLM_URL = os.environ.get(
    "REMOTE_LLM_URL",
    "https://9c20-34-105-127-148.ngrok-free.app/generate",
)
LLM_MODE = os.environ.get("LLM_MODE", "remote")  # "remote" or "local"
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "60"))  # seconds
