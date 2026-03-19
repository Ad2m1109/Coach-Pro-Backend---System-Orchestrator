"""
Global Configuration for the Football Coach Backend
"""

import os

# LLM configuration
# Prefer environment variables so production doesn't rely on hardcoded ngrok URLs.
REMOTE_LLM_URL = os.environ.get(
    "REMOTE_LLM_URL",
    "https://2a8a-34-12-240-241.ngrok-free.app/generate",
)
LLM_MODE = os.environ.get("LLM_MODE", "remote")  # "remote" or "local"
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "60"))  # seconds
