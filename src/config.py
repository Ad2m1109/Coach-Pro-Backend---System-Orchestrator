"""
Global Configuration for the Football Coach Backend
"""

# Remote LLM (Colab-hosted Mistral 7B)
REMOTE_LLM_URL = "https://0de8-34-16-246-121.ngrok-free.app/generate"
# Mode: "remote" or "local" (for future local fallback)
LLM_MODE = "remote"
LLM_TIMEOUT = 60  # seconds
