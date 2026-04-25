"""
Global Configuration for the Football Coach Backend
"""

import os

from env_loader import load_backend_env

load_backend_env()

# LLM configuration
REMOTE_LLM_URL = os.environ.get("REMOTE_LLM_URL")
LLM_MODE = os.environ.get("LLM_MODE", "remote")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "60"))

# Database Configuration (Fallbacks removed, must be in .env)
DB_HOST = os.environ.get("DB_HOST", "127.0.0.1")
DB_USER = os.environ.get("DB_USER", "root")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_NAME = os.environ.get("DB_NAME", "soccer_analytics")
