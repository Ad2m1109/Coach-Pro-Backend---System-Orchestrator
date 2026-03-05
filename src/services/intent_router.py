"""
Intent Router — Lightweight Question Classification
===================================================

Categorizes user questions into broad intents to guide the retrieval process.
This is a rule-based engine and does NOT use an LLM for speed.
"""

import re
from enum import Enum


import re
import logging
import time
from enum import Enum
from typing import List
from rapidfuzz import fuzz

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IntentRouter")

class Intent(str, Enum):
    ANALYTICAL_QUERY = "analytical_query"
    PLAYER_STATS = "player_stats"
    MATCH_STATS = "match_stats"
    TEAM_STATISTICS = "team_statistics"
    TACTICAL_QUESTION = "tactical_question"
    GENERAL_QUESTION = "general_question"

# --- Config ---
FUZZY_THRESHOLD = 85
MAX_TOKENS_TO_CHECK = 20
ALPHA_ONLY_REGEX = re.compile(r"^[a-zA-Z]+$")

# --- Keyword Sets ---
KEYWORDS_ANALYTICAL = {"improve", "trend", "compare", "highest", "average", "increase", "decrease", "over", "last"}
KEYWORDS_PLAYER = {"player", "players", "jersey", "position", "rating", "stat", "stats"}
KEYWORDS_MATCH = {"match", "matches", "game", "games", "result", "score", "opponent", "vs"}
KEYWORDS_TACTICAL = {"formation", "tactics", "tactical", "press", "pressing", "block", "strategy", "style"}
KEYWORDS_TEAM = {"team", "history", "summary", "overview"}

def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, and clean whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())

def contains_similar(target_keywords: set, tokens: List[str]) -> bool:
    """
    Checks if any token is fuzzy-similar to any target keyword.
    Only checks alphabetic tokens to avoid noise from numbers.
    """
    for token in tokens:
        # Strict alphabetic check
        if not ALPHA_ONLY_REGEX.match(token):
            continue
            
        for kw in target_keywords:
            if fuzz.ratio(kw, token) >= FUZZY_THRESHOLD:
                return True
    return False

def detect_intent(question: str) -> Intent:
    """
    Production-grade intent detection using normalization and token-level fuzzy matching.
    Guarantees performance <5ms and handles common typos.
    """
    start_perf = time.perf_counter()
    
    # 1. Normalize
    normalized = normalize_text(question)
    
    # 2. Tokenize and limit
    tokens = normalized.split()[:MAX_TOKENS_TO_CHECK]
    
    # 3. Detect with Priority
    intent = Intent.GENERAL_QUESTION
    
    if contains_similar(KEYWORDS_ANALYTICAL, tokens):
        intent = Intent.ANALYTICAL_QUERY
    elif contains_similar(KEYWORDS_PLAYER, tokens):
        intent = Intent.PLAYER_STATS
    elif contains_similar(KEYWORDS_TACTICAL, tokens):
        intent = Intent.TACTICAL_QUESTION
    elif contains_similar(KEYWORDS_MATCH, tokens):
        intent = Intent.MATCH_STATS
    elif contains_similar(KEYWORDS_TEAM, tokens):
        intent = Intent.TEAM_STATISTICS

    latency_ms = (time.perf_counter() - start_perf) * 1000
    
    logger.info(f"[IntentRouter] Normalized: '{normalized}'")
    logger.info(f"[IntentRouter] Detected: {intent.value} ({latency_ms:.2f}ms)")
    
    return intent
