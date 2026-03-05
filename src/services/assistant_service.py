"""
Assistant Service — RAG Architecture with Phi-3 Mini
===================================================

Orchestrates the AI Assistant pipeline:
1. Intent Detection (IntentRouter)
2. Targeted Retrieval (RetrievalService + TacticalKB)
3. Context Building (ContextBuilder)
4. LLM reasoning (Ollama)
"""

import httpx
import time
import logging
from enum import Enum
from typing import Optional, List
from database import Connection

from services.intent_router import detect_intent, Intent
from services.retrieval_service import RetrievalService
from services.tactical_kb import get_tactical_concept
from services.context_builder import ContextBuilder
from services.analytical_service import AnalyticalService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Assistant")


# ------------------------------------------------------------------
# System Mode
# ------------------------------------------------------------------

class SystemMode(str, Enum):
    ASSISTANT = "assistant"
    ANALYSIS = "analysis"

_current_mode: SystemMode = SystemMode.ASSISTANT


def get_system_mode() -> SystemMode:
    return _current_mode


def set_system_mode(mode: SystemMode) -> None:
    global _current_mode
    _current_mode = mode


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3:mini"
OLLAMA_TIMEOUT = 120  # Total timeout for long reasoning sessions


# ------------------------------------------------------------------
# Prompt Builder
# ------------------------------------------------------------------

def build_prompt(question: str, context: str) -> str:
    return (
        "You are an AI assistant for a professional football analytics platform.\n"
        "Use ONLY the following structured context for your answer.\n"
        "\n"
        "### CONTEXT ###\n"
        f"{context}\n"
        "\n"
        "### CRITICAL INSTRUCTIONS ###\n"
        "- Do NOT perform any mathematical calculations or trend estimates. Only use the values provided in the context.\n"
        "- If an 'ANALYTICAL_RESULT' is present, use those pre-calculated figures as the source of truth.\n"
        "- Answer clearly and concisely using **Markdown**.\n"
        "- Use **bolding** for names and key metrics.\n"
        "- If the answer is not in the context, say you don't have that specific data.\n"
        "\n"
        f"User Question: {question}"
    )


# ------------------------------------------------------------------
# Ollama Caller (async)
# ------------------------------------------------------------------

async def call_ollama(prompt: str) -> str:
    """Send a prompt to the local Ollama API."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0},
    }

    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
        response = await client.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()


# ------------------------------------------------------------------
# Public API (RAG Pipeline)
# ------------------------------------------------------------------

async def query_assistant(question: str, db: Connection, user_id: str) -> dict:
    """Refactored RAG pipeline for the AI assistant."""
    start_time = time.time()

    # 1. Mode Check
    if _current_mode == SystemMode.ANALYSIS:
        return {
            "status": "analysis_mode",
            "message": "Assistant unavailable during live match analysis.",
        }

    try:
        # A. Intent Detection
        intent = detect_intent(question)
        
        # B. Targeted Retrieval & Context Building
        retrieval = RetrievalService(db, user_id)
        builder = ContextBuilder()
        records_found = 0

        if intent == Intent.PLAYER_STATS:
            data = retrieval.retrieve_player_stats(question)
            builder.format_player_stats(data)
            records_found = len(data)
        
        elif intent == Intent.MATCH_STATS:
            data = retrieval.retrieve_match_stats()
            builder.format_match_info(data)
            records_found = len(data)
            
        elif intent == Intent.TEAM_STATISTICS:
            data = retrieval.retrieve_team_summary()
            builder.format_team_info(data)
            records_found = len(data)
            
        elif intent == Intent.TACTICAL_QUESTION:
            kb_info = get_tactical_concept(question)
            if kb_info:
                builder.add_section("TACTICAL_KNOWLEDGE", kb_info)
                records_found = 1

        elif intent == Intent.ANALYTICAL_QUERY:
            analytics = AnalyticalService(db, user_id)
            result = analytics.run_analytics(question)
            
            # If analysis returns an error/insufficient data message, return it directly
            if not result or result.startswith("Insufficient data") or result.startswith("No ") or result.startswith("Error"):
                return {
                    "status": "ok",
                    "answer": result or "No sufficient data available for analysis."
                }
            
            builder.add_section("ANALYTICAL_STATS", result)
            records_found = 1

        # C. Retrieve general team metadata for general questions
        if intent == Intent.GENERAL_QUESTION:
             data = retrieval.retrieve_team_summary()
             builder.format_team_info(data)
             records_found = len(data)

        # Build context
        context = builder.build()
        token_estimate = len(context) // 4  # Rough heuristic
        
        # D. FALLBACK: No data found
        if records_found == 0 and not context:
            return {
                "status": "ok",
                "answer": "I don't have specific data to answer that right now. Try asking about your players, matches, or a tactical concept."
            }

        # E. Prompt & LLM
        prompt = build_prompt(question, context)
        answer = await call_ollama(prompt)
        
        latency = round(time.time() - start_time, 2)
        
        # Log performance
        logger.info(f"[Assistant] Intent: {intent.value}")
        logger.info(f"[Assistant] Retrieved records: {records_found}")
        logger.info(f"[Assistant] Context approx tokens: {token_estimate}")
        logger.info(f"[Assistant] LLM latency: {latency}s")

        return {
            "status": "ok",
            "answer": answer,
        }

    except httpx.TimeoutException:
        logger.error("[Assistant] LLM timeout")
        return {"status": "error", "message": "Assistant timed out. Please try again."}
    except Exception as e:
        logger.error(f"[Assistant] Error: {str(e)}")
        return {"status": "error", "message": "Assistant temporarily unavailable."}
