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
from services.tactical_memory_service import TacticalMemoryService

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
        "- If an 'ANALYTICAL_RESULT' is present, use those pre-calculated figures as the source of truth.\n"
        "- If 'CONSISTENCY_SCORE' is present, use it to evaluate stability and long-term trends. Do NOT recompute percentages.\n"
        "- If 'TACTICAL_MEMORY' is present, use it ONLY for contextual reference and pattern consistency.\n"
        "- Do NOT invent historical insights or perform new calculations based on the memory.\n"
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
            memory_service = TacticalMemoryService(db)
            
            # Identify metric and potential subjects
            query_type = analytics.resolve_query_type(question)
            metric_map = {
                "top_improvers": "sprint_intensity_trend",
                "highest_average_rating": "average_rating",
                "best_pressing_formation": "average_pressures"
            }
            
            # Attempt memory retrieval if subjects are detected
            memory_injected = False
            if query_type in metric_map:
                metric = metric_map[query_type]
                memory_data = []
                
                # 1. Resolve Subjects (Player or Formation)
                try:
                    team_ids = retrieval.get_team_context_ids()
                    
                    if query_type in ["top_improvers", "highest_average_rating"]:
                        # Look for players mentioned in the question
                        from services.player_service import PlayerService
                        p_service = PlayerService(db)
                        all_players = p_service.get_all_players(team_ids)
                        target_p_id = None
                        for p in all_players:
                            if p.name.lower() in question.lower():
                                target_p_id = p.id
                                break
                        
                        if target_p_id:
                            memory_data = memory_service.get_player_history(user_id, target_p_id, metric)
                            logger.info(f"[TacticalMemory] Retrieved records: {len(memory_data)} for Player ID: {target_p_id}")
                    
                    elif query_type == "best_pressing_formation":
                        # Look for formation names (simple keyword check)
                        # We'll get formations the user has or common ones
                        with db.cursor() as cursor:
                            cursor.execute("SELECT id, name FROM formations WHERE user_id = %s OR user_id IS NULL", (user_id,))
                            formations = cursor.fetchall()
                            target_f_id = None
                            for f in formations:
                                if f['name'].lower() in question.lower():
                                    target_f_id = f['id']
                                    break
                            
                            if target_f_id:
                                memory_data = memory_service.get_formation_history(user_id, target_f_id, metric)
                                logger.info(f"[TacticalMemory] Retrieved records: {len(memory_data)} for Formation ID: {target_f_id}")

                    # 2. Format and inject memory
                    memory_block = memory_service.format_memory_block(memory_data)
                    if memory_block:
                        builder.add_section("TACTICAL_MEMORY", memory_block)
                        memory_injected = True
                        logger.info("[Assistant] Memory injected: True")
                        
                        # 3. Compute Consistency Score (only for trends)
                        if metric == "sprint_intensity_trend":
                            consistency = memory_service.compute_trend_consistency(memory_data)
                            if consistency:
                                score_content = (
                                    f"Positive trend windows: {consistency['positive_windows']} / {consistency['total_windows']}\n"
                                    f"Consistency: {consistency['consistency_percentage']}%"
                                )
                                builder.add_section("CONSISTENCY_SCORE", score_content)
                                logger.info(f"[Assistant] Consistency score injected: {consistency['consistency_percentage']}%")
                    else:
                        logger.info("[Assistant] Memory injected: False")
                except Exception as e:
                    logger.warning(f"[TacticalMemory] Error during ID resolution: {str(e)}")

            # Run actual analytics
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
