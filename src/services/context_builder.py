"""
Context Builder — Structured Block Formatting
=============================================

Converts retrieved data into clean, structured sections for the LLM.
Includes safeguards for size and formatting.
"""

from typing import Any, Dict, List, Optional
import datetime

MAX_CONTEXT_CHARS = 3000  # Approx 750-800 tokens

class ContextBuilder:
    def __init__(self):
        self._sections = []

    def add_section(self, title: str, content: str):
        """Add a named section to the context."""
        if not content.strip(): return
        self._sections.append(f"### {title.upper()} ###\n{content}")

    def format_player_stats(self, data: List[Dict[str, Any]]):
        """Format targeted player statistics."""
        output = ""
        for item in data:
            name = item['player_name']
            jersey = item['jersey']
            output += f"\nPLAYER: {name} (#{jersey})\n"
            
            if not item['recent_stats']:
                output += "- No recent match statistics available.\n"
                continue
                
            for s in item['recent_stats']:
                date = s['date_time'].strftime("%Y-%m-%d") if s['date_time'] else "N/A"
                vs = f"{s['home_team']} vs {s['away_team']}"
                rating = f"Rating: {s['rating']}" if s.get('rating') else ""
                output += (
                    f"  Match: {vs} ({date})\n"
                    f"  Mins: {s.get('minutes_played', 0)}, Shots: {s.get('shots', 0)}, "
                    f"Passes: {s.get('passes', 0)}, {rating}\n"
                )
        self.add_section("PLAYER_STATS", output)

    def format_match_info(self, matches: List[Dict[str, Any]]):
        """Format recent match results."""
        output = ""
        for m in matches:
            date = m['date_time'].strftime("%Y-%m-%d") if isinstance(m['date_time'], datetime.datetime) else m['date_time']
            score = f"{m['home_score']}-{m['away_score']}"
            output += f"- {m['home_team_name']} vs {m['away_team_name']} ({date}) | Result: {score}\n"
        self.add_section("MATCH_INFO", output)

    def format_team_info(self, teams: List[Dict[str, Any]]):
        """Format team summaries."""
        output = ""
        for t in teams:
            output += f"- Team: {t['name']} | Colors: {t['colors']}\n"
        self.add_section("TEAM_STATISTICS", output)

    def build(self) -> str:
        """Combine sections and ensure character limit."""
        full_context = "\n\n".join(self._sections)
        
        if len(full_context) > MAX_CONTEXT_CHARS:
            return full_context[:MAX_CONTEXT_CHARS] + "\n... [truncated]"
            
        return full_context
