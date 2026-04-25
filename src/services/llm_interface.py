import logging
from typing import Optional, Dict, Any

# Import the actual LLM client using relative import
from .llm_client import call_llm

logger = logging.getLogger("LLMInterface")


class TacticalAdvisor:
    """Provides tactical advice using the remote LLM."""

    def __init__(self, enable_critique: bool = False):
        self.enable_critique = enable_critique
        logger.info(f"TacticalAdvisor initialized (critique={enable_critique})")

    async def advise_flow(self, summaries: list[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze a sequence of segment summaries and return flow analysis.
        """
        try:
            # Build a prompt from the summaries
            prompt = self._build_flow_prompt(summaries)
            response_text = await call_llm(prompt)

            # Parse the response (expecting JSON-like or structured text)
            return self._parse_flow_response(response_text, summaries)

        except Exception as e:
            logger.error(f"Flow advice error: {e}")
            return {
                "analysis": "Flow analysis failed.",
                "momentum": "Stable",
                "warning": str(e),
                "recommendation": "No change suggested.",
                "confidence_level": 0.0,
            }

    def _build_flow_prompt(self, summaries: list[Dict[str, Any]]) -> str:
        """Build a prompt for the LLM based on segment summaries."""
        lines = ["Analyze the following football match flow based on recent segments:\n"]
        for s in summaries:
            lines.append(
                f"Segment {s.get('segment_index', '?')}: "
                f"Team A possession {s.get('possession_a', 0):.0f}%, "
                f"Team B possession {s.get('possession_b', 0):.0f}%"
            )
        lines.append("\nWhat is the current momentum and any tactical recommendations?")
        return "\n".join(lines)

    def _parse_flow_response(
        self, response_text: str, summaries: list[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Parse the LLM response into structured data."""
        # Simple heuristic parse — replace with JSON parsing if the LLM returns JSON
        return {
            "analysis": response_text[:500],
            "momentum": "Stable" if "stable" in response_text.lower() else "Shifting",
            "warning": "None",
            "recommendation": response_text[:200],
            "confidence_level": 0.5,
        }
