import logging
import json
from typing import List, Dict, Any, Optional
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_BACKEND_SRC = _SCRIPT_DIR.parent
_BACKEND_ROOT = _BACKEND_SRC.parent
_PROJECT_ROOT = _BACKEND_ROOT.parent # This is football-coach/
_PIPELINE_ROOT = _PROJECT_ROOT / "tracking_engine" / "pipeline"

for _p in [str(_PIPELINE_ROOT), str(_PIPELINE_ROOT / "features")]:
    if _p not in sys.path:
        sys.path.append(_p)

# Import TacticalAdvisor from the tracking engine's pipeline
from llm_interface import TacticalAdvisor

logger = logging.getLogger(__name__)

class FlowAnalysisService:
    """
    Hybrid Flow Analysis Service.
    Buffers segment snapshots and triggers intermittent SLM-based flow analysis.
    """
    
    # In-memory buffer for last N segments per analysis run
    # Format: { analysis_id: [segment_data, ...] }
    _buffers: Dict[str, List[dict]] = {}
    _FLOW_WINDOW_SIZE = 3
    
    _advisor = TacticalAdvisor(enable_critique=False)

    @classmethod
    async def process_segment(cls, analysis_id: str, segment_data: dict) -> Optional[dict]:
        """
        Processes a single segment. Buffers it and potentially returns a flow analysis.
        """
        if analysis_id not in cls._buffers:
            cls._buffers[analysis_id] = []
            
        cls._buffers[analysis_id].append(segment_data)
        
        # Trigger flow analysis every N segments
        if len(cls._buffers[analysis_id]) >= cls._FLOW_WINDOW_SIZE:
            logger.info(f"Triggering flow analysis for analysis {analysis_id}")
            history = list(cls._buffers[analysis_id])
            cls._buffers[analysis_id] = [] # Clear buffer after trigger
            
            return await cls._generate_flow_analysis(history)
        
        return None

    @classmethod
    async def _generate_flow_analysis(cls, history: List[dict]) -> dict:
        """
        Calls the SLM to analyze the sequence of segments.
        """
        try:
            # Extract relevant metrics for the LLM to process
            summaries = []
            for h in history:
                analysis = h.get("analysis", {})
                summaries.append({
                    "segment_index": h.get("segment_index"),
                    "possession_a": analysis.get("possession_team_a_pct"),
                    "possession_b": analysis.get("possession_team_b_pct"),
                    "team_a_tags": h.get("team_a_tags", []),
                    "team_b_tags": h.get("team_b_tags", []),
                    "metrics_a": analysis.get("team_a", {}),
                    "metrics_b": analysis.get("team_b", {}),
                })
            
            # Use the TacticalAdvisor's new advise_flow method
            # (Note: advise_flow is sync, so we run it in a thread if needed, 
            # but call_remote_llm_sync handles the internal loop)
            flow_result = cls._advisor.advise_flow(summaries)
            
            return {
                "type": "flow_analysis",
                "analysis": flow_result.get("analysis", "Flow analysis failed."),
                "momentum": flow_result.get("momentum", "Stable"),
                "warning": flow_result.get("warning", "None"),
                "recommendation": flow_result.get("recommendation", "No change suggested."),
                "confidence": flow_result.get("confidence_level", 0.0),
                "segment_range": [history[0]["segment_index"], history[-1]["segment_index"]]
            }
        except Exception as e:
            logger.error(f"Flow analysis generation error: {e}")
            return {
                "type": "flow_analysis",
                "error": str(e)
            }

    @classmethod
    def clear_buffer(cls, analysis_id: str):
        """Cleans up memory when job is done or cancelled."""
        cls._buffers.pop(analysis_id, None)
