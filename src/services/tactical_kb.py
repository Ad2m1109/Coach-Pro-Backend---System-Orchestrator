"""
Tactical Knowledge Base — Domain Expertise
==========================================

Provides static definitions for tactical concepts using caching.
"""

from functools import lru_cache
from typing import Optional

# Pre-defined tactical library
TACTICAL_KNOWLEDGE = {
    "4-3-3": (
        "A balanced formation with 4 defenders, 3 midfielders, and 3 forwards. "
        "Allows for strong wing play and controlling the midfield triangle."
    ),
    "high press": (
        "Defensive strategy where players pressure opponents close to their goal. "
        "Aims to win back possession quickly and capitalize on errors."
    ),
    "low block": (
        "A compact defensive setup where the team sits deep to deny space behind the defense. "
        "Focuses on discipline and counter-attacking."
    ),
    "gegenpressing": (
        "Counter-pressing immediately after losing the ball to prevent the opponent "
        "from launching a counter-attack."
    ),
    "mid block": (
        "Defensive structure positioned in the middle third of the pitch. "
        "Balances space denial with pressing triggers."
    ),
    "overlapping": (
        "Full-backs moving past wide midfielders to create superior numbers in attack."
    )
}

@lru_cache(maxsize=128)
def get_tactical_concept(query: str) -> Optional[str]:
    """Retrieves a definition if keywords match the knowledge base."""
    q_lower = query.lower()
    
    found_definitions = []
    for key, definition in TACTICAL_KNOWLEDGE.items():
        if key in q_lower:
            found_definitions.append(f"{key.upper()}: {definition}")
            
    if not found_definitions:
        return None
        
    return "\n".join(found_definitions)
