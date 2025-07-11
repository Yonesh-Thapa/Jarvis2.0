#
# File: src/cognitive/planning_cortex.py
#
# Description: This module simulates the highest level of cognitive processing (L4).
# It integrates the AI's internal state to generate high-level plans. It decides
# to act when motivation is high and valence is low.
#

from __future__ import annotations
from typing import Dict, Any

class PlanningCortex:
    """
    Purpose: To generate high-level plans based on the agent's internal state.
    Mechanism: Implements a simple rule-based system that checks the motivational
    state (valence and arousal) provided by the HomeostaticCore to decide whether
    to explore or continue.
    """
    def __init__(self):
        # This module is a stateless decision-maker.
        # FIX: The arousal threshold is increased to prevent obsessive looping.
        # The AI will now only explore when highly aroused (very surprised or frustrated).
        self.EXPLORATION_AROUSAL_THRESHOLD = 0.9
        print("Cognitive Module Initialized: Planning Cortex (L4)")

    def generate_plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Purpose: To decide on the next action (plan).
        Mechanism: Checks the agent's arousal level. If arousal is high (indicating
                   surprise, frustration, or curiosity generated by the HomeostaticCore),
                   it generates an EXPLORE plan to seek new information. Otherwise,
                   it generates a CONTINUE plan.

        Args:
            state (Dict[str, Any]): The current system state, including 'valence' and 'arousal'.

        Returns:
            A dictionary representing the chosen plan.
        """
        valence = state.get('valence', 0.0)
        arousal = state.get('arousal', 0.0)

        # The decision to explore is now based directly on the motivational state
        # generated by the Homeostatic Core. High arousal (from surprise or boredom)
        # triggers exploration.
        if arousal > self.EXPLORATION_AROUSAL_THRESHOLD:
            plan = {
                "action": "EXPLORE",
                "reason": f"Arousal ({arousal:.2f}) is high; seeking to reduce prediction error."
            }
            return plan

        # If not highly aroused, the default is to continue observing.
        plan = {
            "action": "CONTINUE",
            "reason": f"Arousal ({arousal:.2f}) is stable; maintaining current state."
        }
        
        return plan