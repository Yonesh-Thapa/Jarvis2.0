#
# File: src/motivational/homeostatic_core.py
#
from __future__ import annotations
import numpy as np

# --- Constants ---
ATTENTION_THRESHOLD = 0.3
STAGNATION_CYCLES_THRESHOLD = 10
BOREDOM_AROUSAL_JOLT = 0.5
SENSITIVITY_AROUSAL_THRESHOLD = 0.75

# FIX: The learning rate modifier was too high, causing oscillating (unstable)
# learning. It has been reduced from 5.0 to 2.0 to allow for smoother,
# more gradual convergence. This should allow stable memories to form.
FAST_LEARNING_RATE_MODIFIER = 2.0

class HomeostaticCore:
    """
    Purpose: To translate cognitive performance into motivational states, including
    an intrinsic drive to escape stagnation (boredom) and the ability to modulate
    the brain's learning rate based on arousal.
    """
    def __init__(self):
        self.valence: float = 0.0
        self.arousal: float = 0.0
        self.attention_tag: float = 0.0
        self._stagnation_counter: int = 0
        self._last_error_level: float = -1.0
        print("Motivational Module Initialized: Homeostatic Core (with Curiosity Drive)")

    def update(self, prediction_error_magnitude: float) -> None:
        """
        Purpose: To update valence and arousal, now with a check for stagnation.
        """
        # --- Boredom and Stagnation Check ---
        if np.isclose(prediction_error_magnitude, self._last_error_level):
            self._stagnation_counter += 1
        else:
            self._stagnation_counter = 0
            self._last_error_level = prediction_error_magnitude
        
        base_arousal = float(np.tanh(prediction_error_magnitude / 10.0))

        # --- Inject Boredom/Frustration Jolt ---
        final_arousal = base_arousal
        if self._stagnation_counter > STAGNATION_CYCLES_THRESHOLD:
            # This check is less relevant during focused training but kept for general use.
            print("STATE: Boredom threshold reached. Injecting frustration/curiosity.")
            final_arousal = min(1.0, base_arousal + BOREDOM_AROUSAL_JOLT)
            self._stagnation_counter = 0

        self.arousal = final_arousal
        self.valence = 1.0 - (2.0 * self.arousal)

        if self.arousal > ATTENTION_THRESHOLD:
            self.attention_tag = self.arousal
        else:
            self.attention_tag = 0.0
            
    def get_status(self) -> dict:
        """Returns the current motivational state."""
        return {"valence": self.valence, "arousal": self.arousal}
        
    def get_learning_modifier(self) -> float:
        """
        Purpose: To determine how plastic the brain should be based on arousal.
        Mechanism: High arousal (from surprise or exploration) leads to a higher
                   learning rate, mimicking neuromodulator effects like adrenaline.
        """
        if self.arousal > SENSITIVITY_AROUSAL_THRESHOLD:
            return FAST_LEARNING_RATE_MODIFIER  # Learn faster
        return 1.0 # Normal learning rate