#
# File: src/core/memory_system.py
#
# Description: This module manages offline cognitive processes, simulating the
# function of sleep. It is responsible for memory consolidation (strengthening
# important connections) and synaptic pruning (removing weak or unused connections)
# to maintain the long-term health and efficiency of the NeuralFabric.
#

from __future__ import annotations
from src.core.neural_fabric import NeuralFabric
import numpy as np
import os

# --- Constants ---
# Synaptic weights below this threshold are considered weak and will be pruned during sleep.
PRUNING_THRESHOLD = 0.01

class MemorySystem:
    """
    Purpose: To perform offline optimization of the neural fabric.
    Mechanism: Simulates a sleep cycle. Currently, its primary role is synaptic
    pruning. A future implementation would add memory replay for consolidation,
    using events tagged by the HomeostaticCore's attention system.
    """
    def __init__(self):
        print("Core Module Initialized: Memory System")
        self.pattern_memory = {}

    def _prune_synapses(self, fabric: NeuralFabric) -> int:
        """
        Purpose: To remove weak connections to improve efficiency and reduce noise.
        Mechanism: Iterates through all synapses in every layer of the fabric. If a
        synapse's weight has decayed below a threshold, it is removed.
        """
        pruned_count = 0
        for layer in fabric.layers:
            # Filter the synapse list, keeping only those above the threshold.
            initial_synapse_count = len(layer.bottom_up_synapses)
            layer.bottom_up_synapses = [
                s for s in layer.bottom_up_synapses if s.weight > PRUNING_THRESHOLD
            ]
            pruned_in_layer = initial_synapse_count - len(layer.bottom_up_synapses)
            pruned_count += pruned_in_layer
        return pruned_count

    def _consolidate_memories(self, fabric: NeuralFabric) -> None:
        """
        Purpose: To strengthen important memories. (Placeholder for future implementation).
        Mechanism: In a full implementation, this method would "replay" experiences
        that were tagged with high attention by the HomeostaticCore. This replay,
        running the signals through the fabric again, would strengthen the relevant
        pathways via the existing Hebbian learning rules.
        """
        # This is where memory replay would occur.
        pass

    def perform_sleep_cycle(self, fabric: NeuralFabric) -> None:
        """
        Purpose: To run a full cycle of memory consolidation and pruning.
        """
        print("\n--- ENTERING SLEEP CYCLE ---")
        
        # 1. Consolidate important memories (currently a placeholder)
        self._consolidate_memories(fabric)
        print("  - Memory consolidation phase complete.")

        # 2. Prune weak synapses
        pruned_count = self._prune_synapses(fabric)
        print(f"  - Synaptic pruning complete. Removed {pruned_count} weak connections.")

        # 3. Reset energy budget (simulates replenishment during sleep)
        fabric.total_energy_consumed = 0
        print(f"  - Homeostatic energy budget restored to {fabric._power_budget}.")
        
        print("--- AWAKENING ---")

    # --- Pattern storage helpers ---
    def store_pattern(self, label: str, pattern: np.ndarray, path: str = "memory/pattern_store.npy"):
        self.pattern_memory[label] = pattern
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.pattern_memory, allow_pickle=True)

    def retrieve_pattern(self, label: str, path: str = "memory/pattern_store.npy"):
        if not self.pattern_memory and os.path.exists(path):
            self.pattern_memory = np.load(path, allow_pickle=True).item()
        return self.pattern_memory.get(label)

