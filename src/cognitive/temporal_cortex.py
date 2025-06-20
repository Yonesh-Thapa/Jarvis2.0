# File: src/cognitive/temporal_cortex.py
#
from __future__ import annotations
import numpy as np
from typing import List, Dict

# Using the V1 architecture for stability
from src.core.neural_fabric import NeuralFabric, Layer 

class TemporalCortex:
    def __init__(self, layer_sizes: List[int]):
        if not layer_sizes:
            raise ValueError("TemporalCortex requires at least one layer.")
        print("Cognitive Module Initializing: Temporal Cortex (L2-L3)...")
        self.fabric = NeuralFabric()
        self.layers: List[Layer] = []
        self.layer_sizes = layer_sizes
        for size in layer_sizes:
            self.layers.append(self.fabric.add_layer(num_neurons=size))
        for i in range(len(self.layers) - 1):
            pre_synaptic_layer = self.layers[i]
            post_synaptic_layer = self.layers[i+1]
            self.fabric.connect_layers(pre_synaptic_layer, post_synaptic_layer)
            print(f"  - Connected L{i+1} ({len(pre_synaptic_layer.neurons)}n) to L{i+2} ({len(post_synaptic_layer.neurons)}n) [Bottom-Up]")
        self.top_down_weights: Dict[int, np.ndarray] = {}
        for i in range(len(self.layers) - 1, 0, -1):
            higher_layer_size = len(self.layers[i].neurons)
            lower_layer_size = len(self.layers[i-1].neurons)
            self.top_down_weights[i] = np.random.rand(higher_layer_size, lower_layer_size) * 0.1
            print(f"  - Created predictive pathway from L{i+1} to L{i} [Top-Down]")
        self.last_prediction: Dict[int, np.ndarray] = {i: np.zeros(size) for i, size in enumerate(layer_sizes)}
        print("Temporal Cortex Initialized.")

    def process_input(self, sdr_input: List[float], learning_modifier: float = 1.0) -> None:
        sdr_input_array = np.array(sdr_input)
        current_prediction = {}
        for i in range(len(self.layers) - 1, 0, -1):
            higher_layer = self.layers[i]
            higher_layer_activity = np.array([n.firing_state for n in higher_layer.neurons], dtype=float)
            predicted_lower_activity = np.dot(higher_layer_activity, self.top_down_weights[i])
            current_prediction[i-1] = predicted_lower_activity
        
        predicted_sdr = current_prediction.get(0, np.zeros(self.layer_sizes[0]))
        self.last_prediction[0] = predicted_sdr
        
        # Use the V1 stable learning mechanism
        self.fabric.process_bottom_up(sdr_input, learning_modifier)

        for i in range(len(self.layers) - 1, 0, -1):
            higher_layer = self.layers[i]
            lower_layer = self.layers[i-1]
            higher_layer_activity = np.array([n.firing_state for n in higher_layer.neurons], dtype=float)
            actual_lower_activity = np.array([n.firing_state for n in lower_layer.neurons], dtype=float)
            error_in_prediction = actual_lower_activity - self.last_prediction[i-1]
            update = np.outer(higher_layer_activity, error_in_prediction)
            self.top_down_weights[i] += (0.01 * learning_modifier) * update
            np.clip(self.top_down_weights[i], 0, 1.0, out=self.top_down_weights[i])

    def get_hierarchy_state(self) -> Dict[str, List[float]]:
        # ... (no changes)
        pass

    # NEW: Generative function to create a mental image from a concept
    def imagine(self, concept_indices: np.ndarray) -> np.ndarray:
        """
        Activates a concept in the top layer and generates a corresponding
        pattern in the input layer through top-down prediction.
        """
        print(f"Attempting to generate image from concept pattern...")
        # Start with the top layer (L3)
        top_layer_activity = np.zeros(self.layer_sizes[-1])
        top_layer_activity[concept_indices] = 1.0

        # Propagate the signal down to L2
        l2_prediction = np.dot(top_layer_activity, self.top_down_weights[2])
        
        # Propagate the L2 signal down to L1
        l1_prediction = np.dot(l2_prediction, self.top_down_weights[1])
        
        # Normalize the final generated image for visualization
        if np.max(l1_prediction) > 0:
            l1_prediction /= np.max(l1_prediction)
            
        return l1_prediction