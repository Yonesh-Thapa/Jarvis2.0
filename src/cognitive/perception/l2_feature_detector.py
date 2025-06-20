#
# File: src/cognitive/perception/l2_feature_detector.py
#
# Description: This module represents the second layer (L2) of the perceptual
# system. It receives edge data from L1 and learns to recognize recurring
# combinations of those edges (corners, curves, etc.)
#

from __future__ import annotations
from typing import List
import numpy as np
from src.core.neural_fabric import NeuralFabric

# --- SDR Constants ---
SDR_DIMENSION: int = 4096  # The final output size of the perceptual system
SDR_ACTIVE_BITS: int = 40

class L2_FeatureDetector:
    """
    Purpose: To learn and recognize combinations of simple features from L1.
    Mechanism: Contains its own internal NeuralFabric to learn feature-conjunctions.
    It takes raw feature vectors from L1 and outputs a stable SDR.
    Uses local receptive fields and sparse activation for efficiency.
    """
    def __init__(self, input_feature_size: int, input_shape=(16,16), rf_size=5):
        # This layer has its own brain to learn concepts from the layer below.
        self.fabric = NeuralFabric()
        
        # The L2 fabric has two layers: an input layer matching L1's output size,
        # and a conceptual layer that will form the final SDR.
        self.input_layer = self.fabric.add_layer(num_neurons=input_feature_size, input_shape=input_shape, rf_size=rf_size)
        self.conceptual_layer = self.fabric.add_layer(num_neurons=SDR_DIMENSION)
        
        self.fabric.connect_layers(self.input_layer, self.conceptual_layer)
        
        print("Perceptual Module Initialized: L2 Feature Detector (Sparse, Local)")

    def process(self, l1_feature_vector: np.ndarray, learning_modifier: float = 1.0) -> List[float]:
        """
        Processes the feature vector from L1 and outputs a high-level conceptual SDR.
        
        Args:
            l1_feature_vector (np.ndarray): The combined edge features from the L1 detector.
            learning_modifier (float): The learning rate modifier from the homeostatic core.

        Returns:
            A Sparse Distributed Representation (SDR) of the perceived L2 features.
        """
        # The L1 feature vector is used as the input for this fabric.
        self.fabric.process_bottom_up(l1_feature_vector.tolist(), learning_modifier)
        
        # The final SDR is the firing state of this layer's conceptual neurons.
        sdr = np.array([float(n.firing_state) for n in self.conceptual_layer.neurons])
        # Competitive inhibition: keep only top SDR_ACTIVE_BITS activations
        if np.count_nonzero(sdr) > SDR_ACTIVE_BITS:
            topk_indices = np.argpartition(sdr, -SDR_ACTIVE_BITS)[-SDR_ACTIVE_BITS:]
            mask = np.zeros_like(sdr)
            mask[topk_indices] = 1
            sdr = sdr * mask
        return sdr.tolist()
