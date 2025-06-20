#
# File: src/vision/factorial_encoder.py (Corrected)
# Description: A more advanced encoder that uses usage tracking and gated
# competition to enforce neural diversity and prevent conceptual collapse.
#

import numpy as np
import os

class FactorialEncoder:
    """
    Encodes a feature vector into a sparse, factorial representation. It uses
    a gating mechanism based on neuron usage to ensure diverse concepts are formed.
    """
    def __init__(self, num_latent_neurons=128, feature_dim=7688, k=10):
        self.num_latent_neurons = num_latent_neurons
        self.feature_dim = feature_dim
        self.k = 10  # Always use exactly 10 winners
        self.weight_path = "memory/factorial_weights.npy"
        self.usage_path = "memory/neuron_usage.npy"

        # --- FIX: Initialize neuron_usage BEFORE it is potentially saved ---
        # Initialize or load neuron usage statistics first.
        if os.path.exists(self.usage_path):
            self.neuron_usage = np.load(self.usage_path)
        else:
            self.neuron_usage = np.zeros(self.num_latent_neurons)

        # Now, initialize or load the dictionary weights.
        if os.path.exists(self.weight_path):
            self.W = np.load(self.weight_path)
        else:
            # If creating weights for the first time, save them.
            self.W = np.random.randn(self.num_latent_neurons, self.feature_dim) * 0.1
            os.makedirs("memory", exist_ok=True)
            self.save_weights()

        print("Vision Module Initialized: Factorial Encoder")

    def encode(self, feature_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Finds the k best-matching neurons using a gated competition that
        penalizes frequently used neurons.
        """
        distances = np.linalg.norm(self.W - feature_vector, axis=1)
        
        # Gated Competition Logic
        usage_penalty = 0.2 
        gated_scores = distances + (self.neuron_usage * usage_penalty)
        
        # Always select exactly k winners, even if there are ties
        sorted_indices = np.argsort(gated_scores)
        active_indices = sorted_indices[:self.k]
        if len(active_indices) < self.k:
            # Pad with additional indices if not enough (shouldn't happen, but for safety)
            pad = np.setdiff1d(np.arange(self.num_latent_neurons), active_indices)[:self.k - len(active_indices)]
            active_indices = np.concatenate([active_indices, pad])
        spikes = np.zeros(self.num_latent_neurons)
        spikes[active_indices] = 1.0
        
        # Update usage statistics
        self.neuron_usage *= 0.995 
        self.neuron_usage[active_indices] += 1.0
        
        return spikes, active_indices

    def reinforce(self, active_indices: np.ndarray, delta_W: np.ndarray):
        """Applies a learning update to the weights of the winning neurons."""
        self.W[active_indices, :] += delta_W

    def save_weights(self):
        """Saves the current weights and usage statistics to memory."""
        np.save(self.weight_path, self.W)
        np.save(self.usage_path, self.neuron_usage)