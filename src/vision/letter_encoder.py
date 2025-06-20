# File: src/vision/letter_encoder.py
# Description: Encodes features into a sparse latent "concept" vector.
#

import numpy as np
import os

class LetterEncoder:
    """Encodes a visual feature vector into a sparse latent representation."""
    def __init__(self, num_latent_neurons=128, feature_dim=7688, k=5):
        self.num_latent_neurons = num_latent_neurons
        self.feature_dim = feature_dim
        self.k = k
        self.weight_path = "memory/latent_weights.npy"
        if os.path.exists(self.weight_path):
            self.W = np.load(self.weight_path)
        else:
            self.W = np.random.randn(self.num_latent_neurons, self.feature_dim) * 0.1
            os.makedirs("memory", exist_ok=True)
            self.save()
        print("Vision Module Initialized: Letter Encoder")

    def encode(self, feature_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        distances = np.linalg.norm(self.W - feature_vector, axis=1)
        active_indices = np.argsort(distances)[:self.k]
        spikes = np.zeros(self.num_latent_neurons)
        spikes[active_indices] = 1.0
        return spikes, active_indices

    def reinforce(self, active_indices: np.ndarray, delta_W: np.ndarray):
        self.W[active_indices, :] += delta_W

    def save(self):
        np.save(self.weight_path, self.W)