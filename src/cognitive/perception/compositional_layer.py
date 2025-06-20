# File: src/cognitive/perception/compositional_layer.py
# Description: Compositional layer (L3) for learning combinations of strokes (SDR, kWTA, Hebbian)

import numpy as np
import os

class CompositionalLayer:
    def __init__(self, num_concepts=64, input_dim=32, k=8, usage_penalty=0.2):
        self.num_concepts = num_concepts
        self.input_dim = input_dim
        self.k = k
        self.usage_penalty = usage_penalty
        self.weight_path = "memory/compositional_weights.npy"
        self.usage_path = "memory/compositional_usage.npy"
        if os.path.exists(self.weight_path):
            W_loaded = np.load(self.weight_path)
            if W_loaded.shape == (num_concepts, input_dim):
                self.W = W_loaded
            else:
                print(f"[WARNING] Weight shape mismatch: found {W_loaded.shape}, expected ({num_concepts}, {input_dim}). Reinitializing weights.")
                self.W = np.random.randn(num_concepts, input_dim) * 0.1
        else:
            self.W = np.random.randn(num_concepts, input_dim) * 0.1
        if os.path.exists(self.usage_path):
            self.usage = np.load(self.usage_path)
        else:
            self.usage = np.zeros(num_concepts)
        print(f"CompositionalLayer initialized with {num_concepts} concept neurons.")

    def encode(self, stroke_sdr: np.ndarray) -> np.ndarray:
        # Compute similarity to each concept neuron
        sims = self.W @ stroke_sdr
        # Gated competition: penalize frequently used neurons
        gated_scores = sims - (self.usage * self.usage_penalty)
        # kWTA: pick top-k
        topk = np.argsort(gated_scores)[-self.k:]
        sdr = np.zeros(self.num_concepts)
        sdr[topk] = 1.0
        # Update usage for the winning neurons
        self.usage *= 0.995  # decay
        self.usage[topk] += 1.0
        np.save(self.usage_path, self.usage)
        return sdr, topk

    def reinforce(self, topk, stroke_sdr, lr=0.01):
        # Hebbian update for active concept neurons
        for idx in topk:
            self.W[idx] += lr * (stroke_sdr - self.W[idx])
        np.save(self.weight_path, self.W)

    def decode(self, concept_sdr: np.ndarray) -> np.ndarray:
        # Reconstruct stroke SDR from active concept neurons
        return concept_sdr @ self.W
