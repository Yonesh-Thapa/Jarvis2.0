#
# File: src/learning/reconstruction_trainer.py (Minor update)
# Description: We only need to update the type hint to use the new encoder.
# The core logic remains the same.
#

import numpy as np
# Import the new FactorialEncoder
from src.vision.factorial_encoder import FactorialEncoder
from src.motor.stroke_canvas import StrokeCanvas

class ReconstructionTrainer:
    """Manages the see -> draw -> compare -> reinforce cycle."""
    def __init__(self, encoder: FactorialEncoder, decoder, canvas_handler, feature_extractor):
        self.encoder = encoder
        self.decoder = decoder
        self.canvas = canvas_handler
        self.feature_extractor = feature_extractor
        self.learning_rate = 1e-3
        print("Learning Module Initialized: Reconstruction Trainer (Using Factorial Encoder)")

    def train_step(self, original_image: np.ndarray) -> tuple[float, np.ndarray]:
        # This function's logic does not need to change. It just receives the
        # new, more diverse latent vectors from the factorial encoder.
        features_in = self.feature_extractor.extract(original_image)
        latent_vector_z, active_indices = self.encoder.encode(features_in)
        blank_canvas = self.canvas.get_blank_canvas()
        generated_canvas = self.decoder.draw(latent_vector_z, blank_canvas)
        generated_image = self.canvas.get_image_from_canvas(generated_canvas)
        # Normalize generated_image to 0-1 float before feature extraction
        generated_image = (generated_image - np.min(generated_image)) / (np.ptp(generated_image) + 1e-8)
        features_gen = self.feature_extractor.extract(generated_image)
        error = np.linalg.norm(features_in - features_gen) / (np.linalg.norm(features_in) + 1e-9)
        delta_W = self.learning_rate * error * (features_in - self.encoder.W[active_indices])
        # Clip delta_W to ±0.05 to avoid runaway weight explosions
        delta_W = np.clip(delta_W, -0.05, 0.05)
        self.encoder.reinforce(active_indices, delta_W)
        return error, latent_vector_z
