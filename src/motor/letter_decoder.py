#
# File: src/motor/letter_decoder.py
# Description: Decodes a latent concept into drawing commands.
#

import pygame
import numpy as np
import os

class LetterDecoder:
    """Decodes a sparse latent representation into motor commands (strokes)."""
    def __init__(self, latent_dim=128, canvas_size=64):
        self.canvas_size = canvas_size
        self.weight_path = "memory/motor_weights.npy"
        num_stroke_params = 40  # 10 strokes * 4 params each
        
        if os.path.exists(self.weight_path):
            self.M = np.load(self.weight_path)
        else:
            self.M = np.random.randn(latent_dim, num_stroke_params) * 0.01
            os.makedirs("memory", exist_ok=True)
            self.save() # This call was causing the error
            
        print("Motor Module Initialized: Letter Decoder")

    def draw(self, latent_vector_z: np.ndarray, canvas: pygame.Surface) -> pygame.Surface:
        stroke_params = latent_vector_z @ self.M
        strokes = stroke_params.reshape(-1, 4)
        canvas.fill((0, 0, 0))
        for (x0, y0, x1, y1) in strokes:
            start_pos = (self.canvas_size * (x0 + 1) / 2, self.canvas_size * (y0 + 1) / 2)
            end_pos = (self.canvas_size * (x1 + 1) / 2, self.canvas_size * (y1 + 1) / 2)
            pygame.draw.line(canvas, (255, 255, 255), start_pos, end_pos, 2)
        return canvas

    # --- FIX: Added the missing save() method ---
    def save(self):
        """Saves the motor weights to memory."""
        np.save(self.weight_path, self.M)
