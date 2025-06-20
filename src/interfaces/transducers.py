#
# File: src/interfaces/transducers.py (Corrected)
#

import pygame
import numpy as np
from typing import Tuple

class Webcam:
    def __init__(self, font: pygame.font.Font, size=(64, 64)):
        self.size = size
        self._current_stimulus: str = 'A'
        
        # --- FIX: The font object is received from main.py ---
        self.font = font
        
        if self.font:
             print(f"Interface Initialized: Webcam")
        else:
             print("ERROR: Webcam received an invalid font object.")

    def set_stimulus(self, character: str):
        self._current_stimulus = character

    def capture_frame(self) -> np.ndarray:
        frame_surface = pygame.Surface(self.size)
        frame_surface.fill((0, 0, 0))
        if self.font:
            text_surface = self.font.render(self._current_stimulus, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(self.size[0] // 2, self.size[1] // 2))
            frame_surface.blit(text_surface, text_rect)
        pixels_3d = pygame.surfarray.pixels3d(frame_surface)
        pixels_gray = pixels_3d.mean(axis=2)
        noise = np.random.randn(*self.size) * 5
        frame_with_noise = np.clip(pixels_gray + noise, 0, 255)
        return frame_with_noise.T.astype(np.float32) / 255.0