#
# File: src/motor/stroke_canvas.py (Corrected)
#

import pygame
import numpy as np

class StrokeCanvas:
    def __init__(self, size=(64, 64)):
        # --- FIX: Redundant initialization call removed ---
        self.size = size
        self.surface = pygame.Surface(size)
        print("Motor Module Initialized: Stroke Canvas")

    def get_blank_canvas(self) -> pygame.Surface:
        self.surface.fill((0, 0, 0))
        return self.surface

    def get_image_from_canvas(self, canvas: pygame.Surface) -> np.ndarray:
        pixels_3d = pygame.surfarray.pixels3d(canvas)
        pixels_gray = pixels_3d.mean(axis=2)
        return pixels_gray.T.astype(np.float32) / 255.0