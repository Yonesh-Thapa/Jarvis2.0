#
# File: src/interfaces/transducers.py (Corrected)
#

import pygame
import numpy as np
from typing import Tuple
import random

class Webcam:
    def __init__(self, font: 'pygame.font.Font', size=(64, 64)):
        if isinstance(font, str):
            raise TypeError("Webcam received a string for 'font', expected a pygame.font.Font object. Please initialize the font with pygame.font.Font or pygame.font.SysFont and pass the object.")
        self.size = size
        self._current_stimulus: str = 'A'
        self.font = font
        
        if self.font:
             print(f"Interface Initialized: Webcam")
        else:
             print("ERROR: Webcam received an invalid font object.")

    def set_stimulus(self, character: str):
        self._current_stimulus = character

    def capture_frame(self, augment: bool = True) -> np.ndarray:
        frame_surface = pygame.Surface(self.size, pygame.SRCALPHA)
        frame_surface.fill((0, 0, 0, 0))
        if self.font:
            text_surface = self.font.render(self._current_stimulus, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(self.size[0] // 2, self.size[1] // 2))
            # Data augmentation
            if augment:
                # Random rotation
                angle = random.uniform(-30, 30)
                text_surface = pygame.transform.rotate(text_surface, angle)
                text_rect = text_surface.get_rect(center=(self.size[0] // 2, self.size[1] // 2))
                # Random scale
                scale = random.uniform(0.8, 1.2)
                new_size = (int(text_surface.get_width() * scale), int(text_surface.get_height() * scale))
                text_surface = pygame.transform.smoothscale(text_surface, new_size)
                text_rect = text_surface.get_rect(center=(self.size[0] // 2, self.size[1] // 2))
                # Random translation
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)
                text_rect = text_rect.move(dx, dy)
            frame_surface.blit(text_surface, text_rect)
        pixels_3d = pygame.surfarray.pixels3d(frame_surface)
        pixels_gray = pixels_3d.mean(axis=2)
        noise = np.random.randn(*self.size) * 5
        frame_with_noise = np.clip(pixels_gray + noise, 0, 255)
        return frame_with_noise.T.astype(np.float32) / 255.0