#
# File: main.py (Fully Integrated - Final Correction)
#
import time
import random
import numpy as np
import pygame  # Import pygame here
from typing import Dict, List

# --- Main AI Components ---
from src.interfaces.transducers import Webcam
from src.cognitive.temporal_cortex import TemporalCortex
from src.motivational.homeostatic_core import HomeostaticCore

# --- Generative Vision Components ---
from src.vision.feature_extractor import FeatureExtractor
from src.vision.factorial_encoder import FactorialEncoder
from src.motor.letter_decoder import LetterDecoder
from src.motor.stroke_canvas import StrokeCanvas
from src.learning.reconstruction_trainer import ReconstructionTrainer

# --- FIX: Initialize Pygame once, globally, at the very start ---
pygame.init()

def main():
    """Initializes and runs the full AI with its generative vision system."""

    print("=========================================")
    print("=      AI v4.0 (Generative Mind)        =")
    print("=========================================\n")

    # 1. --- INITIALIZE ALL SYSTEMS ---
    print("PHASE I: INITIALIZING SYSTEMS...")

    # --- FIX: Create the font object here and inject it as a dependency ---
    try:
        main_font = pygame.font.Font("FreeSans.ttf", 48)
    except FileNotFoundError:
        print("ERROR: FreeSans.ttf not found. Using default pygame font.")
        main_font = pygame.font.Font(None, 48) # Use a default fallback font

    webcam = Webcam(font=main_font) # Pass the font object in
    feature_extractor = FeatureExtractor()
    encoder = FactorialEncoder(num_latent_neurons=512, feature_dim=7688, k=10)
    decoder = LetterDecoder(latent_dim=512)
    canvas = StrokeCanvas()
    trainer = ReconstructionTrainer(encoder, decoder, canvas, feature_extractor)

    temporal_cortex = TemporalCortex(layer_sizes=[512, 128, 64])
    homeostatic_core = HomeostaticCore()

    print("\nINITIALIZATION COMPLETE. AI IS NOW LIVE.\n")

    # 2. --- LEARNING & COGNITIVE LOOP ---
    characters_to_learn: List[str] = ['A', 'B', 'T']
    cycles_per_letter: int = 150
    reconstruction_error_threshold = 0.2
    total_cycles = 0

    try:
        for letter in characters_to_learn:
            print(f"\n--- NEW TASK: Learn the concept of '{letter}' ---")
            webcam.set_stimulus(letter)

            print("Entering Stage 1: Visual Concept Formation...")
            mastered_concept = None
            for i in range(cycles_per_letter):
                total_cycles += 1
                print(f"  - Visual Training Cycle {i+1}/{cycles_per_letter} for '{letter}'")
                original_image = webcam.capture_frame()
                recon_error, latent_concept = trainer.train_step(original_image)

                if recon_error < reconstruction_error_threshold:
                    print(f"  *** VISUAL CONCEPT MASTERED for '{letter}' (Error: {recon_error:.4f}) ***")
                    mastered_concept = latent_concept
                    encoder.save_weights()
                    break
                time.sleep(0.01)

            if mastered_concept is not None:
                print("\nEntering Stage 2: Higher-Level Cognition...")
                print(f"  TemporalCortex: Receiving stable concept for '{letter}'")
                temporal_cortex.process_input(mastered_concept.tolist())
                sequence_prediction = temporal_cortex.last_prediction[0]
                sequence_error = np.linalg.norm(mastered_concept - sequence_prediction)
                homeostatic_core.update(sequence_error)
                print(f"  Homeostasis: Arousal={homeostatic_core.arousal:.2f}, Valence={homeostatic_core.valence:.2f}")
            else:
                print(f"  --- WARNING: Failed to master concept for '{letter}' in time. Moving on. ---")

    except KeyboardInterrupt:
        print("\n\n--- AI Life Cycle Interrupted by User ---")
        encoder.save_weights()
    finally:
        pygame.quit()
        print("\nAI operation complete.")


if __name__ == "__main__":
    main()