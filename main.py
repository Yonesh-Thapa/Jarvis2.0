from src import DEBUG, debug_print

debug_print("[DEBUG] main.py module loaded.")
#
# File: main.py (Fully Integrated - Final Correction)
#
import os
import time
import random
import numpy as np
import string
import argparse

# --- Set random seeds for reproducibility ---
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    print(f"[SEED] Global random seed set to {seed}")

# Use a dummy video and audio driver so pygame can run headless
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
from typing import Dict, List

from src.interfaces.transducers import Webcam
from src.interfaces.action_cortex import ActionCortex
from src.cognitive.temporal_cortex import TemporalCortex
from src.motivational.homeostatic_core import HomeostaticCore
from src.vision.feature_extractor import FeatureExtractor
from src.vision.factorial_encoder import FactorialEncoder
from src.motor.letter_decoder import LetterDecoder
from src.motor.stroke_canvas import StrokeCanvas
from src.learning.reconstruction_trainer import ReconstructionTrainer
from src.llm_tutor import LLMTutor
from src.cognitive.perception.stroke_detector import StrokeDetector
from src.cognitive.perception.compositional_layer import CompositionalLayer
from src.cognitive.architecture.learning_utils import process_letter_sample
from src.cognitive.architecture.cognitive_core import CognitiveCore  # Import CognitiveCore
from src.cognitive.perception.l1_edge_detector import L1_EdgeDetector
from src.cognitive.perception.l2_feature_detector import L2_FeatureDetector  # Import the new edge detectors
from src.core.memory_system import MemorySystem
from src.cognitive.brain_visualizer import BrainVisualizer

# Initialize Pygame once, globally, at the very start
pygame.init()
pygame.font.init()

def main():
    debug_print("[DEBUG] main() function entered.")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exclude-font', type=str, default=None, help='Font to exclude from training (for holdout)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    args = parser.parse_args()

    set_global_seed(args.seed)

    """Initializes and runs the full AI with its generative vision system."""

    # --- FIX: Create a dummy display to fully initialize all pygame subsystems ---
    pygame.display.set_mode((1, 1))

    print("=========================================")
    print("=      AI v4.0 (Generative Mind)        =")
    print("=========================================\n")

    # 1. --- INITIALIZE ALL SYSTEMS ---
    print("PHASE I: INITIALIZING SYSTEMS...")

    # Try to use a system font (Arial) if FreeSans.ttf is missing or invalid
    try:
        main_font = pygame.font.Font("FreeSans.ttf", 48)
        # Check if the font object is valid by rendering a test string
        test_surface = main_font.render("A", True, (255, 255, 255))
    except Exception as e:
        print(f"ERROR: Could not load FreeSans.ttf: {e}. Trying system font 'Arial'.")
        try:
            main_font = pygame.font.SysFont("Arial", 48)
            test_surface = main_font.render("A", True, (255, 255, 255))
        except Exception as e2:
            print(f"ERROR: Could not load system font 'Arial': {e2}. Exiting.")
            pygame.quit()
            exit(1)

    if not main_font:
        print("ERROR: Font initialization failed. Exiting.")
        pygame.quit()
        exit(1)

    webcam = Webcam(font=main_font)
    feature_extractor = FeatureExtractor()
    encoder = FactorialEncoder(num_latent_neurons=512, feature_dim=7688, k=10)
    decoder = LetterDecoder(latent_dim=512)
    canvas = StrokeCanvas()
    trainer = ReconstructionTrainer(encoder, decoder, canvas, feature_extractor)

    temporal_cortex = TemporalCortex(layer_sizes=[512, 128, 64])
    homeostatic_core = HomeostaticCore()
    stroke_detector = StrokeDetector(num_strokes=32, input_shape=(62, 62))
    compositional_layer = CompositionalLayer(num_concepts=64, input_dim=32, k=8)

    # Initialize the hierarchical perceptual system
    l1_detector = L1_EdgeDetector(filter_type='sobel', input_size=64, downsample_to=None)
    l2_detector = L2_FeatureDetector(input_feature_size=32768)
    temporal_cortex = TemporalCortex(layer_sizes=[4096, 1024, 512])
    # Add brain visualizer for L1, L2, L3
    brain_visualizer = BrainVisualizer([32768, 4096, 512])

    print("\nINITIALIZATION COMPLETE. AI IS NOW LIVE.\n")

    # 2. --- LEARNING & COGNITIVE LOOP ---
    num_epochs = args.epochs
    reps_per_letter = 200  # Increased for more training per letter
    total_cycles = 0
    characters_to_learn: List[str] = list(string.ascii_uppercase)
    reconstruction_error_threshold = 0.2
    recognized_patterns = {}
    error_log = {letter: [] for letter in characters_to_learn}  # Track errors per letter

    # --- FONT LIST FOR GENERALIZATION ---
    font_names = [
        "FreeSans", "Arial", "Times New Roman", "Comic Sans MS", "Calibri", "Verdana", "Courier New", "Georgia"
    ]
    if args.exclude_font and args.exclude_font in font_names:
        font_names = [f for f in font_names if f != args.exclude_font]
    font_sizes = [48, 56, 64]  # Vary font size for more diversity

    cognitive_core = CognitiveCore()  # Instantiate CognitiveCore
    memory_system = MemorySystem()  # Ensure memory_system is defined
    llm_tutor = LLMTutor()

    # Move Webcam instantiation out of the per-letter loop
    webcam = Webcam(font=main_font)
    action_cortex = ActionCortex()

    running_error = 0  # Initialize in outer scope to avoid UnboundLocalError
    try:
        for epoch in range(1, num_epochs + 1):
            running_error = 0
            print(f"\n==================== EPOCH {epoch} ====================")
            training_sequence = characters_to_learn * reps_per_letter
            random.shuffle(training_sequence)
            print(f"Interleaved training sequence created with {len(training_sequence)} steps.")
            for i, letter in enumerate(training_sequence):
                total_cycles += 1
                webcam.set_stimulus(letter)
                print(llm_tutor.describe_letter(letter))
                original_image = webcam.capture_frame(augment=True)
                edge_maps = feature_extractor.extract_edge_maps(original_image)
                stroke_sdr = stroke_detector.detect((edge_maps['vertical'] + edge_maps['horizontal']) / 2)
                # --- Normalize stroke_sdr ---
                if np.max(stroke_sdr) > 0:
                    stroke_sdr = stroke_sdr / np.max(stroke_sdr)
                # Use unified process_letter_sample for error-driven learning and diagnostics
                result = process_letter_sample(
                    letter,
                    stroke_sdr,
                    compositional_layer,
                    cognitive_core=cognitive_core,
                    use_memory_manager=True,
                    verbose=True
                )
                running_error += result['recon_error']
                error_log[letter].append(result['recon_error'])
                debug_print(
                    f"[DEBUG] Cycle {total_cycles}: letter={letter}, error={result['recon_error']:.3f}, "
                    f"memory_size={len(cognitive_core.memory_manager.memory)}"
                )
                time.sleep(0.01)
            avg_error = running_error / len(training_sequence)
            print(f"[EPOCH {epoch}] Average recon error = {avg_error:.3f}")
            # --- VERIFICATION PHASE ---
            # Use a fixed font for verification (e.g., Arial)
            font_obj = pygame.font.SysFont("Arial", 48)
            webcam = Webcam(font=font_obj, size=(64, 64))
            webcam.set_stimulus(letter)
            for _ in range(5):
                original_image = webcam.capture_frame(augment=False)
                edge_maps = feature_extractor.extract_edge_maps(original_image)
                stroke_sdr = stroke_detector.detect((edge_maps['vertical'] + edge_maps['horizontal']) / 2)
                if np.max(stroke_sdr) > 0:
                    stroke_sdr = stroke_sdr / np.max(stroke_sdr)
                concept_sdr, _ = compositional_layer.encode(stroke_sdr)
            active_neurons = np.where(concept_sdr > 0)[0]
            print(f"A stable pattern of {len(active_neurons)} neurons has formed in L3.")
            print(f"L3 Active Neuron Indices: {active_neurons}")
            recognized_patterns[letter] = active_neurons

        # --- FINAL RECOGNIZED PATTERNS ---
        print("\n--- FINAL RECOGNIZED PATTERNS ---")
        for letter, pattern in recognized_patterns.items():
            print(f"Letter '{letter}': {len(pattern)} active neurons -> {pattern}")
        # --- GENERATIVE VISUALIZATION ---
        print("\n--- GENERATIVE VISUALIZATION (RECONSTRUCTED STROKES) ---")
        for letter, active_neurons in recognized_patterns.items():
            concept_sdr = np.zeros(compositional_layer.num_concepts)
            concept_sdr[active_neurons] = 1.0
            recon_stroke = compositional_layer.decode(concept_sdr)
            print(f"Letter '{letter}': Reconstructed stroke SDR: {np.round(recon_stroke, 2)}")

        # 3. --- FINAL VERIFICATION PHASE ---
        print("\n--- TRAINING COMPLETE. BEGINNING FINAL VERIFICATION. ---")
        cycles_for_stabilization = 10
        for letter in characters_to_learn:
            print(f"Presenting '{letter}' to verify pattern formation...")
            webcam.set_stimulus(letter)
            for _ in range(cycles_for_stabilization):
                frame = webcam.capture_frame()
                l1_features = l1_detector.process(frame)
                l1_active = np.count_nonzero(l1_features)
                print(f"[DIAG] L1 active: {l1_active} / {len(l1_features)}")
                visual_sdr = np.array(l2_detector.process(frame, learning_modifier=0))
                l2_active = np.count_nonzero(visual_sdr)
                print(f"[DIAG] L2 active: {l2_active} / {len(visual_sdr)}")
                # Optionally print a sample of the SDRs
                print(f"[DIAG] L1 sample: {l1_features[:10]}")
                print(f"[DIAG] L2 sample: {visual_sdr[:10]}")
                temporal_cortex.process_input(visual_sdr.tolist(), learning_modifier=0)
                # Visualize L1, L2, L3 firing states
                l3_state = np.array([n.firing_state for n in temporal_cortex.layers[-1].neurons])
                l2_state = np.array([n.firing_state for n in temporal_cortex.layers[-2].neurons])
                # For L1, use the sparse SDR directly (nonzero = firing)
                l1_state = (np.array(l1_features) > 0).astype(float)
                brain_visualizer.update([l1_state, l2_state, l3_state])
            # Capture the resulting pattern from the highest temporal layer
            top_layer_activity = np.array([n.firing_state for n in temporal_cortex.layers[-1].neurons])
            active_neurons = np.where(top_layer_activity)[0]
            recognized_patterns[letter] = active_neurons
            # Store L3 pattern in memory_system
            temporal_cortex.store_l3_pattern(letter, memory_system)

        # --- SIMULATED SLEEP: Replay stored patterns through cortex ---
        print("\n--- SIMULATED SLEEP: Replaying stored L3 patterns to reinforce memory ---")
        for letter in characters_to_learn:
            temporal_cortex.replay_l3_pattern(letter, memory_system, learning_modifier=1.0)
            # Visualize replayed L3 pattern
            l3_state = np.array([n.firing_state for n in temporal_cortex.layers[-1].neurons])
            l2_state = np.array([n.firing_state for n in temporal_cortex.layers[-2].neurons])
            # For L1, reconstruct from L3
            l1_recon = temporal_cortex.reconstruct_l1_from_l3(np.where(l3_state > 0)[0])
            l1_state = (np.array(l1_recon) > 0.1).astype(float)
            brain_visualizer.update([l1_state, l2_state, l3_state])

    except KeyboardInterrupt:
        print("\n\n--- AI Life Cycle Interrupted by User ---")
        encoder.save_weights()
    finally:
        pygame.quit()
        print("\nAI operation complete.")


if __name__ == "__main__":
    main()
