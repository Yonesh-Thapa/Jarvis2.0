import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Test: Generalization and Prediction on a Completely Unseen Font (Times New Roman)
# This script evaluates the system's ability to generalize and predict letters from a new font (Times New Roman)
# using a nearest-neighbor classifier in SDR space.

import string
import numpy as np
import pygame
from src.interfaces.transducers import Webcam
from src.vision.feature_extractor import FeatureExtractor
from src.cognitive.perception.stroke_detector import StrokeDetector
from src.cognitive.perception.compositional_layer import CompositionalLayer

pygame.init()

# Reference font for building the SDR dictionary (use the training font, e.g., FreeSans)
REFERENCE_FONT = "FreeSans"
TEST_FONT = "Times New Roman"  # Unseen font
IMG_SIZE = 64
LETTERS = string.ascii_uppercase

# Initialize pipeline
feature_extractor = FeatureExtractor()
stroke_detector = StrokeDetector()
compositional_layer = CompositionalLayer()

# Create font objects
ref_font = pygame.font.SysFont(REFERENCE_FONT, IMG_SIZE)
test_font = pygame.font.SysFont(TEST_FONT, IMG_SIZE)

# Build reference SDR dictionary from the training font
ref_webcam = Webcam(ref_font, size=(IMG_SIZE, IMG_SIZE))
ref_sdr_dict = {}
for letter in LETTERS:
    ref_webcam.set_stimulus(letter)
    img = ref_webcam.capture_frame(augment=False)
    edge_map = feature_extractor.extract_edge_maps(img)
    stroke_sdr = stroke_detector.detect(edge_map['vertical'])
    high_level_sdr, _ = compositional_layer.encode(stroke_sdr)
    ref_sdr_dict[letter] = high_level_sdr

# Test on the new font
webcam = Webcam(test_font, size=(IMG_SIZE, IMG_SIZE))
results = {}
correct = 0
for letter in LETTERS:
    webcam.set_stimulus(letter)
    img = webcam.capture_frame(augment=False)
    edge_map = feature_extractor.extract_edge_maps(img)
    stroke_sdr = stroke_detector.detect(edge_map['vertical'])
    high_level_sdr, _ = compositional_layer.encode(stroke_sdr)
    # Nearest neighbor prediction
    best_letter = None
    best_dist = float('inf')
    for ref_letter, ref_sdr in ref_sdr_dict.items():
        dist = np.linalg.norm(high_level_sdr - ref_sdr)
        if dist < best_dist:
            best_dist = dist
            best_letter = ref_letter
    is_correct = (best_letter == letter)
    if is_correct:
        correct += 1
    results[letter] = {
        "predicted": best_letter,
        "correct": is_correct,
        "active_neurons": np.where(high_level_sdr > 0)[0].tolist(),
        "distance": best_dist,
    }

# Print results
print("Generalization and Prediction Results (Times New Roman):\n")
for letter, info in results.items():
    print(f"True: {letter} | Predicted: {info['predicted']} | Correct: {info['correct']} | Distance: {info['distance']:.2f}")
    print(f"  Active neurons: {info['active_neurons']}")
    print()
print(f"Total accuracy: {correct}/{len(LETTERS)} ({100*correct/len(LETTERS):.1f}%)\n")
