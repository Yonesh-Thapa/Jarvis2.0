# Prototype-based classifier for robust letter generalization
# Builds SDR prototypes for each letter from multiple fonts/augmentations, then tests on unseen fonts/styles.

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import string
import numpy as np
import pygame
from src.interfaces.transducers import Webcam
from src.vision.feature_extractor import FeatureExtractor
from src.cognitive.perception.stroke_detector import StrokeDetector
from src.cognitive.perception.compositional_layer import CompositionalLayer

pygame.init()

LETTERS = string.ascii_uppercase
IMG_SIZE = 64
N_PROTOTYPES = 10  # Number of prototypes per letter
TRAIN_FONTS = ["FreeSans", "Arial", "Calibri", "Verdana", "Courier New"]
TEST_FONTS = ["Times New Roman", "Comic Sans MS", "Georgia"]

feature_extractor = FeatureExtractor()
stroke_detector = StrokeDetector()
compositional_layer = CompositionalLayer()

# --- Build prototype SDRs for each letter ---
prototypes = {l: [] for l in LETTERS}
for letter in LETTERS:
    for font_name in TRAIN_FONTS:
        for _ in range(N_PROTOTYPES):
            font_obj = pygame.font.SysFont(font_name, IMG_SIZE)
            webcam = Webcam(font=font_obj, size=(IMG_SIZE, IMG_SIZE))
            webcam.set_stimulus(letter)
            img = webcam.capture_frame(augment=True)
            edge_map = feature_extractor.extract_edge_maps(img)
            stroke_sdr = stroke_detector.detect((edge_map['vertical'] + edge_map['horizontal']) / 2)
            sdr, _ = compositional_layer.encode(stroke_sdr)
            prototypes[letter].append(sdr)

# --- Test on unseen fonts ---
results = []
for test_font in TEST_FONTS:
    for letter in LETTERS:
        font_obj = pygame.font.SysFont(test_font, IMG_SIZE)
        webcam = Webcam(font=font_obj, size=(IMG_SIZE, IMG_SIZE))
        webcam.set_stimulus(letter)
        img = webcam.capture_frame(augment=False)
        edge_map = feature_extractor.extract_edge_maps(img)
        stroke_sdr = stroke_detector.detect((edge_map['vertical'] + edge_map['horizontal']) / 2)
        sdr, _ = compositional_layer.encode(stroke_sdr)
        # Nearest prototype classification
        best_letter = None
        best_dist = float('inf')
        for l, proto_list in prototypes.items():
            for proto in proto_list:
                dist = np.linalg.norm(sdr - proto)
                if dist < best_dist:
                    best_dist = dist
                    best_letter = l
        correct = (best_letter == letter)
        results.append((test_font, letter, best_letter, correct, best_dist))

# --- Report results ---
correct_total = sum(1 for r in results if r[3])
print(f"\nPrototype Classifier Generalization Results:")
for test_font in TEST_FONTS:
    font_results = [r for r in results if r[0] == test_font]
    acc = sum(1 for r in font_results if r[3]) / len(font_results)
    print(f"Font: {test_font} | Accuracy: {acc*100:.1f}%")
    for r in font_results:
        print(f"  True: {r[1]} | Predicted: {r[2]} | Correct: {r[3]} | Distance: {r[4]:.2f}")
print(f"\nTotal accuracy: {correct_total}/{len(results)} ({100*correct_total/len(results):.1f}%)\n")
