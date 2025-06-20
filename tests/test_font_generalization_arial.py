# Test: Generalization to an Unseen Font (Arial)
# This script evaluates the system's ability to generalize to a new font (Arial)
# and visualizes the neuron activations and reconstructed SDRs for all uppercase letters.

import string
import numpy as np
from src.interfaces.transducers import Webcam
from src.vision.feature_extractor import FeatureExtractor
from src.cognitive.perception.stroke_detector import StrokeDetector
from src.cognitive.perception.compositional_layer import CompositionalLayer

# Set up the font to test (Arial, not seen during training)
TEST_FONT = "Arial"
IMG_SIZE = 64
LETTERS = string.ascii_uppercase

# Initialize pipeline
webcam = Webcam(font=TEST_FONT, img_size=IMG_SIZE, augment=False)
feature_extractor = FeatureExtractor()
stroke_detector = StrokeDetector()
compositional_layer = CompositionalLayer(load_weights=True)  # Load learned weights

results = {}

for letter in LETTERS:
    img = webcam.render_letter(letter)
    edge_map = feature_extractor.extract_edge_maps(img)
    stroke_sdr = stroke_detector.detect_strokes(edge_map)
    high_level_sdr = compositional_layer.encode(stroke_sdr)
    recon_stroke_sdr = compositional_layer.decode(high_level_sdr)
    # Optionally, reconstruct image from strokes (if implemented)
    # recon_img = stroke_detector.visualize_strokes(recon_stroke_sdr)
    results[letter] = {
        "active_neurons": np.where(high_level_sdr > 0)[0].tolist(),
        "recon_sdr": recon_stroke_sdr.tolist(),
        # "recon_img": recon_img if recon_img is not None else None
    }

# Print results
for letter, info in results.items():
    print(f"Letter: {letter}")
    print(f"  Active neurons: {info['active_neurons']}")
    print(f"  Reconstructed SDR (first 20): {info['recon_sdr'][:20]}")
    print()

# Optionally, visualize reconstructed images if available
# for letter, info in results.items():
#     if info["recon_img"] is not None:
#         info["recon_img"].show(title=f"Recon {letter}")
