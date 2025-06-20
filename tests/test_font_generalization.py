# File: tests/test_font_generalization.py
# Test the model's ability to recognize and reconstruct letters from an unseen font

import pygame
import numpy as np
import string
import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import itertools
# Add the project root (one level up from this file) to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.interfaces.transducers import Webcam
from src.vision.feature_extractor import FeatureExtractor
from src.cognitive.perception.stroke_detector import StrokeDetector
from src.cognitive.perception.compositional_layer import CompositionalLayer
from src.cognitive.perception.l1_edge_detector import L1_EdgeDetector
from src.cognitive.perception.l2_feature_detector import L2_FeatureDetector
from src.cognitive.architecture.cognitive_core import CognitiveCore

pygame.init()
pygame.font.init()

LETTERS = list('ABCDEF')  # Expand to A–F for next step

# Parameters for tuning
ORIENTATIONS = 4  # Lowered for efficiency
KERNEL_SIZE = 5
RF_SIZE = 3
INPUT_SIZE = 8  # Lowered for efficiency
EPOCHS = 10  # More epochs for learning
RECON_ERROR_THRESHOLD = 2.5  # Relaxed threshold
PARAM_SWEEP = [
    {'ORIENTATIONS': 4, 'KERNEL_SIZE': 5, 'RF_SIZE': 3, 'K_WINNERS': 0.02},
    {'ORIENTATIONS': 6, 'KERNEL_SIZE': 5, 'RF_SIZE': 3, 'K_WINNERS': 0.03},
    {'ORIENTATIONS': 8, 'KERNEL_SIZE': 7, 'RF_SIZE': 5, 'K_WINNERS': 0.05},
]

SEED = 42  # Default seed for reproducibility

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    print(f"[SEED] Global random seed set to {seed}")

def test_unseen_font():
    # Use a system font that is likely different from Arial (e.g., 'Comic Sans MS')
    test_font = pygame.font.SysFont('Comic Sans MS', 48)
    webcam = Webcam(font=test_font)
    feature_extractor = FeatureExtractor()
    stroke_detector = StrokeDetector(num_strokes=32, input_shape=(62, 62))
    # Remove fixed input_dim, will create compositional_layer dynamically below
    results = {}
    for letter in string.ascii_uppercase:
        webcam.set_stimulus(letter)
        frame = webcam.capture_frame(augment=False)
        edge_maps = feature_extractor.extract_edge_maps(frame)
        stroke_sdr = stroke_detector.detect((edge_maps['vertical'] + edge_maps['horizontal']) / 2)
        compositional_layer = CompositionalLayer(num_concepts=64, input_dim=len(stroke_sdr), k=8)
        concept_sdr, _ = compositional_layer.encode(stroke_sdr)
        active_neurons = np.where(concept_sdr > 0)[0]
        recon_stroke = compositional_layer.decode(concept_sdr)
        results[letter] = {
            'active_neurons': active_neurons,
            'recon_stroke': np.round(recon_stroke, 2)
        }
    return results

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def plot_confusion_matrix(cm, classes):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def recognize_letters_hierarchical():
    MAX_EPOCHS = 100  # Prevent infinite loops
    RECON_ERROR_THRESHOLD = 0.2  # Tune as needed
    for sweep_params in PARAM_SWEEP:
        print(f"\n=== PARAMS: {sweep_params} ===")
        accuracy_per_epoch = []
        best_cm = None
        best_sdrs = None
        best_labels = None
        best_recon_errors = None
        epoch = 0
        # --- Persistent objects for this sweep ---
        l1_detector = L1_EdgeDetector(orientations=sweep_params['ORIENTATIONS'], kernel_size=sweep_params['KERNEL_SIZE'])
        test_img = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        l1_features_shape = l1_detector.process(test_img).shape
        l2_detector = L2_FeatureDetector(input_feature_size=l1_features_shape[0], input_shape=(INPUT_SIZE,INPUT_SIZE), rf_size=sweep_params['RF_SIZE'])
        # Get actual SDR shape from a real sample
        ref_font = pygame.font.SysFont('Arial', INPUT_SIZE)
        webcam_ref = Webcam(font=ref_font, size=(INPUT_SIZE, INPUT_SIZE))
        webcam_ref.set_stimulus(LETTERS[0])
        frame = webcam_ref.capture_frame(augment=False)
        l1_features = l1_detector.process(frame)
        sdr_sample = np.array(l2_detector.process(l1_features, learning_modifier=0))
        print(f"[DEBUG] Using SDR shape {sdr_sample.shape} for CompositionalLayer input_dim")
        compositional_layer = CompositionalLayer(num_concepts=64, input_dim=len(sdr_sample), k=8)
        cognitive_core = CognitiveCore()
        ref_sdrs = {}
        for letter in LETTERS:
            webcam_ref.set_stimulus(letter)
            frame = webcam_ref.capture_frame(augment=False)
            l1_features = l1_detector.process(frame)
            sdr = np.array(l2_detector.process(l1_features, learning_modifier=0))
            ref_sdrs[letter] = sdr
        test_font = pygame.font.SysFont('Comic Sans MS', INPUT_SIZE)
        webcam_test = Webcam(font=test_font, size=(INPUT_SIZE, INPUT_SIZE))
        while True:
            epoch += 1
            print(f"\n--- Epoch {epoch} ---")
            correct = 0
            total = 0
            y_true, y_pred, sdr_list, label_list, recon_errors = [], [], [], [], []
            for letter in LETTERS:
                webcam_test.set_stimulus(letter)
                frame = webcam_test.capture_frame(augment=False)
                l1_features = l1_detector.process(frame)
                if np.all(l1_features == 0):
                    print(f"True: {letter} | Predicted: None | Correct: False | Reason: Blank input, no computation performed.")
                    continue
                sdr = np.array(l2_detector.process(l1_features, learning_modifier=0))
                dists = np.array([np.linalg.norm(sdr - ref_sdrs[l]) for l in LETTERS])
                probs = softmax(-dists)
                pred_idx = np.argmax(probs)
                pred_letter = LETTERS[pred_idx]
                is_correct = (pred_letter == letter)
                # --- Debug: Print SDR shape ---
                print(f"[DEBUG] SDR shape for letter '{letter}':", sdr.shape)
                # --- Debug: Print W shape ---
                print(f"[DEBUG] CompositionalLayer W shape:", compositional_layer.W.shape)
                concept_sdr, _ = compositional_layer.encode(sdr)
                recon_sdr = compositional_layer.decode(concept_sdr)
                recon_error = np.linalg.norm(sdr - recon_sdr) / (np.linalg.norm(sdr) + 1e-8)
                recon_errors.append(recon_error)
                print(f"True: {letter} | Predicted: {pred_letter} | Correct: {is_correct} | Recon error: {recon_error:.3f} | Top-3: {[LETTERS[i] for i in probs.argsort()[-3:][::-1]]} | Probabilities: {[round(float(p),3) for p in probs[probs.argsort()[-3:][::-1]]]}")
                # --- Learning logic ---
                if not is_correct or recon_error > RECON_ERROR_THRESHOLD:
                    compressed = cognitive_core.compression.compress(sdr)
                    schema = cognitive_core.schema.generate(letter)
                    cognitive_core.memory_manager.mark_incorrect(compressed, schema)
                    print(f"[LEARNING] Pattern for '{letter}' marked for relearning (reason: {'misclassified' if not is_correct else 'high recon error'}).")
                else:
                    print(f"[DEBUG] Calling store() for letter '{letter}' with verified=True")
                    compressed = cognitive_core.compression.compress(sdr)
                    schema = cognitive_core.schema.generate(letter)
                    cognitive_core.memory_manager.store(sdr, letter, error=0, reward=1, verified=True)
                if is_correct:
                    correct += 1
                total += 1
                y_true.append(LETTERS.index(letter))
                y_pred.append(pred_idx)
                sdr_list.append(sdr)
                label_list.append(letter)
            acc = correct/total if total > 0 else 0
            accuracy_per_epoch.append(acc)
            print(f"Epoch {epoch} accuracy: {correct}/{total} ({100*acc:.1f}%)\n")
            # Print SDR stats
            sdrs = np.array(sdr_list)
            if len(sdrs) > 0:
                print(f"SDR mean: {np.mean(sdrs):.3f}, std: {np.std(sdrs):.3f}")
                print(f"Mean recon error: {np.mean(recon_errors):.3f}, min: {np.min(recon_errors):.3f}, max: {np.max(recon_errors):.3f}")
            # Print CompositionalLayer weight stats
            print(f"[DIAG] CompositionalLayer W mean: {np.mean(compositional_layer.W):.4f}, std: {np.std(compositional_layer.W):.4f}")
            # Confusion matrix
            cm = np.zeros((len(LETTERS), len(LETTERS)), dtype=int)
            for t, p in zip(y_true, y_pred):
                cm[t, p] += 1
            # Save best results for later visualization
            if acc >= 1.0 or (best_cm is None or acc > np.trace(best_cm)/np.sum(best_cm)):
                best_cm = cm.copy()
                best_sdrs = sdrs.copy()
                best_labels = label_list.copy()
                best_recon_errors = recon_errors.copy()
            # Stop if 100% accuracy or max epochs reached
            if acc >= 1.0:
                print(f"\n100% accuracy reached after {epoch} epochs!")
                break
            if epoch >= MAX_EPOCHS:
                print(f"\nMax epochs ({MAX_EPOCHS}) reached. Best accuracy: {100*max(accuracy_per_epoch):.1f}%")
                break
        print(f"\nPARAMS {sweep_params} - Accuracy per epoch: {[round(100*a,1) for a in accuracy_per_epoch]}")
        # Show confusion matrix and PCA only after training
        if best_cm is not None:
            plot_confusion_matrix(best_cm, LETTERS)
        if best_sdrs is not None and len(best_sdrs) > 2:
            pca = PCA(n_components=2)
            proj = pca.fit_transform(best_sdrs)
            plt.figure()
            for i, letter in enumerate(LETTERS):
                idxs = [j for j, l in enumerate(best_labels) if l == letter]
                plt.scatter(proj[idxs, 0], proj[idxs, 1], label=letter)
            plt.legend()
            plt.title(f'SDR PCA projection (test font, final/best)')
            plt.show()
        # Plot reconstruction error histogram
        if best_recon_errors is not None:
            plt.figure()
            plt.hist(best_recon_errors, bins=10)
            plt.title('Reconstruction Error Histogram (final/best)')
            plt.xlabel('Reconstruction Error')
            plt.ylabel('Count')
            plt.show()

if __name__ == "__main__":
    set_global_seed(SEED)
    recognize_letters_hierarchical()
