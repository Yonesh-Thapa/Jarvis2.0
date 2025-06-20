#
# File: main.py
#
import time
import random
import numpy as np
from typing import Dict, List
import cv2
from scipy.ndimage import convolve

# --- Import the V2.0 architecture components ---
from src.interfaces.transducers import Webcam
from src.interfaces.action_cortex import ActionCortex
# Remove circular import
# from src.cognitive.perception.l1_edge_detector import L1_EdgeDetector
from src.cognitive.perception.l2_feature_detector import L2_FeatureDetector
from src.cognitive.temporal_cortex import TemporalCortex
from src.cognitive.planning_cortex import PlanningCortex
from src.core.memory_system import MemorySystem
from src.motivational.homeostatic_core import HomeostaticCore

class L1_EdgeDetector:
    """
    L1 edge detector: supports Gabor (default) or Sobel (fast) filters on 64x64 input.
    Produces 8-orientation sparse activation map as a flat SDR vector.
    Optional downsampling for speed. Structure allows future GPU optimization.
    """
    def __init__(self, orientations=8, kernel_size=7, normalize=True, filter_type='gabor',
                 input_size=64, downsample_to=None):
        self.orientations = orientations
        self.kernel_size = kernel_size
        self.normalize = normalize
        self.filter_type = filter_type  # 'gabor' or 'sobel'
        self.input_size = input_size
        self.downsample_to = downsample_to
        if filter_type == 'gabor':
            self.kernels = self._create_gabor_kernels()
        elif filter_type == 'sobel':
            self.kernels = self._create_sobel_kernels()
        else:
            raise ValueError("filter_type must be 'gabor' or 'sobel'")
        print(f"L1_EdgeDetector initialized: {orientations} orientations, {filter_type} filter, input {input_size}x{input_size}.")

    def _create_gabor_kernels(self):
        kernels = []
        for i in range(self.orientations):
            theta = np.pi * i / self.orientations
            kernel = cv2.getGaborKernel((self.kernel_size, self.kernel_size),
                                        sigma=2.0, theta=theta, lambd=4.0, gamma=0.5, psi=0)
            kernels.append(kernel)
        return kernels

    def _create_sobel_kernels(self):
        # 8 orientations: 0, 22.5, 45, ..., 157.5 deg
        base_kernels = [
            np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),  # 0 deg (horizontal)
            np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]]),  # 22.5 deg
            np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]),  # 45 deg
            np.array([[-1, 2, 1], [-2, 0, 2], [-1, -2, 1]]), # 67.5 deg
            np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),  # 90 deg (vertical)
            np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]),  # 112.5 deg
            np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]]),  # 135 deg
            np.array([[1, -2, -1], [2, 0, -2], [1, 2, -1]])  # 157.5 deg
        ]
        return base_kernels[:self.orientations]

    def process(self, frame: np.ndarray) -> np.ndarray:
        # Resize to input_size x input_size
        img = cv2.resize(frame, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        if img.ndim == 3:
            img = np.mean(img, axis=2)
        img = img.astype(np.float32)
        img = (img - img.min()) / (np.ptp(img) + 1e-8)
        # Optional downsampling for speed
        if self.downsample_to is not None and self.downsample_to < self.input_size:
            img = cv2.resize(img, (self.downsample_to, self.downsample_to), interpolation=cv2.INTER_AREA)
        # Only compute edge maps for nonzero regions
        if np.sum(img) < 1e-3:
            size = self.orientations * img.shape[0] * img.shape[1]
            return np.zeros(size, dtype=np.float32)
        edge_maps = []
        for kernel in self.kernels:
            filtered = convolve(img, kernel, mode='reflect')
            filtered = np.maximum(filtered, 0)
            edge_maps.append(filtered)
        edge_maps = np.stack(edge_maps, axis=0)  # (orientations, H, W)
        if self.normalize:
            norm = np.sqrt(np.sum(edge_maps ** 2, axis=0, keepdims=True) + 1e-6)
            edge_maps = edge_maps / norm
        # Flatten to SDR vector
        features = edge_maps.reshape(-1)
        # Enforce sparsity: keep top 5% activations only
        k = max(1, int(0.05 * features.size))
        if k < features.size:
            thresh = np.partition(features, -k)[-k]
            features = np.where(features >= thresh, features, 0)
        return features

def main():
    """The main function that initializes and runs the AI's training protocol."""
    print("=========================================")
    print("= Biologically Plausible Sentience v0.1 =")
    print("=      --- OPERATION ALPHABET ---       =")
    print("=========================================\n")

    # 1. --- INITIALIZE AI COMPONENTS ---
    print("PHASE I: INITIALIZING SYSTEMS...")
    webcam = Webcam()
    action_cortex = ActionCortex()

    # Initialize the hierarchical perceptual system
    l1_detector = L1_EdgeDetector()
    # A 64x64 frame convolved with a 3x3 kernel results in a 62x62 map.
    # Two maps (H and V) flattened are 3844 + 3844 = 7688 features.
    l2_detector = L2_FeatureDetector(input_feature_size=7688)

    temporal_cortex = TemporalCortex(layer_sizes=[4096, 1024, 512])
    planning_cortex = PlanningCortex()
    homeostatic_core = HomeostaticCore()
    memory_system = MemorySystem()

    print("\nINITIALIZATION COMPLETE. AI IS NOW LIVE.\n")

    # 2. --- INTERLEAVED TRAINING PROTOCOL ---
    letters_to_learn: List[str] = ['A', 'B', 'C']
    reps_per_epoch: int = 75
    num_epochs: int = 1
    total_cycles: int = 0
    recognized_patterns: Dict[str, np.ndarray] = {}

    print("STATE: Engaging focused attention protocol. EXPLORE reflex suppressed for training.")
    planning_cortex.EXPLORATION_AROUSAL_THRESHOLD = 1.1 

    try:
        for epoch in range(1, num_epochs + 1):
            print(f"\n==================== EPOCH {epoch} ====================")
            
            training_sequence = letters_to_learn * reps_per_epoch
            random.shuffle(training_sequence)
            print(f"Interleaved training sequence created with {len(training_sequence)} steps.")

            for i, letter in enumerate(training_sequence):
                total_cycles += 1
                webcam.set_stimulus(letter)
                print(f"--- Cycle {total_cycles} (Epoch {epoch}, Step {i+1}/{len(training_sequence)}, Stimulus: '{letter}') ---")

                # Use the new perceptual pipeline
                frame = webcam.capture_frame()
                l1_features = l1_detector.process(frame)
                learning_mod = homeostatic_core.get_learning_modifier()
                visual_sdr = np.array(l2_detector.process(frame, learning_modifier=learning_mod))
                
                # The rest of the cognitive cycle proceeds as before
                temporal_cortex.process_input(visual_sdr.tolist(), learning_modifier=learning_mod)
                prediction = temporal_cortex.last_prediction[0]
                error_magnitude = np.linalg.norm(visual_sdr - prediction)
                homeostatic_core.update(error_magnitude)
                status = homeostatic_core.get_status()
                print(f"STATE: Valence={status['valence']:.2f}, Arousal={status['arousal']:.2f}, Error={error_magnitude:.2f}")
                
                current_state_dict = {"valence": status['valence'], "arousal": status['arousal'], "error": error_magnitude}
                plan = planning_cortex.generate_plan(current_state_dict)
                action_cortex.execute_plan(plan, {"webcam": webcam})
                
                time.sleep(0.01)

        # 3. --- FINAL VERIFICATION PHASE ---
        print("\n--- TRAINING COMPLETE. BEGINNING FINAL VERIFICATION. ---")
        cycles_for_stabilization = 10
        for letter in letters_to_learn:
            print(f"Presenting '{letter}' to verify pattern formation...")
            webcam.set_stimulus(letter)
            # Run a few cycles to let the network settle on the concept
            for _ in range(cycles_for_stabilization):
                frame = webcam.capture_frame()
                # l1_features = l1_detector.process(frame)
                # No learning during verification
                visual_sdr = np.array(l2_detector.process(frame, learning_modifier=0))
                # We still need to cycle the temporal cortex to get a final state
                temporal_cortex.process_input(visual_sdr.tolist(), learning_modifier=0)
            
            # Capture the resulting pattern from the highest temporal layer
            top_layer_activity = np.array([n.firing_state for n in temporal_cortex.layers[-1].neurons])
            active_neurons = np.where(top_layer_activity)[0]
            recognized_patterns[letter] = active_neurons

    except Exception as e:
        print(f"\n\n--- CATASTROPHIC AI FAILURE ---")
        import traceback
        traceback.print_exc()
    finally:
        print("\n--- FINAL RECOGNIZED PATTERNS ---")
        for letter, pattern in recognized_patterns.items():
            print(f"Letter '{letter}': {len(pattern)} active neurons -> {pattern}")
        
        print("\n--- CONCEPTUAL OVERLAP ANALYSIS ---")
        keys = list(recognized_patterns.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                key1 = keys[i]
                key2 = keys[j]
                if key1 in recognized_patterns and key2 in recognized_patterns:
                    overlap = len(set(recognized_patterns[key1]) & set(recognized_patterns[key2]))
                    print(f"Overlap between '{key1}' and '{key2}': {overlap} neurons.")

        print("\nAI is shutting down.")

if __name__ == "__main__":
    main()