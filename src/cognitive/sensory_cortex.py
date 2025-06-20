#
# File: src/cognitive/sensory_cortex.py
#
# Description: This module represents the L1 (Layer 1) processing center. Its sole
# responsibility is to convert raw sensory data from transducers into SDRs.
# This version is upgraded to perform basic edge detection for shape recognition.
#

from __future__ import annotations
import numpy as np
from typing import List

# --- SDR Constants ---
SDR_DIMENSION: int = 4096
SDR_ACTIVE_BITS: int = 40

class SensoryCortex:
    """
    Purpose: To perform the initial encoding of sensory data into SDRs.
    Mechanism: It now uses 2D convolution with edge detection kernels to extract
    shape-relevant features from visual input. These features are then used to
    create a stable "neural code" for what the AI sees.
    """
    def __init__(self, sdr_dimension: int = SDR_DIMENSION, active_bits: int = SDR_ACTIVE_BITS):
        self.sdr_dimension = sdr_dimension
        self.active_bits = active_bits
        
        # --- NEW: Edge Detection Kernels (Simplified Sobel Filters) ---
        self._vertical_kernel = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])
        self._horizontal_kernel = np.array([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ])

        # NEW: The projection matrix size is updated for the new feature vector.
        # We will get 64 features from the vertical map and 64 from the horizontal.
        # Total features = 128.
        self._visual_projection = np.random.randn(128, self.sdr_dimension)
        self._audio_projection = np.random.randn(128, self.sdr_dimension)
        
        print("Cognitive Module Initialized: Sensory Cortex (L1) (Upgraded for Edge Detection)")

    def _convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """A simplified 2D convolution function."""
        k_h, k_w = kernel.shape
        img_h, img_w = image.shape
        
        # Create an output image with dimensions adjusted for 'valid' convolution
        out_h = img_h - k_h + 1
        out_w = img_w - k_w + 1
        output = np.zeros((out_h, out_w))

        for y in range(out_h):
            for x in range(out_w):
                # Extract the region of interest
                roi = image[y:y+k_h, x:x+k_w]
                # Apply the kernel
                output[y, x] = np.sum(roi * kernel)
        
        return output

    def _create_sdr(self, projection: np.ndarray) -> List[float]:
        """
        Helper function to generate an SDR from a projection. This mechanism remains the same.
        """
        top_k_indices = np.argsort(projection)[-self.active_bits:]
        sdr = [0.0] * self.sdr_dimension
        for i in top_k_indices:
            sdr[i] = 1.0
        return sdr

    def process_visual_input(self, frame: np.ndarray) -> List[float]:
        """
        Purpose: To convert a raw image frame into a stable visual SDR using shape features.
        Mechanism: This method now performs edge detection to extract features.
        1. It convolves the image with horizontal and vertical kernels.
        2. It downsamples the resulting feature maps.
        3. It combines these features and projects them to create the final SDR.
        """
        # 1. Feature Extraction: Use convolution to find edges.
        vertical_edges = self._convolve2d(frame, self._vertical_kernel)
        horizontal_edges = self._convolve2d(frame, self._horizontal_kernel)

        # 2. Downsampling: Pool the feature maps to reduce dimensionality.
        # We take the max value in 8x8 blocks from the edge maps.
        pooled_features = []
        for edge_map in [vertical_edges, horizontal_edges]:
            h, w = edge_map.shape
            block_size = 8
            for r in range(0, h, block_size):
                for c in range(0, w, block_size):
                    block = edge_map[r:r+block_size, c:c+block_size]
                    if block.size > 0:
                        pooled_features.append(block.max())
                    else:
                        pooled_features.append(0)

        features_vector = np.array(pooled_features)
        
        # 3. Encoding: Project features onto the stable random projection matrix.
        projection = np.dot(features_vector, self._visual_projection)
        
        # 4. Create SDR
        return self._create_sdr(projection)

    def process_audio_input(self, chunk: np.ndarray) -> List[float]:
        """
        Purpose: To convert a raw audio chunk into a stable auditory SDR.
        (This method remains unchanged).
        """
        fft_result = np.fft.rfft(chunk)
        fft_magnitude = np.abs(fft_result)
        num_features = self._audio_projection.shape[0]
        
        if len(fft_magnitude) < num_features:
            padded_magnitude = np.zeros(num_features)
            padded_magnitude[:len(fft_magnitude)] = fft_magnitude
            features_vector = padded_magnitude
        else:
            features_vector = fft_magnitude[:num_features]
            
        if np.sum(features_vector) > 0:
            features_vector /= np.linalg.norm(features_vector)

        projection = np.dot(features_vector, self._audio_projection)
        return self._create_sdr(projection)