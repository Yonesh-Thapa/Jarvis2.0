#
# File: src/vision/feature_extractor.py (New)
# Description: A dedicated module for extracting features from an image.
#

import numpy as np

class FeatureExtractor:
    """Extracts a feature vector from a raw image using edge detection."""
    def __init__(self):
        self._vertical_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self._horizontal_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        print("Vision Module Initialized: Feature Extractor")

    def _convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        k_h, k_w = kernel.shape
        img_h, img_w = image.shape
        out_h = img_h - k_h + 1
        out_w = img_w - k_w + 1
        output = np.zeros((out_h, out_w))
        for y in range(out_h):
            for x in range(out_w):
                output[y, x] = np.sum(image[y:y+k_h, x:x+k_w] * kernel)
        return output

    def extract(self, frame: np.ndarray) -> np.ndarray:
        """Processes a raw frame and returns a flattened feature vector."""
        if frame.max() == 0: # Handle blank frames
            return np.zeros(7688) # Return a zero vector of the correct size
        vertical_edges = self._convolve2d(frame, self._vertical_kernel)
        horizontal_edges = self._convolve2d(frame, self._horizontal_kernel)
        features = np.concatenate((vertical_edges.flatten(), horizontal_edges.flatten()))
        # Normalize to prevent exploding values
        features /= np.max(np.abs(features))
        return features

    def extract_edge_maps(self, frame: np.ndarray) -> dict:
        """Extracts edge maps from the frame for stroke detection."""
        if frame.max() == 0:
            return {'vertical': np.zeros((62, 62)), 'horizontal': np.zeros((62, 62))}
        vertical_edges = self._convolve2d(frame, self._vertical_kernel)
        horizontal_edges = self._convolve2d(frame, self._horizontal_kernel)
        return {'vertical': vertical_edges, 'horizontal': horizontal_edges}
