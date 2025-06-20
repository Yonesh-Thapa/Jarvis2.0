# File: src/cognitive/perception/stroke_detector.py
# Description: Stroke detector layer (L2) for lines/curves at various angles/positions.

import numpy as np

class StrokeDetector:
    def __init__(self, num_strokes=32, input_shape=(62, 62)):
        self.num_strokes = num_strokes
        self.input_shape = input_shape
        # Each neuron is tuned to a primitive: (angle, x, y, length, curvature)
        self.tuning = [self._random_primitive() for _ in range(num_strokes)]
        print(f"StrokeDetector initialized with {num_strokes} primitives.")

    def _random_primitive(self):
        angle = np.random.uniform(0, np.pi)
        x = np.random.randint(0, self.input_shape[0])
        y = np.random.randint(0, self.input_shape[1])
        length = np.random.uniform(10, 30)
        curvature = np.random.uniform(-0.5, 0.5)  # 0 = straight line
        return (angle, x, y, length, curvature)

    def detect(self, edge_map: np.ndarray) -> np.ndarray:
        # For each primitive, compute a response based on how well it matches the edge map
        responses = np.zeros(self.num_strokes)
        for i, (angle, x, y, length, curvature) in enumerate(self.tuning):
            # Simple scoring: sum edge pixels along the primitive's path
            score = self._score_primitive(edge_map, angle, x, y, length, curvature)
            responses[i] = score
        # Normalize and sparsify (kWTA)
        k = max(1, int(0.2 * self.num_strokes))
        topk = np.argsort(responses)[-k:]
        sdr = np.zeros_like(responses)
        sdr[topk] = 1.0
        return sdr

    def _score_primitive(self, edge_map, angle, x, y, length, curvature):
        # For simplicity, sample points along the primitive and sum edge values
        points = []
        for t in np.linspace(0, 1, int(length)):
            dx = t * length * np.cos(angle)
            dy = t * length * np.sin(angle)
            # Add curvature (simple quadratic)
            cx = x + dx
            cy = y + dy + curvature * (t - 0.5) ** 2 * length
            ix, iy = int(round(cx)), int(round(cy))
            if 0 <= ix < edge_map.shape[0] and 0 <= iy < edge_map.shape[1]:
                points.append((ix, iy))
        score = sum(edge_map[ix, iy] for ix, iy in points)
        return score / (len(points) + 1e-6)
