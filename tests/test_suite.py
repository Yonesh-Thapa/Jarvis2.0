import os
import unittest
import numpy as np

from src.vision.feature_extractor import FeatureExtractor
from src.vision.factorial_encoder import FactorialEncoder

pygame_available = True
try:
    import pygame
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    pygame.init()
    pygame.display.set_mode((1, 1))
except Exception:
    pygame_available = False

if pygame_available:
    from src.motor.letter_decoder import LetterDecoder
    from src.motor.stroke_canvas import StrokeCanvas
    from src.learning.reconstruction_trainer import ReconstructionTrainer


class FeatureExtractorTest(unittest.TestCase):
    def test_extract_shape(self):
        extractor = FeatureExtractor()
        sample = np.zeros((64, 64), dtype=np.float32)
        features = extractor.extract(sample)
        self.assertEqual(features.shape, (7688,))


class FactorialEncoderTest(unittest.TestCase):
    def test_encode_sparsity(self):
        encoder = FactorialEncoder(num_latent_neurons=20, feature_dim=7688, k=5)
        feature = np.random.rand(7688)
        spikes, indices = encoder.encode(feature)
        self.assertEqual(spikes.shape, (20,))
        self.assertEqual(int(np.sum(spikes)), 5)
        self.assertEqual(len(indices), 5)


@unittest.skipUnless(pygame_available, "pygame required")
class LetterDecoderTest(unittest.TestCase):
    def test_draw_runs(self):
        decoder = LetterDecoder(latent_dim=20, canvas_size=64)
        canvas = StrokeCanvas().get_blank_canvas()
        latent = np.random.randn(20)
        result = decoder.draw(latent, canvas)
        import pygame
        self.assertIsInstance(result, pygame.Surface)


@unittest.skipUnless(pygame_available, "pygame required")
class ReconstructionTrainerTest(unittest.TestCase):
    def test_train_step(self):
        encoder = FactorialEncoder(num_latent_neurons=20, feature_dim=7688, k=5)
        decoder = LetterDecoder(latent_dim=20, canvas_size=64)
        canvas = StrokeCanvas()
        extractor = FeatureExtractor()
        trainer = ReconstructionTrainer(encoder, decoder, canvas, extractor)
        img = np.zeros((64, 64), dtype=np.float32)
        error, latent = trainer.train_step(img)
        self.assertIsInstance(error, float)
        self.assertEqual(latent.shape, (20,))


if __name__ == "__main__":
    unittest.main()
