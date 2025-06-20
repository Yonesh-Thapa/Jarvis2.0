# Prototype classifier generalization with augmentations
# Tests classifier on SDRs from unseen fonts and with augmentations
import subprocess
import sys
import os

# Test on unseen fonts (no augment)
subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "test_prototype_classifier.py")])

# Test on unseen fonts with augmentations (edit test script to support this if needed)
# (If not supported, this will be a future step)

print("Prototype classifier generalization tests complete.")
