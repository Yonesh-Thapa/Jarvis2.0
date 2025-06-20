# Font Holdout SDR Cluster Evaluation
# Retrains the model with one font held out, then visualizes SDRs for the held-out font.
import sys
import os
import string
import numpy as np
import subprocess
import shutil

# --- Settings ---
FONTS = ["FreeSans", "Arial", "Times New Roman", "Comic Sans MS", "Calibri", "Verdana", "Courier New", "Georgia"]
IMG_SIZE = 64
LETTERS = string.ascii_uppercase

# --- Paths ---
AGI_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MEMORY_DIR = os.path.join(AGI_ROOT, "memory")

# --- Font Holdout Loop ---
for heldout_font in FONTS:
    print(f"\n=== Holding out font: {heldout_font} ===")
    # Remove memory to force retraining
    if os.path.exists(MEMORY_DIR):
        shutil.rmtree(MEMORY_DIR)
    os.makedirs(MEMORY_DIR, exist_ok=True)
    # Retrain (main.py should skip heldout_font)
    subprocess.run([sys.executable, os.path.join(AGI_ROOT, "main.py"), "--exclude-font", heldout_font], check=True)
    # Visualize SDRs for held-out font only
    subprocess.run([sys.executable, os.path.join(AGI_ROOT, "tests", "visualize_sdr_clusters.py"), "--only-font", heldout_font, "--save", f"sdr_clusters_holdout_{heldout_font.replace(' ', '_')}.png"])
print("\nAll font holdout evaluations complete.")
