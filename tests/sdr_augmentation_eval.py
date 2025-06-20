# SDR Augmentation Robustness Visualization
# Visualizes SDR clusters for each letter across fonts with augmentations (rotation, noise, etc)
import subprocess
import sys
import os
import string

FONTS = ["FreeSans", "Arial", "Times New Roman", "Comic Sans MS", "Calibri", "Verdana", "Courier New", "Georgia"]

for font in FONTS:
    print(f"Visualizing SDRs for font {font} with augmentations...")
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "visualize_sdr_clusters.py"),
                    "--only-font", font, "--augment", "--save", f"sdr_clusters_aug_{font.replace(' ', '_')}.png"])
print("Augmentation robustness visualizations complete.")
