# Visualize SDRs for each letter across fonts using PCA
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import string
import numpy as np
import pygame
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, pairwise_distances
import umap
from sklearn.manifold import TSNE
from src.interfaces.transducers import Webcam
from src.vision.feature_extractor import FeatureExtractor
from src.cognitive.perception.stroke_detector import StrokeDetector
from src.cognitive.perception.compositional_layer import CompositionalLayer
import argparse

pygame.init()
LETTERS = string.ascii_uppercase
IMG_SIZE = 64
FONTS = ["FreeSans", "Arial", "Times New Roman", "Comic Sans MS", "Calibri", "Verdana", "Courier New", "Georgia"]

parser = argparse.ArgumentParser()
parser.add_argument('--only-font', type=str, default=None, help='Visualize only this font')
parser.add_argument('--save', type=str, default=None, help='Custom filename for PCA plot')
parser.add_argument('--augment', action='store_true', help='Apply augmentations (rotation, noise)')
args = parser.parse_args()

feature_extractor = FeatureExtractor()
stroke_detector = StrokeDetector()
compositional_layer = CompositionalLayer()

fonts_to_use = FONTS if args.only_font is None else [args.only_font]

sdrs = []
labels = []
font_labels = []
for font_name in fonts_to_use:
    for letter in LETTERS:
        font_obj = pygame.font.SysFont(font_name, IMG_SIZE)
        webcam = Webcam(font=font_obj, size=(IMG_SIZE, IMG_SIZE))
        webcam.set_stimulus(letter)
        if args.augment:
            # Example: rotate by 15 deg, add noise
            img = webcam.capture_frame(augment=True)
        else:
            img = webcam.capture_frame(augment=False)
        edge_map = feature_extractor.extract_edge_maps(img)
        stroke_sdr = stroke_detector.detect((edge_map['vertical'] + edge_map['horizontal']) / 2)
        sdr, _ = compositional_layer.encode(stroke_sdr)
        sdrs.append(sdr)
        labels.append(letter)
        font_labels.append(font_name)

sdrs = np.array(sdrs)
pca = PCA(n_components=2)
proj = pca.fit_transform(sdrs)

# PCA plot
plt.figure(figsize=(12, 8))
for i, letter in enumerate(LETTERS):
    idxs = [j for j, l in enumerate(labels) if l == letter]
    plt.scatter(proj[idxs, 0], proj[idxs, 1], label=letter, alpha=0.7)
plt.legend()
plt.title("SDR Clusters by Letter Across Fonts (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
if args.save:
    plt.savefig(args.save, dpi=300)
else:
    plt.savefig("sdr_clusters.png", dpi=300)
plt.close()

# --- Quantitative Cluster Metrics ---
# Silhouette score (higher is better, max=1)
try:
    # Silhouette score requires at least 2 clusters and n_samples > n_labels
    unique_labels = set(labels)
    if len(unique_labels) < 2 or len(sdrs) <= len(unique_labels):
        sil_score = None
        print(f"[WARN] Not enough clusters or samples for silhouette_score: n_labels={len(unique_labels)}, n_samples={len(sdrs)}")
    else:
        sil_score = silhouette_score(sdrs, labels)
except Exception as e:
    sil_score = None
    print(f"[ERROR] Silhouette score calculation failed: {e}")

# Intra/inter-class distances
pairwise = pairwise_distances(sdrs)
letters = np.array(labels)

intra = np.mean([np.mean(pairwise[(letters==l)][:,letters==l]) for l in LETTERS])
inter = np.mean([np.mean(pairwise[(letters==l1)][:,letters==l2]) for l1 in LETTERS for l2 in LETTERS if l1!=l2])

with open("sdr_cluster_metrics.txt", "w") as f:
    if sil_score is not None:
        f.write(f"Silhouette score: {sil_score:.4f}\n")
    else:
        f.write("Silhouette score: N/A (not enough clusters or samples)\n")
    f.write(f"Mean intra-class distance: {intra:.4f}\n")
    f.write(f"Mean inter-class distance: {inter:.4f}\n")

# --- t-SNE Visualization ---
tsne_proj = TSNE(n_components=2, random_state=42).fit_transform(sdrs)
plt.figure(figsize=(12, 8))
for i, letter in enumerate(LETTERS):
    idxs = [j for j, l in enumerate(labels) if l == letter]
    plt.scatter(tsne_proj[idxs, 0], tsne_proj[idxs, 1], label=letter, alpha=0.7)
plt.legend()
plt.title("SDR Clusters by Letter Across Fonts (t-SNE)")
plt.xlabel("tSNE-1")
plt.ylabel("tSNE-2")
plt.savefig("sdr_clusters_tsne.png", dpi=300)
plt.close()

# --- UMAP Visualization ---
umap_proj = umap.UMAP(n_components=2, random_state=42).fit_transform(sdrs)
plt.figure(figsize=(12, 8))
for i, letter in enumerate(LETTERS):
    idxs = [j for j, l in enumerate(labels) if l == letter]
    plt.scatter(umap_proj[idxs, 0], umap_proj[idxs, 1], label=letter, alpha=0.7)
plt.legend()
plt.title("SDR Clusters by Letter Across Fonts (UMAP)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.savefig("sdr_clusters_umap.png", dpi=300)
plt.close()
