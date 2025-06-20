#
# File: src/cognitive/recognizer.py
#
# Description: This module defines the new "brain" of the AI, a
# Convolutional Neural Network (CNN) specifically designed for
# character recognition.
#

import torch
import torch.nn as nn
import torch.nn.functional as F

class CharacterRecognizer(nn.Module):
    """
    A Convolutional Neural Network (CNN) designed to recognize character shapes.
    This architecture preserves spatial information, allowing it to differentiate
    between visually similar letters like 'B', 'C', and 'T'.
    """
    def __init__(self, num_classes=26):
        super(CharacterRecognizer, self).__init__()
        
        # --- Stage 1: First Convolutional Layer ---
        # Takes the 1-channel image (64x64). The 16 filters will learn to
        # detect low-level features like simple edges and curves.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        
        # Pooling layer to reduce dimensionality from 64x64 to 32x32
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- Stage 2: Second Convolutional Layer ---
        # Takes the 16 feature maps from Stage 1. The 32 filters will learn
        # more complex features by combining the simple features from the first layer.
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        # Another pooling layer reduces dimensionality from 32x32 to 16x16

        # --- Stage 3: The Classifier (Fully Connected Layers) ---
        # A standard neural network that takes the final feature map and classifies it.
        # The input size is 32 (channels) * 16 (height) * 16 (width).
        self.fc1 = nn.Linear(32 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes) # Output layer for each letter class

    def forward(self, x):
        """Defines the forward pass of information through the network."""
        # Pass through first convolutional layer, then activation, then pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Pass through second convolutional layer, then activation, then pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the 2D feature maps into a 1D vector for the classifier
        x = x.view(-1, 32 * 16 * 16)
        
        # Pass through the classification layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x