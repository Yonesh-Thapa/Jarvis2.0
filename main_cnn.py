#
# File: main.py (Version for CNN)
#
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import List, Dict

# Import the new recognizer model and the webcam transducer
from src.cognitive.recognizer import CharacterRecognizer
from src.interfaces.transducers import Webcam

def main():
    """The main function that initializes and runs the CNN training and verification."""
    print("=========================================")
    print("=     AI Recognizer v3.0 (CNN)          =")
    print("=========================================\n")

    # 1. --- INITIALIZE MODEL, DATA, AND TRAINING COMPONENTS ---
    print("PHASE I: INITIALIZING SYSTEMS...")
    
    # Use a CPU or GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the CNN model and move it to the selected device
    model = CharacterRecognizer(num_classes=3).to(device)
    
    # The webcam will provide our image data
    webcam = Webcam()
    
    # Define the loss function (Cross Entropy is standard for classification)
    # and the optimizer (Adam is a robust choice)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Define the training data
    letters_to_learn: List[str] = ['A', 'B', 'C']
    letter_to_label: Dict[str, int] = {'A': 0, 'B': 1, 'C': 2}
    
    num_epochs: int = 10
    steps_per_epoch: int = 150
    
    print("\nINITIALIZATION COMPLETE. BEGINNING TRAINING.\n")

    # 2. --- TRAINING LOOP ---
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        # Create a shuffled sequence for training
        training_sequence = letters_to_learn * (steps_per_epoch // len(letters_to_learn))
        random.shuffle(training_sequence)

        for i, letter_char in enumerate(training_sequence):
            # --- Get Data ---
            webcam.set_stimulus(letter_char)
            frame = webcam.capture_frame()
            label = letter_to_label[letter_char]

            # Convert numpy image and label to PyTorch tensors
            # Add batch and channel dimensions to the image tensor: [1, 1, 64, 64]
            images = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).to(device)
            labels = torch.tensor([label]).to(device)

            # --- Training Step ---
            optimizer.zero_grad()      # Clear previous gradients
            outputs = model(images)    # Forward pass: get model predictions
            loss = criterion(outputs, labels) # Calculate loss
            loss.backward()            # Backward pass: calculate gradients
            optimizer.step()           # Update model weights

            running_loss += loss.item()

            # Track accuracy for this epoch
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / steps_per_epoch
        epoch_acc = 100 * correct_predictions / total_predictions
        print(f"--- Epoch {epoch + 1}/{num_epochs} --- Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    # 3. --- FINAL VERIFICATION ---
    print("\n--- TRAINING COMPLETE. BEGINNING FINAL VERIFICATION. ---")
    model.eval() # Set the model to evaluation mode
    with torch.no_grad(): # Disable gradient calculation for verification
        for letter_char in letters_to_learn:
            webcam.set_stimulus(letter_char)
            frame = webcam.capture_frame()
            images = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).to(device)
            
            outputs = model(images)
            _, predicted_idx = torch.max(outputs.data, 1)
            
            predicted_char = [k for k, v in letter_to_label.items() if v == predicted_idx.item()][0]
            
            print(f"Presented with '{letter_char}', model predicted: '{predicted_char}' -> {'Correct' if predicted_char == letter_char else 'INCORRECT'}")

    print("\nAI operation complete.")

if __name__ == "__main__":
    main()