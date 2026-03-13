import sys
sys.path.append("src")

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from model import LSTMAttention
import os
from torch.utils.data import DataLoader, TensorDataset


def train_model():

    # Load dataset
    X = np.load("data/processed/X.npy")
    y = np.load("data/processed/y.npy")

    print("Dataset shape:", X.shape)

    # Train-test split (time-series safe)
    split = int(len(X) * 0.8)

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Create DataLoader (mini-batch training)
    batch_size = 256

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = LSTMAttention(input_size=X.shape[2])

    pos_weight = torch.tensor([1.5])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 20

    # Training loop
    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for batch_X, batch_y in train_loader:

            outputs = model(batch_X).squeeze()

            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    

    # Evaluation
    model.eval()

    with torch.no_grad():

        logits = model(X_test).squeeze()

        probs = torch.sigmoid(logits)

        preds = (probs > 0.65).float()

    acc = accuracy_score(y_test.numpy(), preds.numpy())

    print("\nTest Accuracy:", acc)

    # Save model
    os.makedirs("models", exist_ok=True)

    torch.save(model.state_dict(), "models/crash_model.pth")

    print("Model saved to models/crash_model.pth")


if __name__ == "__main__":
    train_model()