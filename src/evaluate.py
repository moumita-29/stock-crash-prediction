import torch
import numpy as np
from sklearn.metrics import classification_report
from model import LSTMAttention

X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")

X = torch.tensor(X,dtype=torch.float32)

model = LSTMAttention(input_size=X.shape[2])
model.load_state_dict(torch.load("models/crash_model.pth"))

with torch.no_grad():

    logits = model(X).squeeze()

    probs = torch.sigmoid(logits)

    preds = (probs > 0.35).float()

print(classification_report(y,preds))