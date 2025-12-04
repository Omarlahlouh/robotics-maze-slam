"""
train_model.py

Trains a small PyTorch MLP on the dataset produced by prepare_dataset.py.
The model learns to predict the next robot (x, y) position from the
current state / sensor features.

Outputs:
  - ml_model.pt           (PyTorch weights)
  - ml_model_meta.npz     (input dim etc)
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "ml" / "datasets" / "slam_dataset.npz"
MODEL_PATH = ROOT / "ml" / "ml_model.pt"
META_PATH = ROOT / "ml" / "ml_model_meta.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # output: next x, y
        )

    def forward(self, x):
        return self.net(x)


def load_dataset(test_ratio=0.2):
    data = np.load(DATA_PATH)
    X = data["X"]
    y = data["y"]

    n = X.shape[0]
    n_test = int(n * test_ratio)

    X_train, X_test = X[:-n_test], X[-n_test:]
    y_train, y_test = y[:-n_test], y[-n_test:]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    return train_ds, test_ds


def train():
    train_ds, test_ds = load_dataset()
    in_dim = train_ds.tensors[0].shape[1]

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    model = MLP(in_dim).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    epochs = 25
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optim.step()
            total_loss += loss.item() * xb.size(0)

        train_loss = total_loss / len(train_ds)

        # quick validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = model(xb)
                loss = loss_fn(preds, yb)
                val_loss += loss.item() * xb.size(0)
            val_loss /= len(test_ds)

        print(f"[Epoch {epoch:02d}] train MSE={train_loss:.5f} | val MSE={val_loss:.5f}")

    # save model + meta
    torch.save(model.state_dict(), MODEL_PATH)
    meta = {
        "input_dim": in_dim,
        "hidden_dim": 64,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[train_model] Saved model to {MODEL_PATH}")
    print(f"[train_model] Saved meta  to {META_PATH}")


if __name__ == "__main__":
    train()