"""
evaluate_model.py

Loads the saved model and dataset, runs it on the test split,
and reports basic metrics (MSE + average position error in cm).
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_model import MLP, ROOT, DATA_PATH, MODEL_PATH, META_PATH, DEVICE  # reuse constants


def load_model_and_data():
    # dataset
    data = np.load(DATA_PATH)
    X = data["X"]
    y = data["y"]

    n = X.shape[0]
    n_test = int(n * 0.2)
    X_test = X[-n_test:]
    y_test = y[-n_test:]

    test_ds = TensorDataset(
        torch.from_numpy(X_test.astype(np.float32)),
        torch.from_numpy(y_test.astype(np.float32)),
    )
    test_loader = DataLoader(test_ds, batch_size=256)

    # meta + model
    with open(META_PATH) as f:
        meta = json.load(f)

    model = MLP(in_dim=meta["input_dim"], hidden_dim=meta["hidden_dim"])
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model, test_loader


def evaluate():
    model, test_loader = load_model_and_data()
    mse_loss = nn.MSELoss(reduction="sum")

    total_mse = 0.0
    total_samples = 0
    all_errors = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            total_mse += mse_loss(preds, yb).item()
            total_samples += xb.size(0)

            # Euclidean position error
            err = torch.linalg.norm(preds - yb, dim=1)  # metres
            all_errors.append(err.cpu().numpy())

    mse = total_mse / total_samples
    errors = np.concatenate(all_errors, axis=0)
    mean_err = errors.mean()
    median_err = np.median(errors)
    max_err = errors.max()

    print("\n=== Model Evaluation ===")
    print(f"MSE (x,y in metres^2): {mse:.6f}")
    print(f"Mean position error   : {mean_err*100:.2f} cm")
    print(f"Median position error : {median_err*100:.2f} cm")
    print(f"Max position error    : {max_err*100:.2f} cm")


if __name__ == "__main__":
    evaluate()