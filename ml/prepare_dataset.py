"""
prepare_dataset.py

Takes raw CSV log files from the robot and builds a clean numpy dataset
for training the ML model.

Expected CSV format (you can tweak this):
    time, x, y, theta, f1, f2, ..., fN

We will:
  - load all CSVs from LOG_DIR
  - stack them into one big array
  - choose some columns as input features (X)
  - choose next-position (x,y one step ahead) as target (y)
  - normalise features
  - save dataset as datasets/slam_dataset.npz
"""
import os
from pathlib import Path

import numpy as np

# ---- paths ----
ROOT = Path(__file__).resolve().parents[1]          
LOG_DIR = ROOT / "logs"                             
OUT_DIR = ROOT / "ml" / "datasets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# change this to match your actual CSV column order
# e.g. 0=time, 1=x, 2=y, 3=theta, 4..=sensor features
X_COLS = slice(3, None)   # all columns from index 3 onwards as input
POS_X_COL = 1             # x position column index
POS_Y_COL = 2             # y position column index


def load_all_logs():
    csv_files = sorted(LOG_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV logs found in {LOG_DIR}. "
            "Drop your robot logs there (e.g. run_logs_*.csv)."
        )

    data_list = []
    for f in csv_files:
        arr = np.genfromtxt(f, delimiter=",", dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        data_list.append(arr)

    data = np.vstack(data_list)
    # drop NaNs if any
    data = data[~np.isnan(data).any(axis=1)]
    return data


def build_supervised_dataset(data: np.ndarray):
    """
    Turn a time-series into (state_t -> position_{t+1}) pairs.
    """
    X_raw = data[:-1, X_COLS]  # all but last row
    next_pos = data[1:, [POS_X_COL, POS_Y_COL]]  # x,y one step ahead

    # normalise features (mean/std)
    mean = X_raw.mean(axis=0, keepdims=True)
    std = X_raw.std(axis=0, keepdims=True) + 1e-8
    X = (X_raw - mean) / std

    return X.astype(np.float32), next_pos.astype(np.float32), mean, std


def main():
    print(f"[prepare_dataset] Loading logs from: {LOG_DIR}")
    data = load_all_logs()
    print(f"[prepare_dataset] Raw data shape: {data.shape}")

    X, y, mean, std = build_supervised_dataset(data)
    out_path = OUT_DIR / "slam_dataset.npz"

    np.savez(out_path, X=X, y=y, mean=mean, std=std)
    print(f"[prepare_dataset] Saved dataset to: {out_path}")
    print(f"  X shape: {X.shape}, y shape: {y.shape}")


if __name__ == "__main__":
    main()