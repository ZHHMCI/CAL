import os
import numpy as np


# ============================================================
# Paths
# ============================================================

CAL_DIR = "data/features_cal"
ONEHOT_DIR = "data/features_additional"
SAVE_DIR = "data/features_blind"

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# CAL features
# ============================================================

feature_files = [
    f for f in os.listdir(CAL_DIR)
    if f.endswith("_features.npy")
]


# ============================================================
# Fusion
# ============================================================

for file in feature_files:

    pdbid = file.replace("_features.npy", "")

    cal_path = os.path.join(CAL_DIR, file)
    onehot_path = os.path.join(ONEHOT_DIR, f"{pdbid}-onehot.csv")

    if not os.path.exists(onehot_path):
        continue

    X_cal = np.load(cal_path)
    X_onehot = np.loadtxt(onehot_path, delimiter=",")

    if len(X_cal) != len(X_onehot):
        continue

    X = np.concatenate([X_cal, X_onehot], axis=1)

    np.save(
        os.path.join(SAVE_DIR, f"{pdbid}_features.npy"),
        X
    )