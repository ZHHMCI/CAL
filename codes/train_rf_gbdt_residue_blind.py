import os
import numpy as np
from Bio.PDB import PDBParser

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


# ============================================================
# Paths
# ============================================================

DATA_ROOT = "data"
FEATURE_ROOT = "data/features_blind"
PDB_ROOT = "data/pdbs"
LIST_ROOT = "data/lists"
RESULT_DIR = "results/benchmark"
os.makedirs(RESULT_DIR, exist_ok=True)


datasets = ["small", "medium", "large", "superset"]
models = ["RF", "GBDT"]

NKF = 10
NSEED = 10


parser = PDBParser(QUIET=True)


# ============================================================
# Label
# ============================================================

def read_ca_bfactors(pdbid, pdb_path):
    structure = parser.get_structure(pdbid, pdb_path)
    return np.array([
        atom["CA"].get_bfactor()
        for model in structure
        for chain in model
        for residue in chain
        if "CA" in residue
    ])


# ============================================================
# Main evaluation
# ============================================================

results = []


for dataset in datasets:

    list_file = f"{LIST_ROOT}/list-{dataset}.txt"

    with open(list_file) as f:
        pdb_list = [x.strip() for x in f if x.strip()]

    for model_name in models:

        for seed in range(NSEED):

            for fold in range(NKF):

                X_all, y_all = [], []

                for pdbid in pdb_list:

                    feat_path = f"{FEATURE_ROOT}/{pdbid}_features.npy"
                    pdb_path = f"{PDB_ROOT}/{pdbid}.pdb"

                    if not os.path.exists(feat_path) or not os.path.exists(pdb_path):
                        continue

                    X = np.load(feat_path)
                    y = read_ca_bfactors(pdbid, pdb_path)

                    if len(X) != len(y):
                        continue

                    X_all.append(X)
                    y_all.append(y)

                X_all = np.concatenate(X_all)
                y_all = np.concatenate(y_all)

                kf = KFold(n_splits=NKF, shuffle=True, random_state=seed)
                train_idx, test_idx = list(kf.split(X_all))[fold]

                X_train, X_test = X_all[train_idx], X_all[test_idx]
                y_train, y_test = y_all[train_idx], y_all[test_idx]

                if model_name == "RF":
                    model = RandomForestRegressor(
                        n_estimators=1000,
                        max_depth=8,
                        n_jobs=-1,
                        random_state=seed
                    )
                else:
                    model = GradientBoostingRegressor(
                        n_estimators=1000,
                        max_depth=7,
                        learning_rate=0.002,
                        subsample=0.8,
                        random_state=seed
                    )

                model.fit(X_train, y_train)
                pred = model.predict(X_test)

                corr, _ = pearsonr(y_test, pred)
                rmse = np.sqrt(mean_squared_error(y_test, pred))

                results.append([
                    dataset,
                    model_name,
                    seed,
                    fold,
                    corr,
                    rmse
                ])

                print(dataset, model_name, seed, fold, corr, rmse)


# ============================================================
# Save results
# ============================================================

import pandas as pd

df = pd.DataFrame(results, columns=[
    "dataset", "model", "seed", "fold", "pearson", "rmse"
])

df.to_csv(f"{RESULT_DIR}/residue_ml_results", index=False)

print("Saved:", f"{RESULT_DIR}/residue_ml_results.csv")
print("Total experiments:", len(df))