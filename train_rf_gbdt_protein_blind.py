import os
import numpy as np
from Bio.PDB import PDBParser

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


# ============================================================
# paths
# ============================================================

DATA_ROOT = "data"
FEATURE_DIR = "data/features_blind"
PDB_DIR = "data/pdbs"
LIST_DIR = "data/lists"
RESULT_DIR = "results/protein_ml_benchmark"

os.makedirs(RESULT_DIR, exist_ok=True)


datasets = ["small", "medium", "large", "superset"]
models = ["RF", "GBDT"]

NKF = 10
NSEED = 10


parser = PDBParser(QUIET=True)


# ============================================================
# label
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
# main loop
# ============================================================

results = []

for dataset in datasets:

    list_file = f"{LIST_DIR}/list-{dataset}.txt"

    with open(list_file) as f:
        pdb_list = [x.strip() for x in f if x.strip()]

    for model_name in models:

        for seed in range(NSEED):

            # ====================================================
            # protein-level CV split
            # ====================================================

            valid_proteins = []

            protein_features = {}
            protein_labels = {}

            for pdbid in pdb_list:

                feat_path = f"{FEATURE_DIR}/{pdbid}_features.npy"
                pdb_path = f"{PDB_DIR}/{pdbid}.pdb"

                if not (os.path.exists(feat_path) and os.path.exists(pdb_path)):
                    continue

                X = np.load(feat_path)
                y = read_ca_bfactors(pdbid, pdb_path)

                if len(X) != len(y):
                    continue

                protein_features[pdbid] = X
                protein_labels[pdbid] = y
                valid_proteins.append(pdbid)

            valid_proteins = np.array(valid_proteins)

            kf = KFold(
                n_splits=NKF,
                shuffle=True,
                random_state=seed
            )

            for fold, (train_idx, test_idx) in enumerate(kf.split(valid_proteins)):

                train_pdbs = valid_proteins[train_idx]
                test_pdbs = valid_proteins[test_idx]

                X_train, y_train = [], []
                X_test, y_test = [], []

                for p in train_pdbs:
                    X_train.append(protein_features[p])
                    y_train.append(protein_labels[p])

                for p in test_pdbs:
                    X_test.append(protein_features[p])
                    y_test.append(protein_labels[p])

                X_train = np.concatenate(X_train)
                y_train = np.concatenate(y_train)
                X_test = np.concatenate(X_test)
                y_test = np.concatenate(y_test)

                # ====================================================
                # model
                # ====================================================

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

                corr = pearsonr(y_test, pred)[0]
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
# save results
# ============================================================

import pandas as pd

df = pd.DataFrame(results, columns=[
    "dataset", "model", "seed", "fold", "pearson", "rmse"
])

df.to_csv(f"{RESULT_DIR}/protein_ml_results.csv", index=False)

print("Saved:", f"{RESULT_DIR}/protein_ml_results.csv")
print("Total experiments:", len(df))