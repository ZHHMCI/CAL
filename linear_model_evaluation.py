import os
import numpy as np
from Bio.PDB import PDBParser
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


# ============================================================
# Dataset paths
# ============================================================

list_files = {
    "small": "data/lists/list-small.txt",
    "medium": "data/lists/list-medium.txt",
    "large": "data/lists/list-large.txt",
    "superset": "data/lists/list-365.txt"
}

feature_folder = "data/features_cal"
pdb_folder = "data/364"


parser = PDBParser(QUIET=True)


# ============================================================
# CA B-factor extraction
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


final_results = {}


# ============================================================
# Evaluation
# ============================================================

for set_name, list_file in list_files.items():

    with open(list_file, "r") as f:
        pdb_names = [x.strip() for x in f if x.strip()]

    pearsons, mses = [], []

    for pdbid in pdb_names:

        feature_path = os.path.join(feature_folder, f"{pdbid}_features.npy")
        pdb_path = os.path.join(pdb_folder, f"{pdbid}.pdb")

        if not (os.path.exists(feature_path) and os.path.exists(pdb_path)):
            continue

        X = np.load(feature_path)
        y = read_ca_bfactors(pdbid, pdb_path)

        if len(X) != len(y):
            continue

        model = LinearRegression()
        y_pred = model.fit(X, y).predict(X)

        pearsons.append(pearsonr(y, y_pred)[0])
        mses.append(mean_squared_error(y, y_pred))

    final_results[set_name] = {
        "pearson": float(np.mean(pearsons)),
        "mse": float(np.mean(mses))
    }


# ============================================================
# Final output
# ============================================================

for k, v in final_results.items():
    print(f"{k:<10} | Pearson: {v['pearson']:.4f} | MSE: {v['mse']:.4f}")