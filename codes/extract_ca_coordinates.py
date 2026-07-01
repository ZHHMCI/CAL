import os
import numpy as np


# =========================
# Paths
# =========================

list_file = "data/lists/list-365.txt"
pdb_folder = "data/364"
save_folder = "data/ca_coordinates"

os.makedirs(save_folder, exist_ok=True)


# =========================
# Load PDB list
# =========================

with open(list_file, "r") as f:
    pdb_ids = [line.strip() for line in f if line.strip()]


# =========================
# Extract CA coordinates
# =========================

for pdb_id in pdb_ids:

    pdb_path = os.path.join(pdb_folder, f"{pdb_id}.pdb")

    if not os.path.exists(pdb_path):
        continue

    ca_coords = []

    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                ca_coords.append([
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54])
                ])

    ca_coords = np.array(ca_coords)

    np.save(
        os.path.join(save_folder, f"{pdb_id}_ca_coords.npy"),
        ca_coords
    )