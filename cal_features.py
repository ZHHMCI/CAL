import os
import numpy as np
from sklearn.preprocessing import StandardScaler


# ============================================================
# f_vector class (unchanged logic, cleaned version)
# ============================================================

class f_vector:

    def __init__(self, points, min_edge_length=1, max_edge_length=12, max_dim=3, num_samples=23):
        self.points = points
        self.min_edge_length = min_edge_length
        self.max_edge_length = max_edge_length
        self.max_dim = max_dim
        self.num_samples = num_samples

    def compute_f_vector_at_t(self, simplices_with_filtration, t):
        f_vector = [0] * (self.max_dim + 1)
        for simplex, filtration in simplices_with_filtration:
            if filtration <= t:
                dim = len(simplex) - 1
                if dim <= self.max_dim:
                    f_vector[dim] += 1
        return f_vector

    def compute_f_vector_curves(self):

        import gudhi

        rips_complex = gudhi.RipsComplex(
            points=self.points,
            max_edge_length=self.max_edge_length
        )

        simplex_tree = rips_complex.create_simplex_tree(
            max_dimension=self.max_dim
        )

        simplices_with_filtration = list(simplex_tree.get_simplices())

        t_values = np.linspace(
            self.min_edge_length,
            self.max_edge_length,
            self.num_samples
        )

        f_curves = {d: [] for d in range(self.max_dim + 1)}

        for t in t_values:
            f_vec = self.compute_f_vector_at_t(simplices_with_filtration, t)
            for d in range(self.max_dim + 1):
                f_curves[d].append(f_vec[d])

        self.f_curves = f_curves
        return f_curves

    def compute_rate_curves(self):

        f_curves = self.f_curves

        t_values = np.linspace(
            self.min_edge_length,
            self.max_edge_length,
            self.num_samples
        )

        cumulative_rates = {}

        for d, f_vals in f_curves.items():
            f_vals = np.array(f_vals)
            rate = np.zeros_like(f_vals)

            mask = t_values > 0
            rate[mask] = f_vals[mask] / t_values[mask]

            cumulative_rates[d] = rate

        return cumulative_rates


# ============================================================
# Paths
# ============================================================

list_file = "data/lists/list-365.txt"
coord_folder = "data/ca_coordinates"
base_save_folder = "data/features_cal"


# ============================================================
# Parameters
# ============================================================

R_values = [12, 12]
num_samples = 4
dimensions = [1, 2]


R_str = "-".join(map(str, R_values))
dim_str = "-".join(map(str, dimensions))

folder_name = f"R_{R_str}_dims_{dim_str}_samples_{num_samples}"

save_folder = os.path.join(base_save_folder, folder_name)
os.makedirs(save_folder, exist_ok=True)


# ============================================================
# Load protein list
# ============================================================

with open(list_file, "r") as f:
    pdb_names = [line.strip() for line in f if line.strip()]


# ============================================================
# Feature extraction
# ============================================================

def process_protein(pdbid):

    coord_path = os.path.join(coord_folder, f"{pdbid}_coords.npy")

    if not os.path.exists(coord_path):
        return None

    coords = np.load(coord_path)

    protein_features = []

    for center_idx in range(len(coords)):

        center = coords[center_idx]
        atom_feature = []

        for R in R_values:

            dists = np.linalg.norm(coords - center, axis=1)

            neighbors = coords[(dists <= R) & (dists > 0)]

            if len(neighbors) < 2:

                zero_len = len(dimensions) * num_samples
                atom_feature.extend([0] * zero_len)
                continue

            fv = f_vector(
                points=neighbors,
                max_dim=max(dimensions),
                min_edge_length=0,
                max_edge_length=R,
                num_samples=num_samples
            )

            f_curves = fv.compute_f_vector_curves()

            for dim in dimensions:

                curve = f_curves[dim]

                if R == R_values[0]:
                    curve = curve[1:]

                atom_feature.extend(curve)

        protein_features.append(atom_feature)

    protein_features = np.array(protein_features)

    scaler = StandardScaler()
    protein_features = scaler.fit_transform(protein_features)

    np.save(
        os.path.join(save_folder, f"{pdbid}_features.npy"),
        protein_features
    )

    return pdbid


# ============================================================
# Run sequentially (parallel removed as requested)
# ============================================================

for pdbid in pdb_names:
    process_protein(pdbid)

print("Finished.")