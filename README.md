## Introduction

Protein flexibility, commonly measured by B-factors, is closely related to structure and function. How-
ever, accurate prediction remains challenging due to the multiscale nature of protein structures and com-
plex atomic interactions. In this work, we propose a commutative algebra-based learning framework,
dubbed as CAL, for protein B-factor prediction.
Our CAL model demonstrates an increase in accuracy of 34.5% compared to the classical Gaussian
network model (GNM) in predicting B-factors for a data set of 364 proteins.
Experiments with both linear and ensemble models demonstrate that CAL achieves strong and stable
performance across all datasets and is competitive with existing methods. The results show that CAL
provides an effective and efficient framework for protein flexibility prediction.



## CAL Feature Generation and Model Training

### I. CA coordinate extraction
Extract CA atomic coordinates from protein structures for downstream feature construction.

```bash
python codes/extract_ca_coordinates.py
```

### II. CAL feature generation
Construct CAL-based topological representations from CA coordinates for each protein.

```bash
python python codes/cal_features.py
```

### III. Linear modeling with CAL features
Evaluate the predictive capability of CAL features using a linear regression model.

```bash
python codes/linear_model_evaluation.py
```

### IV. Feature fusion (CAL + auxiliary descriptors)
Merge CAL features with additional descriptors to form the blind prediction input representation.

```bash
python python codes/merge_cal_additional_features.py
```

### V. Residue-level blind prediction (RF / GBDT)
Perform residue-level blind evaluation using Random Forest and Gradient Boosting models.

```bash
python codes/train_rf_gbdt_residue_blind.py
```

### VI. Protein-level blind prediction (RF / GBDT)
Evaluate protein-level prediction performance under blind setting using ensemble learning models.

```bash
python codes/train_rf_gbdt_protein_blind.py
```

### VII. Leave-One-Protein-Out (LOPO) evaluation
Perform strict generalization testing using LOPO cross-validation strategy.

```bash
python codes/train_rf_gbdt_lopo_blind.py
```



## Prerequisites

- numpy 1.21.0  
- scipy 1.7.3  
- scikit-learn 1.0.2  
- python 3.10.12  
- biopython 1.75  
- pandas 1.4.1  
- gudhi 3.5.0  