import numpy as np
import pandas as pd
import anndata as ad

from ledc import generate_pseudobulk

# Simulate 100 cells and 20 genes
rng = np.random.default_rng(42)

X = rng.poisson(
    lam=5,
    size=(100, 20)
)

# Create an AnnData object
adata = ad.AnnData(X)

# Assign cell-type labels
adata.obs["cell_type"] = (
    ["B cell"] * 50 +
    ["T cell"] * 50
)

# Assign gene names
adata.var_names = [
    f"Gene_{i}" for i in range(20)
]

# Generate pseudobulk samples
T, C, P = generate_pseudobulk(
    adata=adata,
    n_samples=10,
    cell_types=["B cell", "T cell"],
    rng=np.random.default_rng(42),
    mode="count",
    poisson_mean=10
)

print("Expression shape:", T.shape)
print("Count shape:", C.shape)
print("Proportion shape:", P.shape)

print("\nCell counts:")
print(C.head())

print("\nProportions:")
print(P.head())