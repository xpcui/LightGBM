
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt

from ledc import LEDC, generate_pseudobulk, evaluate_predictions


# --------------------------------------------------
# 1. Generate synthetic single-cell RNA-seq data
# --------------------------------------------------

rng = np.random.default_rng(42)

cell_types = ["B cell", "T cell", "Macrophage"]
n_genes = 30
n_cells_per_type = 100

# Give each cell type a distinct expression pattern
X_list = []

for k in range(len(cell_types)):
    means = np.ones(n_genes)
    means[k * 10:(k + 1) * 10] = 8

    X_list.append(
        rng.poisson(
            lam=means,
            size=(n_cells_per_type, n_genes)
        )
    )

X_sc = np.vstack(X_list)

adata = ad.AnnData(X_sc)

adata.obs["cell_type"] = np.repeat(
    cell_types,
    n_cells_per_type
)

adata.var_names = [
    f"Gene_{i}" for i in range(n_genes)
]

print("Single-cell data:", adata.shape)


# --------------------------------------------------
# 2. Generate pseudobulk training and test data
# --------------------------------------------------

T_train, C_train, _ = generate_pseudobulk(
    adata=adata,
    n_samples=200,
    cell_types=cell_types,
    rng=np.random.default_rng(42),
    mode="count",
    poisson_mean=10
)

T_test, C_test, _ = generate_pseudobulk(
    adata=adata,
    n_samples=50,
    cell_types=cell_types,
    rng=np.random.default_rng(123),
    mode="count",
    poisson_mean=10
)

X_train = pd.DataFrame(
    T_train.toarray(),
    columns=adata.var_names
)

X_test = pd.DataFrame(
    T_test.toarray(),
    columns=adata.var_names
)

print("Training data:", X_train.shape)
print("Test data:", X_test.shape)


# --------------------------------------------------
# 3. Train LEDC
# --------------------------------------------------

model = LEDC(
    target="count",
    n_estimators=100,
    nfold=3,
    early_stopping_rounds=10,
    random_state=42
)

model.fit(X_train, C_train)

print("\nLEDC training completed!")


# --------------------------------------------------
# 4. Predict cell-type counts
# --------------------------------------------------

predictions = model.predict(X_test)

print("\nPredicted cell counts:")
print(predictions.head().round(2))


# --------------------------------------------------
# 5. Evaluate predictions
# --------------------------------------------------

scores = evaluate_predictions(
    C_test,
    predictions
)

print("\nEvaluation results:")
print(scores.round(3))


# --------------------------------------------------
# 6. Plot true versus predicted cell counts
# --------------------------------------------------

fig, axes = plt.subplots(
    1, 3,
    figsize=(12, 4)
)

for ax, ct in zip(axes, cell_types):

    ax.scatter(
        C_test[ct],
        predictions[ct],
        alpha=0.7
    )

    max_value = max(
        C_test[ct].max(),
        predictions[ct].max()
    )

    ax.plot(
        [0, max_value],
        [0, max_value],
        "k--"
    )

    ax.set_title(ct)
    ax.set_xlabel("True cell count")
    ax.set_ylabel("Predicted cell count")
    ax.set_aspect("equal", adjustable="box")

plt.tight_layout()

plt.savefig(
    "examples/demo_output.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nDemo completed successfully!")
