
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd

from ledc import LEDC
from ledc.plotting import plot_spatial


# 1. File locations
data_dir = Path(
    "/bigdata/stat288/shared/data/processed/benchmark_visium"
)

adata_path = Path(
    "/bigdata/stat288/shared/code/makenagrigsby/"
    "benchmark_artifacts/visium_T.h5ad"
)

model_dir = Path.home() / "ledc_notebook_validation/new_models"

output_dir = Path.home() / "ledc_lymph_results"
output_dir.mkdir(parents=True, exist_ok=True)


# 2. Load data and trained model
adata = ad.read_h5ad(adata_path)
pos = pd.read_csv(data_dir / "tissue_positions_aligned.csv")
model = LEDC.load(model_dir)

print("Expression shape:", adata.shape)
print("Coordinate shape:", pos.shape)


# 3. Predict cell-type proportions
predictions = model.predict_adata(
    adata,
    gene_id_column="gene_ids",
    batch_size=128,
    missing_genes="error"
)

print("Prediction shape:", predictions.shape)

predictions.to_csv(
    output_dir / "lymph_predictions.csv"
)


# 4. Select 8 cell types with highest mean predicted abundance
selected_types = (
    predictions.mean()
    .sort_values(ascending=False)
    .head(8)
    .index.tolist()
)

print("\nSelected cell types:")
print(selected_types)


# 5. Plot spatial predictions
fig = plot_spatial(
    predictions=predictions,
    coordinates=pos,
    cell_types=selected_types,
    ncols=4,
    cmap="viridis",
    scale="individual",
    vmax="quantile",
    quantile=0.99,
    tissue_only=True,
    spot_size=20
)

fig.suptitle(
    "LEDC predicted cell-type proportions — Lymph Node",
    fontsize=16
)

fig.savefig(
    output_dir / "lymph_spatial_predictions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("\nResults saved to:", output_dir)
