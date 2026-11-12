
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ledc.plotting import plot_spatial


# Load saved predictions and coordinates
result_dir = Path.home() / "ledc_lymph_results"

data_dir = Path(
    "/bigdata/stat288/shared/data/processed/benchmark_visium"
)

predictions = pd.read_csv(
    result_dir / "lymph_predictions.csv",
    index_col=0
)

pos = pd.read_csv(
    data_dir / "tissue_positions_aligned.csv"
)

# Select the 8 most abundant predicted cell types
selected_types = (
    predictions.mean()
    .sort_values(ascending=False)
    .head(8)
    .index.tolist()
)


# Plot 1: Individual color scales
fig = plot_spatial(
    predictions=predictions,
    coordinates=pos,
    cell_types=selected_types,
    ncols=4,
    cmap="jet",
    scale="individual",
    vmax="quantile",
    quantile=0.99,
    tissue_only=True,
    spot_size=15,
)

fig.suptitle("LEDC — Lymph Node (Individual Scales)")
fig.savefig(
    result_dir / "lymph_individual.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)


# Plot 2: Shared color scale
fig = plot_spatial(
    predictions=predictions,
    coordinates=pos,
    cell_types=selected_types,
    ncols=4,
    cmap="viridis",
    scale="shared",
    vmax="max",
    tissue_only=True,
    spot_size=15,
)

fig.suptitle("LEDC — Lymph Node (Shared Scale)")
fig.savefig(
    result_dir / "lymph_shared.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)


# Plot 3: All 39 cell types
fig = plot_spatial(
    predictions=predictions,
    coordinates=pos,
    ncols=4,
    scale="individual",
    vmax="quantile",
    quantile=0.99,
    tissue_only=True,
    spot_size=15,
)

fig.savefig(
    result_dir / "lymph_all_cell_types.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close(fig)

print("All spatial plots saved successfully!")
