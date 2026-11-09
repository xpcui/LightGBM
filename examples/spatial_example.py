
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ledc import plot_spatial

rng = np.random.default_rng(42)

# Create 100 spatial spots
spot_ids = [f"Spot_{i}" for i in range(100)]

positions = pd.DataFrame(
    {
        "pxl_col_in_fullres": np.tile(np.arange(10), 10),
        "pxl_row_in_fullres": np.repeat(np.arange(10), 10),
        "in_tissue": np.ones(100)
    },
    index=spot_ids
)

# Create synthetic cell-type abundance predictions
predictions = pd.DataFrame(
    {
        "B cell": rng.uniform(0, 10, 100),
        "T cell": rng.uniform(0, 15, 100),
        "Macrophage": rng.uniform(0, 5, 100)
    },
    index=spot_ids
)

# Generate spatial maps
fig = plot_spatial(
    predictions=predictions,
    positions=positions,
    cell_types=["B cell", "T cell", "Macrophage"],
    title="LEDC Spatial Prediction Example",
    shared_scale=True,
    ncols=3,
    point_size=80,
    save_path="examples/spatial_example.png"
)

plt.close(fig)

print("Spatial figure saved successfully!")
