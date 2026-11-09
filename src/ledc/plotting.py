
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_spatial(
    predictions,
    positions,
    cell_types=None,
    title="LEDC Spatial Predictions",
    ncols=4,
    point_size=12,
    cmap="viridis",
    shared_scale=True,
    vmax=None,
    quantile=None,
    save_path=None,
    dpi=300,
    show=True
):
    """
    Plot spatial cell-type abundance predictions.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Predicted counts or proportions.
        Rows = spots, columns = cell types.

    positions : pandas.DataFrame
        Spatial coordinates for the same spots.
        Required columns:
        - pxl_col_in_fullres
        - pxl_row_in_fullres
        Optional column:
        - in_tissue

    cell_types : list of str, optional
        Cell types to plot. Default is all cell types.

    title : str
        Overall figure title.

    ncols : int
        Maximum number of subplot columns.

    point_size : float
        Size of Visium spots.

    cmap : str
        Matplotlib colormap.

    shared_scale : bool
        If True, use one color scale for all cell types.

    vmax : float, optional
        Upper limit of the color scale.

    quantile : float, optional
        Quantile used to determine upper color limits.
        If None, use maximum observed predictions.

    save_path : str or Path, optional
        Path for saving the figure.

    dpi : int
        Figure resolution when saving.

    show : bool
        Whether to display the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The generated spatial figure.
    """

    # --------------------------------------------------
    # 1. Check input data
    # --------------------------------------------------

    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a DataFrame")

    if not isinstance(positions, pd.DataFrame):
        raise TypeError("positions must be a DataFrame")

    if predictions.empty:
        raise ValueError("predictions cannot be empty")

    if not predictions.index.is_unique:
        raise ValueError("Prediction spot indices must be unique")

    if not positions.index.is_unique:
        raise ValueError("Position spot indices must be unique")

    required_cols = [
        "pxl_col_in_fullres",
        "pxl_row_in_fullres"
    ]

    missing_cols = [
        col for col in required_cols
        if col not in positions.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing spatial columns: {missing_cols}"
        )

    if not predictions.index.equals(positions.index):
        if set(predictions.index) != set(positions.index):
            raise ValueError(
                "Prediction and position spot indices do not match"
            )

        positions = positions.reindex(predictions.index)

    if cell_types is None:
        cell_types = predictions.columns.tolist()
    else:
        cell_types = list(cell_types)

    if not cell_types or len(set(cell_types)) != len(cell_types):
        raise ValueError("cell_types must be nonempty and unique")

    missing_types = [
        ct for ct in cell_types
        if ct not in predictions.columns
    ]

    if missing_types:
        raise ValueError(
            f"Missing cell types: {missing_types}"
        )

    if not isinstance(ncols, int) or ncols < 1:
        raise ValueError("ncols must be a positive integer")

    if quantile is not None and not 0 < quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")

    if vmax is not None and (
        not np.isfinite(vmax) or vmax <= 0
    ):
        raise ValueError("vmax must be positive and finite")

    # --------------------------------------------------
    # 2. Combine predictions and spatial coordinates
    # --------------------------------------------------

    plot_df = pd.concat(
        [
            positions[required_cols],
            predictions[cell_types]
        ],
        axis=1
    )

    # Filter spots inside the tissue
    if "in_tissue" in positions.columns:
        plot_df = plot_df.loc[
            positions["in_tissue"] == 1
        ]

    if plot_df.empty:
        raise ValueError("No spots available for plotting")

    numeric_cols = required_cols + cell_types

    values = plot_df[numeric_cols].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Spatial coordinates and predictions must be finite"
        )

    # --------------------------------------------------
    # 3. Determine color scale
    # --------------------------------------------------

    abundance = plot_df[cell_types].to_numpy(dtype=float)

    if (abundance < 0).any():
        raise ValueError("Predicted abundances must be nonnegative")

    def get_upper_limit(values):
        if vmax is not None:
            return vmax

        if quantile is not None:
            upper = np.quantile(values, quantile)
        else:
            upper = np.max(values)

        return max(float(upper), 1e-12)

    if shared_scale:
        shared_vmax = get_upper_limit(abundance)
    else:
        shared_vmax = None

    # --------------------------------------------------
    # 4. Create subplot grid
    # --------------------------------------------------

    n = len(cell_types)
    ncols_use = min(ncols, n)
    nrows = math.ceil(n / ncols_use)

    fig, axes = plt.subplots(
        nrows,
        ncols_use,
        figsize=(4.5 * ncols_use, 4.2 * nrows),
        squeeze=False,
        constrained_layout=True
    )

    axes = axes.ravel()

    # --------------------------------------------------
    # 5. Plot each cell type
    # --------------------------------------------------

    for i, ct in enumerate(cell_types):

        ax = axes[i]

        cell_values = plot_df[ct].to_numpy(dtype=float)

        current_vmax = (
            shared_vmax
            if shared_scale
            else get_upper_limit(cell_values)
        )

        sc = ax.scatter(
            plot_df["pxl_col_in_fullres"],
            plot_df["pxl_row_in_fullres"],
            c=cell_values,
            s=point_size,
            cmap=cmap,
            vmin=0,
            vmax=current_vmax,
            edgecolors="none",
            rasterized=True
        )

        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_title(ct, fontsize=11)

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

        cbar = fig.colorbar(
            sc,
            ax=ax,
            fraction=0.04,
            pad=0.01
        )

        cbar.ax.tick_params(labelsize=8)

    # Remove unused subplot axes
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=16)

    # --------------------------------------------------
    # 6. Save and display figure
    # --------------------------------------------------

    if save_path is not None:
        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight"
        )

    if show:
        plt.show()

    return fig
