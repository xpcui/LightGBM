
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_spatial_legacy(
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



import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_spatial(
    predictions,
    coordinates,
    cell_types=None,
    ncols=4,
    cmap="viridis",
    scale="individual",
    vmax="quantile",
    quantile=0.99,
    tissue_only=True,
    spot_size=10,
    figsize=None,
    image=None,
    image_extent=None,
    image_alpha=0.6,
    show=False,
    save_path=None,
):
    """
    Plot predicted cell-type abundances on spatial coordinates.

    Parameters
    ----------
    predictions : pd.DataFrame
        Spots x cell types, indexed by spot barcode.
    coordinates : pd.DataFrame
        Spatial coordinates, indexed by barcode or containing
        a 'barcode' column. Requires pxl_col_in_fullres and
        pxl_row_in_fullres.
    cell_types : list or None
        Cell types to display. None plots all cell types.
    ncols : int
        Number of subplot columns.
    cmap : str
        Matplotlib colormap.
    scale : {'individual', 'shared'}
        Use separate or common color limits.
    vmax : {'quantile', 'max'} or number
        Upper color limit.
    quantile : float
        Quantile used when vmax='quantile'.
    tissue_only : bool
        Keep spots with in_tissue == 1 when available.
    spot_size : float
        Marker size.
    figsize : tuple or None
        Figure dimensions.
    image : array-like or None
        Optional histology image.
    image_extent : tuple or None
        Image extent (xmin, xmax, ymin, ymax) in
        the same coordinate system as the spots.
    image_alpha : float
        Histology image opacity.
    show : bool
        Whether to display the figure. Default is False.
    save_path : str or pathlib.Path or None
        Optional path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """

    # 1. Validate inputs
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a DataFrame")

    if not isinstance(coordinates, pd.DataFrame):
        raise TypeError("coordinates must be a DataFrame")

    if predictions.index.has_duplicates:
        raise ValueError("Duplicate spot IDs in predictions")

    if predictions.columns.has_duplicates:
        raise ValueError("Duplicate cell-type names")

    if ncols < 1:
        raise ValueError("ncols must be at least 1")

    if scale not in ("individual", "shared"):
        raise ValueError("scale must be 'individual' or 'shared'")

    if isinstance(vmax, str) and vmax not in ("quantile", "max"):
        raise ValueError("vmax must be 'quantile', 'max', or a number")

    if not 0 < quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")

    if not 0 <= image_alpha <= 1:
        raise ValueError("image_alpha must be between 0 and 1")

    if image is not None and image_extent is None:
        raise ValueError(
            "image_extent is required when providing an image"
        )

    # 2. Select cell types
    if cell_types is None:
        cell_types = predictions.columns.tolist()
    else:
        cell_types = list(cell_types)

    if not cell_types:
        raise ValueError("No cell types selected")

    missing_types = set(cell_types) - set(predictions.columns)
    if missing_types:
        raise ValueError(f"Unknown cell types: {sorted(missing_types)}")

    # 3. Align spots by barcode
    pos = coordinates.copy()

    if "barcode" in pos.columns:
        pos = pos.set_index("barcode")

    if pos.index.has_duplicates:
        raise ValueError("Duplicate spot IDs in coordinates")

    if isinstance(predictions.index, pd.RangeIndex):
        raise ValueError(
            "Predictions must be indexed by spot barcodes"
        )

    required = ["pxl_col_in_fullres", "pxl_row_in_fullres"]
    missing_columns = set(required) - set(pos.columns)

    if missing_columns:
        raise ValueError(
            f"Missing coordinate columns: {sorted(missing_columns)}"
        )

    missing_spots = predictions.index.difference(pos.index)
    if len(missing_spots) > 0:
        raise ValueError(
            f"{len(missing_spots)} predicted spots have no coordinates"
        )

    pos = pos.reindex(predictions.index)
    pred = predictions.loc[:, cell_types].copy()

    # 4. Optionally filter tissue spots
    if tissue_only and "in_tissue" in pos.columns:
        keep = pd.to_numeric(
            pos["in_tissue"], errors="coerce"
        ).eq(1)

        pos = pos.loc[keep]
        pred = pred.loc[keep]

    if pred.empty:
        raise ValueError("No spots remain after filtering")

    # 5. Extract coordinates and prediction values
    x = pd.to_numeric(
        pos["pxl_col_in_fullres"], errors="coerce"
    ).to_numpy()

    y = pd.to_numeric(
        pos["pxl_row_in_fullres"], errors="coerce"
    ).to_numpy()

    values = pred.to_numpy(dtype=float)

    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Coordinates must be finite numbers")

    if not np.isfinite(values).all():
        raise ValueError("Predictions contain NaN or infinity")

    # 6. Determine color limits
    def get_vmax(v):
        if isinstance(vmax, str):
            if vmax == "quantile":
                upper = np.quantile(v, quantile)
            else:
                upper = np.max(v)
        else:
            upper = float(vmax)

        if not np.isfinite(upper):
            raise ValueError("vmax must be finite")

        return max(float(upper), 1e-12)

    if scale == "shared":
        shared_vmax = get_vmax(values)

    # 7. Create subplot layout
    n = len(cell_types)
    ncols_used = min(ncols, n)
    nrows = math.ceil(n / ncols_used)

    if figsize is None:
        figsize = (4.5 * ncols_used, 4.5 * nrows)

    fig, axes = plt.subplots(
        nrows,
        ncols_used,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True
    )

    axes = axes.ravel()
    scatter_plots = []

   # 8. Plot each cell type
    for i, cell_type in enumerate(cell_types):
        ax = axes[i]
        v = values[:, i]

        upper = (
            shared_vmax
            if scale == "shared"
            else get_vmax(v)
        )

        # Optional H&E background
        if image is not None:
            ax.imshow(
                image,
                extent=image_extent,
                origin="upper",
                alpha=image_alpha
            )

        sc = ax.scatter(
            x,
            y,
            c=v,
            s=spot_size,
            cmap=cmap,
            vmin=0,
            vmax=upper,
            edgecolors="none",
            rasterized=True
        )

        scatter_plots.append(sc)

        ax.set_title(cell_type, fontsize=10)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")

        # Always display one colorbar per subplot
        cbar = fig.colorbar(
            sc,
            ax=ax,
            fraction=0.046,
            pad=0.02,
            shrink=0.85
        )

        cbar.ax.tick_params(labelsize=8)

    # 9. Hide unused subplots
    for ax in axes[n:]:
        ax.set_visible(False)

    # Save the figure if requested
    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    # Display the figure if requested
    if show:
        plt.show()

    return fig
