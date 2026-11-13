
from pathlib import Path
import argparse

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd

from ledc import LEDC, plot_spatial


def main():
    parser = argparse.ArgumentParser(
        description="Predict and visualize spatial cell-type abundances using LEDC."
    )

    # Required input paths
    parser.add_argument(
        "--adata", required=True,
        help="Path to spatial expression data (.h5ad)"
    )
    parser.add_argument(
        "--coordinates", required=True,
        help="Path to spatial coordinates (.csv)"
    )
    parser.add_argument(
        "--models", required=True,
        help="Directory containing saved LEDC models"
    )
    parser.add_argument(
        "--output", default="ledc_spatial_results",
        help="Output directory"
    )

    # Prediction options
    parser.add_argument(
        "--gene-id-column", default=None,
        help="Column in adata.var containing gene IDs; default uses var_names"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Number of spots predicted per batch"
    )
    parser.add_argument(
        "--missing-genes", choices=["error", "zero"],
        default="error",
        help="How to handle missing training genes"
    )
    parser.add_argument(
        "--validate-scale",
        action="store_true",
        help="Validate expression values against the model's declared scale"
    )

    # Visualization options
    parser.add_argument(
        "--cell-types", nargs="+", default=None,
        help="Cell types to plot; default plots all cell types"
    )
    parser.add_argument(
        "--ncols", type=int, default=4,
        help="Number of subplot columns"
    )
    parser.add_argument(
        "--cmap", default="viridis",
        help="Matplotlib colormap"
    )
    parser.add_argument(
        "--scale", choices=["individual", "shared"],
        default="individual",
        help="Use individual or shared color scales"
    )
    parser.add_argument(
        "--vmax", choices=["quantile", "max"],
        default="quantile",
        help="How to determine the upper color limit"
    )
    parser.add_argument(
        "--quantile", type=float, default=0.99,
        help="Quantile for color limit when --vmax=quantile"
    )
    parser.add_argument(
        "--spot-size", type=float, default=20,
        help="Scatter marker size"
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load expression, coordinates and model
    print("Loading spatial data...")
    adata = ad.read_h5ad(args.adata)
    positions = pd.read_csv(args.coordinates)
    model = LEDC.load(args.models)

    print("Expression shape:", adata.shape)
    print("Coordinate shape:", positions.shape)
    print("Model target:", model.target)

    # 2. Predict cell-type abundances
    print("\nRunning LEDC prediction...")

    predictions = model.predict_adata(
        adata,
        gene_id_column=args.gene_id_column,
        batch_size=args.batch_size,
        missing_genes=args.missing_genes,
        validate_scale=args.validate_scale
    )

    print("Prediction shape:", predictions.shape)

    prediction_file = output_dir / "spatial_predictions.csv"
    predictions.to_csv(prediction_file)

    # 3. Select cell types for visualization
    if args.cell_types is None:
        selected_types = predictions.columns.tolist()
    else:
        selected_types = args.cell_types

    print("\nCell types selected:")
    for cell_type in selected_types:
        print(" -", cell_type)

    # 4. Generate spatial plots
    print("\nCreating spatial maps...")

    fig = plot_spatial(
        predictions=predictions,
        coordinates=positions,
        cell_types=selected_types,
        ncols=args.ncols,
        cmap=args.cmap,
        scale=args.scale,
        vmax=args.vmax,
        quantile=args.quantile,
        tissue_only=True,
        spot_size=args.spot_size
    )

    fig.suptitle(
        f"LEDC predicted cell-type {model.target}s",
        fontsize=16
    )

    figure_file = output_dir / "spatial_maps.png"
    fig.savefig(
        figure_file,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    print("\nResults saved:")
    print("Predictions:", prediction_file)
    print("Spatial maps:", figure_file)


if __name__ == "__main__":
    main()
