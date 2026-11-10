
"""
Generate Endo/Epi pseudobulk datasets using LEDC.

This script:
1. Loads pre-split training and testing AnnData references.
2. Generates pseudobulk datasets for three folds.
3. Supports count and proportion prediction targets.
4. Preserves the same gene and cell-type ordering across folds.
5. Generates full-reference pseudobulk training datasets.

The input AnnData files must contain raw gene counts in .X
and cell-type labels in .obs["cell_type"].
"""

import argparse
from pathlib import Path

import anndata as ad

from ledc import (
    generate_fold_pseudobulk,
    generate_full_pseudobulk,
)


def main():

    # --------------------------------------------------
    # 1. Command-line arguments
    # --------------------------------------------------

    parser = argparse.ArgumentParser(
        description="Generate Endo/Epi pseudobulk datasets"
    )

    parser.add_argument(
        "--reference-dir",
        required=True,
        help="Directory containing the fold-specific AnnData files"
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where pseudobulk datasets will be saved"
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=5000,
        help="Number of pseudobulk samples per split"
    )

    args = parser.parse_args()

    reference_dir = Path(args.reference_dir)
    output_dir = Path(args.output_dir)

    # --------------------------------------------------
    # 2. Initialize shared cell-type and gene ordering
    # --------------------------------------------------

    cell_types = None
    training_genes = None
    full_reference = None

    # --------------------------------------------------
    # 3. Generate pseudobulk datasets for three folds
    # --------------------------------------------------

    for fold in [1, 2, 3]:

        print(f"\nGenerating Fold {fold}")

        train_path = (
            reference_dir /
            f"combined_fold_{fold}_train_rawcounts.h5ad"
        )

        test_path = (
            reference_dir /
            f"combined_fold_{fold}_test_rawcounts.h5ad"
        )

        if not train_path.exists():
            raise FileNotFoundError(train_path)

        if not test_path.exists():
            raise FileNotFoundError(test_path)

        train = ad.read_h5ad(train_path)
        test = ad.read_h5ad(test_path)

        print("Train reference:", train.shape)
        print("Test reference:", test.shape)

        # Establish the shared ordering from Fold 1.
        if fold == 1:

            cell_types = sorted(
                train.obs["cell_type"].astype(str).unique()
            )

            training_genes = train.var_names.tolist()

            print("Number of cell types:", len(cell_types))
            print("Number of genes:", len(training_genes))

            # Combine Fold 1 train/test references to reproduce
            # the full-reference construction in the notebook.
            full_reference = ad.concat(
                [train, test],
                join="inner",
                merge="same",
            )

        # Verify gene names and their order across folds.
        if (
            train.var_names.tolist() != training_genes
            or test.var_names.tolist() != training_genes
        ):
            raise ValueError(
                f"Gene names or ordering differ in Fold {fold}"
            )

        # --------------------------------------------------
        # 4. Generate proportion and count datasets
        # --------------------------------------------------

        for mode in ["proportion", "count"]:

            print(f"\nFold {fold} — Mode: {mode}")

            paths = generate_fold_pseudobulk(
                train_adata=train,
                test_adata=test,
                output_dir=output_dir,
                fold=fold,
                mode=mode,
                n_samples=args.n_samples,
                pb_cells=10,
                poisson_mean=10,
                alpha=1.0,
                cell_types=cell_types,
            )

            print("Saved files:")

            for name, path in paths.items():
                print(f"  {name}: {path}")

        del train, test

    # --------------------------------------------------
    # 5. Generate full-reference training datasets
    # --------------------------------------------------

    print("\nGenerating full-reference pseudobulk datasets")

    for mode in ["proportion", "count"]:

        print(f"\nFull reference — Mode: {mode}")

        paths = generate_full_pseudobulk(
            reference_adata=full_reference,
            output_dir=output_dir,
            mode=mode,
            n_samples=args.n_samples,
            pb_cells=10,
            poisson_mean=10,
            alpha=1.0,
            cell_types=cell_types,
        )

        print("Saved files:")

        for name, path in paths.items():
            print(f"  {name}: {path}")

    print("\nAll pseudobulk datasets generated successfully!")


if __name__ == "__main__":
    main()
