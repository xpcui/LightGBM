
"""Pseudobulk dataset generation workflows for LEDC."""

from pathlib import Path

import numpy as np
import pandas as pd

from .pseudobulk import generate_pseudobulk, save_T_csv


def generate_fold_pseudobulk(
    train_adata,
    test_adata,
    output_dir,
    fold=1,
    mode="count",
    n_samples=5000,
    pb_cells=10,
    poisson_mean=10,
    alpha=1.0,
    seed=None,
    cell_types=None,
):
    """
    Generate train/test pseudobulk datasets for one fold.

    The reference AnnData objects must already be split
    into training and testing biological units.

    Returns
    -------
    dict
        Paths to the six saved output files.
    """

    if fold not in (1, 2, 3):
        raise ValueError("fold must be 1, 2, or 3")

    if mode not in ("count", "proportion"):
        raise ValueError(
            "mode must be 'count' or 'proportion'"
        )

    # Preserve matching genes and their order.
    if not train_adata.var_names.equals(test_adata.var_names):
        raise ValueError(
            "Training and testing references must have "
            "identical genes in the same order"
        )

    gene_names = train_adata.var_names.astype(str).tolist()

    if len(set(gene_names)) != len(gene_names):
        raise ValueError("Gene names must be unique")

    # Derive the cell types from the training reference.
    if cell_types is None:
        cell_types = sorted(
            train_adata.obs["cell_type"].astype(str).unique()
    )
    else:
        cell_types = list(cell_types)

    # Both references must contain the requested cell types.
    test_types = set(
        test_adata.obs["cell_type"].astype(str)
    )

    if set(cell_types) != test_types:
        raise ValueError(
            "Training and testing cell types do not match"
        )

    # Match the notebook's random seeds.
    if seed is None:
        seed = (3025 if mode == "count" else 2025) + fold

    # One RNG, shared sequentially by train and test.
    rng = np.random.default_rng(seed)

    T_train, C_train, P_train = generate_pseudobulk(
        adata=train_adata,
        n_samples=n_samples,
        cell_types=cell_types,
        rng=rng,
        mode=mode,
        pb_cells=pb_cells,
        poisson_mean=poisson_mean,
        alpha=alpha,
    )

    T_test, C_test, P_test = generate_pseudobulk(
        adata=test_adata,
        n_samples=n_samples,
        cell_types=cell_types,
        rng=rng,
        mode=mode,
        pb_cells=pb_cells,
        poisson_mean=poisson_mean,
        alpha=alpha,
    )

    # Match the notebook's directory layout.
    output_dir = Path(output_dir) / mode / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    for split, T, C, P in [
        ("train", T_train, C_train, P_train),
        ("test", T_test, C_test, P_test),
    ]:
        prefix = f"fold_{fold}"

        paths[f"T_{split}"] = (
            output_dir / f"{prefix}_T_{split}.csv"
        )
        paths[f"C_{split}"] = (
            output_dir / f"{prefix}_C_{split}.csv"
        )
        paths[f"P_{split}"] = (
            output_dir / f"{prefix}_P_{split}.csv"
        )

        save_T_csv(T, paths[f"T_{split}"], gene_names)
        C.to_csv(paths[f"C_{split}"], index=False)
        P.to_csv(paths[f"P_{split}"], index=False)

    return paths


def generate_full_pseudobulk(
    reference_adata,
    output_dir,
    mode="count",
    n_samples=5000,
    pb_cells=10,
    poisson_mean=10,
    alpha=1.0,
    seed=None,
    cell_types=None
):
    """Generate a full-reference pseudobulk training dataset."""

    if mode not in ("count", "proportion"):
        raise ValueError(
            "mode must be 'count' or 'proportion'"
        )

    if seed is None:
        seed = 3029 if mode == "count" else 2029

    if cell_types is None:
        cell_types = sorted(
            reference_adata.obs["cell_type"].astype(str).unique()
        )
    else:
        cell_types = list(cell_types)

    gene_names = reference_adata.var_names.astype(str).tolist()

    if len(set(gene_names)) != len(gene_names):
        raise ValueError("Gene names must be unique")

    T, C, P = generate_pseudobulk(
        adata=reference_adata,
        n_samples=n_samples,
        cell_types=cell_types,
        rng=np.random.default_rng(seed),
        mode=mode,
        pb_cells=pb_cells,
        poisson_mean=poisson_mean,
        alpha=alpha,
    )

    output_dir = Path(output_dir) / "full_train" / mode
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "T": output_dir / "full_T_train.csv",
        "C": output_dir / "full_C_train.csv",
        "P": output_dir / "full_P_train.csv",
    }

    save_T_csv(T, paths["T"], gene_names)
    C.to_csv(paths["C"], index=False)
    P.to_csv(paths["P"], index=False)

    return paths
