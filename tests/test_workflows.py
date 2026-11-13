
import numpy as np
import pandas as pd
import anndata as ad

from ledc import (
    generate_fold_pseudobulk,
    generate_full_pseudobulk,
)


def make_reference(seed):
    rng = np.random.default_rng(seed)

    X = rng.poisson(5, size=(60, 20))

    adata = ad.AnnData(X)

    adata.obs["cell_type"] = (
        ["B cell"] * 30 + ["T cell"] * 30
    )

    adata.var_names = [
        f"Gene_{i}" for i in range(20)
    ]

    return adata


def test_fold_generation(tmp_path):
    train = make_reference(42)
    test = make_reference(123)

    paths = generate_fold_pseudobulk(
        train_adata=train,
        test_adata=test,
        output_dir=tmp_path,
        fold=1,
        mode="count",
        n_samples=10,
    )

    for path in paths.values():
        assert path.exists()

    T = pd.read_csv(paths["T_train"])
    C = pd.read_csv(paths["C_train"])
    P = pd.read_csv(paths["P_train"])

    assert T.shape == (10, 20)
    assert C.shape == (10, 2)
    assert P.shape == (10, 2)

    # In count mode, P contains cell counts
    pd.testing.assert_frame_equal(P, C)

    # Each sample contains at least one cell
    assert (P.sum(axis=1) >= 1).all()


def test_full_generation(tmp_path):
    reference = make_reference(42)

    paths = generate_full_pseudobulk(
        reference_adata=reference,
        output_dir=tmp_path,
        mode="count",
        n_samples=10,
    )

    for path in paths.values():
        assert path.exists()

    C = pd.read_csv(paths["C"])

    assert C.shape == (10, 2)
    assert (C.sum(axis=1) >= 1).all()


def test_reproducibility(tmp_path):
    train = make_reference(42)
    test = make_reference(123)

    paths1 = generate_fold_pseudobulk(
        train_adata=train,
        test_adata=test,
        output_dir=tmp_path / "run1",
        fold=1,
        mode="count",
        n_samples=10,
        seed=3026,
    )

    paths2 = generate_fold_pseudobulk(
        train_adata=train,
        test_adata=test,
        output_dir=tmp_path / "run2",
        fold=1,
        mode="count",
        n_samples=10,
        seed=3026,
    )

    for key in paths1:
        data1 = pd.read_csv(paths1[key])
        data2 = pd.read_csv(paths2[key])

        pd.testing.assert_frame_equal(data1, data2)
