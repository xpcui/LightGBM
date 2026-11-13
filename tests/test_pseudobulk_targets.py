
import numpy as np
import pandas as pd
import anndata as ad

from scipy import sparse
from ledc import generate_pseudobulk


def make_adata():
    X = sparse.csr_matrix([
        [2, 0, 1],
        [1, 3, 0],
        [0, 2, 4],
        [3, 0, 2],
        [1, 1, 1],
        [0, 4, 2],
    ])

    obs = pd.DataFrame(
        {
            "cell_type": [
                "B cell", "B cell",
                "T cell", "T cell",
                "Macrophage", "Macrophage"
            ]
        },
        index=[f"cell_{i}" for i in range(6)]
    )

    return ad.AnnData(X=X, obs=obs)


def test_count_mode_p_equals_c():
    adata = make_adata()

    T, C, P = generate_pseudobulk(
        adata=adata,
        n_samples=50,
        cell_types=[
            "B cell", "T cell", "Macrophage"
        ],
        rng=np.random.default_rng(42),
        mode="count",
        poisson_mean=10
    )

    # P contains absolute counts
    pd.testing.assert_frame_equal(P, C)

    # Both matrices contain nonnegative integers
    assert (P.to_numpy() >= 0).all()
    assert np.equal(
        P.to_numpy(),
        np.floor(P.to_numpy())
    ).all()

    assert T.shape == (50, 3)


def test_proportion_mode_p_is_normalized_c():
    adata = make_adata()

    T, C, P = generate_pseudobulk(
        adata=adata,
        n_samples=50,
        cell_types=[
            "B cell", "T cell", "Macrophage"
        ],
        rng=np.random.default_rng(42),
        mode="proportion",
        pb_cells=10
    )

    expected = C.div(
        C.sum(axis=1),
        axis=0
    )

    np.testing.assert_allclose(
        P.to_numpy(),
        expected.to_numpy()
    )

    # Each sample has exactly 10 cells
    assert (C.sum(axis=1) == 10).all()

    # Each P row sums to 1
    np.testing.assert_allclose(
        P.sum(axis=1).to_numpy(),
        np.ones(50)
    )


def test_default_mode_is_count():
    adata = make_adata()

    T, C, P = generate_pseudobulk(
        adata=adata,
        n_samples=20,
        cell_types=[
            "B cell", "T cell", "Macrophage"
        ],
        rng=np.random.default_rng(42)
    )

    pd.testing.assert_frame_equal(P, C)
