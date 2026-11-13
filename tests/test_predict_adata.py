
import numpy as np
import pandas as pd
import anndata as ad
import pytest

from ledc import LEDC


@pytest.fixture
def trained_model():
    # Small synthetic training dataset
    rng = np.random.default_rng(42)

    X = pd.DataFrame(
        rng.poisson(5, size=(30, 4)),
        columns=["g1", "g2", "g3", "g4"]
    )

    Y = pd.DataFrame(
        rng.uniform(size=(30, 2)),
        columns=["B cell", "T cell"]
    )

    model = LEDC(target="proportion")
    model.fit(X, Y)

    return model


def test_predict_adata_matches_predict(trained_model):
    X = pd.DataFrame(
        [[2, 3, 4, 5], [5, 4, 3, 2]],
        columns=["g1", "g2", "g3", "g4"],
        index=["spot1", "spot2"]
    )

    adata = ad.AnnData(
        X=X.to_numpy(),
        obs=pd.DataFrame(index=X.index),
        var=pd.DataFrame(index=X.columns)
    )

    expected = trained_model.predict(X)
    actual = trained_model.predict_adata(adata)

    np.testing.assert_allclose(actual, expected)


def test_predict_adata_gene_order(trained_model):
    adata = ad.AnnData(
        X=np.array([[4, 2, 5, 3]]),
        obs=pd.DataFrame(index=["spot1"]),
        var=pd.DataFrame(index=["g3", "g1", "g4", "g2"])
    )

    expected_X = pd.DataFrame(
        [[2, 3, 4, 5]],
        columns=["g1", "g2", "g3", "g4"],
        index=["spot1"]
    )

    actual = trained_model.predict_adata(adata)
    expected = trained_model.predict(expected_X)

    np.testing.assert_allclose(actual, expected)


def test_predict_adata_missing_genes(trained_model):
    adata = ad.AnnData(
        X=np.array([[2, 3, 4]]),
        obs=pd.DataFrame(index=["spot1"]),
        var=pd.DataFrame(index=["g1", "g2", "g3"])
    )

    with pytest.raises(ValueError):
        trained_model.predict_adata(
            adata,
            missing_genes="error"
        )


def test_predict_adata_sparse(trained_model):
    from scipy.sparse import csr_matrix

    adata = ad.AnnData(
        X=csr_matrix([[2, 3, 4, 5]]),
        obs=pd.DataFrame(index=["spot1"]),
        var=pd.DataFrame(index=["g1", "g2", "g3", "g4"])
    )

    predictions = trained_model.predict_adata(adata)

    assert predictions.shape == (1, 2)
    assert predictions.index.tolist() == ["spot1"]


def test_predict_adata_duplicate_genes(trained_model):
    adata = ad.AnnData(
        X=np.array([[2, 3, 4, 5]]),
        obs=pd.DataFrame(index=["spot1"]),
        var=pd.DataFrame(index=["g1", "g1", "g3", "g4"])
    )

    with pytest.raises(ValueError):
        trained_model.predict_adata(adata)

def test_predict_adata_zero_fill(trained_model):
    adata = ad.AnnData(
        X=np.array([[2, 3, 4]]),
        obs=pd.DataFrame(index=["spot1"]),
        var=pd.DataFrame(index=["g1", "g2", "g3"])
    )

    actual = trained_model.predict_adata(
        adata,
        missing_genes="zero"
    )

    expected_X = pd.DataFrame(
        [[2, 3, 4, 0]],
        columns=["g1", "g2", "g3", "g4"],
        index=["spot1"]
    )

    expected = trained_model.predict(expected_X)

    np.testing.assert_allclose(actual, expected)


def test_predict_adata_preserves_barcodes(trained_model):
    adata = ad.AnnData(
        X=np.array([[2, 3, 4, 5], [5, 4, 3, 2]]),
        obs=pd.DataFrame(index=["spot_B", "spot_A"]),
        var=pd.DataFrame(index=["g1", "g2", "g3", "g4"])
    )

    predictions = trained_model.predict_adata(
        adata,
        batch_size=1
    )

    assert predictions.index.tolist() == ["spot_B", "spot_A"]


def test_predict_adata_validate_scale(trained_model):
    trained_model.expression_scale = "raw_counts"

    adata = ad.AnnData(
        X=np.array([[1.5, 2.0, 3.0, 4.0]]),
        obs=pd.DataFrame(index=["spot1"]),
        var=pd.DataFrame(index=["g1", "g2", "g3", "g4"])
    )

    with pytest.warns(UserWarning, match="non-integer"):
        predictions = trained_model.predict_adata(
            adata,
            validate_scale=True
        )

    assert predictions.shape == (1, 2)


def test_predict_adata_validation_optional(trained_model):
    trained_model.expression_scale = "raw_counts"

    adata = ad.AnnData(
        X=np.array([[1.5, 2.0, 3.0, 4.0]]),
        obs=pd.DataFrame(index=["spot1"]),
        var=pd.DataFrame(index=["g1", "g2", "g3", "g4"])
    )

    # No preprocessing-scale warning when validation is disabled
    predictions = trained_model.predict_adata(
        adata,
        validate_scale=False
    )

    assert predictions.shape == (1, 2)
