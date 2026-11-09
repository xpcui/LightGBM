
import numpy as np
import pandas as pd

from ledc import LEDC


def make_test_data():
    """Create a small synthetic training dataset."""

    rng = np.random.default_rng(42)

    X = pd.DataFrame(
        rng.poisson(5, size=(60, 20)),
        columns=[f"Gene_{i}" for i in range(20)]
    )

    Y = pd.DataFrame({
        "B cell": X["Gene_0"] + X["Gene_1"],
        "T cell": X["Gene_2"] + X["Gene_3"]
    })

    return X, Y


def test_model_training():
    """Check that LEDC can train and predict."""

    X, Y = make_test_data()

    model = LEDC(
        target="count",
        n_estimators=30,
        early_stopping_rounds=5
    )

    model.fit(X, Y)
    predictions = model.predict(X)

    assert predictions.shape == (60, 2)
    assert list(predictions.columns) == list(Y.columns)
    assert (predictions.to_numpy() >= 0).all()


def test_model_save_load(tmp_path):
    """Check that saving and loading preserves predictions."""

    X, Y = make_test_data()

    model = LEDC(
        target="count",
        n_estimators=30,
        early_stopping_rounds=5
    )

    model.fit(X, Y)

    original_predictions = model.predict(X)

    save_path = tmp_path / "saved_models"
    model.save(save_path)

    loaded_model = LEDC.load(save_path)
    loaded_predictions = loaded_model.predict(X)

    np.testing.assert_allclose(
        original_predictions.to_numpy(),
        loaded_predictions.to_numpy(),
        rtol=1e-6,
        atol=1e-8
    )


def test_gene_alignment():
    """Check that changing gene order does not change predictions."""

    X, Y = make_test_data()

    model = LEDC(
        target="count",
        n_estimators=30,
        early_stopping_rounds=5
    )

    model.fit(X, Y)

    original_predictions = model.predict(X)

    # Reverse the order of gene columns
    X_reordered = X[X.columns[::-1]]

    reordered_predictions = model.predict(X_reordered)

    np.testing.assert_allclose(
        original_predictions.to_numpy(),
        reordered_predictions.to_numpy(),
        rtol=1e-6,
        atol=1e-8
    )
