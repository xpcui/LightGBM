
import numpy as np
import pandas as pd
import pytest

from ledc import evaluate_predictions


def test_perfect_predictions():

    true = pd.DataFrame({
        "B cell": [2, 5, 8],
        "T cell": [8, 5, 2]
    })

    pred = true.copy()

    results = evaluate_predictions(true, pred)

    assert np.allclose(results["RMSE"], 0)

    assert np.allclose(
        results["Pearson"].iloc[:2],
        1
    )

    assert results.loc["-- By sample --", "Pearson"] == pytest.approx(
        1.0
    )


def test_different_column_order():

    true = pd.DataFrame({
        "B cell": [2, 5, 8],
        "T cell": [8, 5, 2]
    })

    pred = true[["T cell", "B cell"]]

    results = evaluate_predictions(true, pred)

    assert np.allclose(results["RMSE"], 0)


def test_mismatched_cell_types():

    true = pd.DataFrame({
        "B cell": [1, 2, 3]
    })

    pred = pd.DataFrame({
        "T cell": [1, 2, 3]
    })

    with pytest.raises(ValueError):
        evaluate_predictions(true, pred)
