
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def safe_pearson(x, y):
    """Calculate Pearson correlation, handling constant vectors."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return pearsonr(x, y).statistic


def evaluate_predictions(y_true, y_pred):
    """
    Evaluate cell-type abundance predictions.

    Parameters
    ----------
    y_true : pandas.DataFrame
        True cell counts or proportions.
        Rows = samples, columns = cell types.

    y_pred : pandas.DataFrame
        Predicted cell counts or proportions.
        Must contain the same samples and cell types.

    Returns
    -------
    pandas.DataFrame
        Pearson correlation and RMSE by cell type,
        plus mean metrics across samples.
    """

    if not isinstance(y_true, pd.DataFrame):
        raise TypeError("y_true must be a pandas DataFrame")

    if not isinstance(y_pred, pd.DataFrame):
        raise TypeError("y_pred must be a pandas DataFrame")

    if not y_true.index.is_unique or not y_pred.index.is_unique:
        raise ValueError("Sample indices must be unique")

    if not y_true.columns.is_unique or not y_pred.columns.is_unique:
        raise ValueError("Cell-type names must be unique")

    if set(y_true.index) != set(y_pred.index):
        raise ValueError("Sample indices do not match")

    if set(y_true.columns) != set(y_pred.columns):
        raise ValueError("Cell types do not match")

    # Align rows and columns
    y_pred = y_pred.reindex(
        index=y_true.index,
        columns=y_true.columns
    )

    true = y_true.to_numpy(dtype=float)
    pred = y_pred.to_numpy(dtype=float)

    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Inputs must contain only finite values")

    results = []

    # Evaluation by cell type
    for j, cell_type in enumerate(y_true.columns):

        pearson = safe_pearson(
            true[:, j],
            pred[:, j]
        )

        rmse = np.sqrt(
            np.mean((true[:, j] - pred[:, j]) ** 2)
        )

        results.append({
            "cell_type": cell_type,
            "Pearson": pearson,
            "RMSE": rmse
        })

    # Evaluation by sample
    sample_pearson = []
    sample_rmse = []

    for i in range(true.shape[0]):

        sample_pearson.append(
            safe_pearson(true[i, :], pred[i, :])
        )

        sample_rmse.append(
            np.sqrt(np.mean((true[i, :] - pred[i, :]) ** 2))
        )

    # Mean sample-level scores
    results.append({
        "cell_type": "-- By sample --",
        "Pearson": (
            np.nanmean(sample_pearson)
            if np.isfinite(sample_pearson).any()
            else np.nan
        ),
        "RMSE": np.mean(sample_rmse)
    })

    return pd.DataFrame(results).set_index("cell_type")
