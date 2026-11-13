
import numpy as np
import pandas as pd


def validate_expression(X):
    """
    Validate a gene-expression matrix.

    Parameters
    ----------
    X : pandas.DataFrame
        Expression matrix with samples as rows
        and genes as columns.

    Returns
    -------
    pandas.DataFrame
        Validated expression matrix.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")

    if X.empty:
        raise ValueError("Expression matrix cannot be empty")

    if not X.index.is_unique:
        raise ValueError("Sample identifiers must be unique")

    if not X.columns.is_unique:
        raise ValueError("Gene names must be unique")

    if X.columns.isna().any():
        raise ValueError("Gene names cannot be missing")

    if not all(pd.api.types.is_numeric_dtype(dt) for dt in X.dtypes):
        raise ValueError("Expression values must be numeric")

    values = X.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Expression matrix contains NaN or infinity")

    if (values < 0).any():
        raise ValueError("Expression values cannot be negative")

    return X


def align_genes(X, training_genes, missing_genes="error"):
    """
    Align expression data to the training gene order.

    Parameters
    ----------
    X : pandas.DataFrame
        New expression matrix (samples x genes).

    training_genes : list
        Gene names used during model training.

    missing_genes : {"error", "zero"}
        "error": raise an error for missing training genes.
        "zero": fill missing training genes with zeros.

    Returns
    -------
    pandas.DataFrame
        Expression matrix aligned to training genes.
    """

    validate_expression(X)

    training_genes = list(training_genes)

    if len(training_genes) == 0:
        raise ValueError("training_genes cannot be empty")

    if len(set(training_genes)) != len(training_genes):
        raise ValueError("training_genes contains duplicates")

    if missing_genes not in ("error", "zero"):
        raise ValueError(
            "missing_genes must be 'error' or 'zero'"
        )

    missing = [
        gene for gene in training_genes
        if gene not in X.columns
    ]

    if missing and missing_genes == "error":
        raise ValueError(
            f"{len(missing)} training genes are missing. "
            f"Examples: {missing[:5]}"
        )

    X_aligned = X.reindex(
        columns=training_genes,
        fill_value=0
    )

    return X_aligned


def log_normalize_counts(df, scale_factor=1e6):
    """
    Perform log1p CPM normalization on raw counts.

    Rows = samples; columns = genes.
    """

    validate_expression(df)

    if not np.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError("scale_factor must be positive")

    df = df.copy()

    libsize = df.sum(axis=1)

    # Avoid division by zero
    libsize = libsize.replace(0, np.nan)

    df_norm = df.div(libsize, axis=0) * scale_factor

    df_norm = np.log1p(df_norm).fillna(0)

    return df_norm


def matrix_summary(df, name):
    """
    Calculate summary statistics for the entire matrix.
    """

    validate_expression(df)

    values = df.to_numpy().ravel()

    return pd.DataFrame({
        "dataset": [name],
        "min": [np.nanmin(values)],
        "q25": [np.nanpercentile(values, 25)],
        "median": [np.nanmedian(values)],
        "mean": [np.nanmean(values)],
        "q75": [np.nanpercentile(values, 75)],
        "max": [np.nanmax(values)],
        "zero_fraction": [np.mean(values == 0)]
    })


def sample_summary(df, name="data"):
    """
    Calculate summary statistics for each sample.
    """

    validate_expression(df)

    libsize = df.sum(axis=1)

    detected_genes = (df > 0).sum(axis=1)

    summary = pd.DataFrame({
        "dataset": name,
        "library_size": libsize,
        "detected_genes": detected_genes,
        "mean_expression": df.mean(axis=1),
        "median_expression": df.median(axis=1),
        "zero_fraction": (df == 0).mean(axis=1)
    })

    return summary


def summarize_expression(X):
    """
    Summarize a gene-expression matrix (samples x genes).

    Supports NumPy arrays, pandas DataFrames, and SciPy sparse
    matrices without converting the full matrix to dense.
    """
    import numpy as np
    from scipy import sparse

    if sparse.issparse(X):
        X = X.tocsr(copy=True)
        X.eliminate_zeros()

        values = X.data
        row_sums = np.asarray(X.sum(axis=1)).ravel()
        detected = np.diff(X.indptr)
        zero_fraction = 1 - X.nnz / (X.shape[0] * X.shape[1])
    else:
        X = np.asarray(X)
        values = X.ravel()
        row_sums = X.sum(axis=1)
        detected = np.count_nonzero(X, axis=1)
        zero_fraction = np.mean(X == 0)

    if X.ndim != 2 or 0 in X.shape:
        raise ValueError("X must be a nonempty 2D matrix")

    if not np.isfinite(values).all():
        raise ValueError("Expression contains NaN or infinity")

    positive = values[values > 0]

    return {
        "n_samples": X.shape[0],
        "n_genes": X.shape[1],
        "median_total_counts": float(np.median(row_sums)),
        "median_detected_genes": float(np.median(detected)),
        "median_positive_expression": (
            float(np.median(positive))
            if positive.size else 0.0
        ),
        "zero_fraction": float(zero_fraction),
        "integer_fraction": (
            float(np.mean(np.isclose(values, np.round(values))))
            if values.size else 1.0
        ),
    }


import warnings
import numpy as np
from scipy import sparse


def validate_expression_scale(X, expected_scale="unknown"):
    """
    Validate expression values against a declared expression scale.

    This function reports potential preprocessing mismatches.
    It does not modify the expression matrix.
    """
    valid_scales = {"unknown", "raw_counts", "cpm", "log_cpm"}

    if expected_scale not in valid_scales:
        raise ValueError(f"expected_scale must be one of {valid_scales}")

    if sparse.issparse(X):
        values = X.data
    else:
        values = np.asarray(X).ravel()

    if not np.isfinite(values).all():
        raise ValueError("Expression contains NaN or infinity")

    if (values < 0).any():
        raise ValueError("Expression values must be nonnegative")

    if expected_scale == "raw_counts":
        if not np.allclose(values, np.round(values)):
            warnings.warn(
                "Input contains non-integer expression values, "
                "but the model expects raw counts.",
                UserWarning
            )

    if expected_scale in ("cpm", "log_cpm"):
        warnings.warn(
            f"Expected expression scale is '{expected_scale}'. "
            "This function cannot confirm normalization from "
            "values alone. Check that input preprocessing "
            "matches the training data.",
            UserWarning
        )

    return True
