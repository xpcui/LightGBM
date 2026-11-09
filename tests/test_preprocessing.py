
import numpy as np
import pandas as pd
import pytest

from ledc import (
    validate_expression,
    align_genes,
    log_normalize_counts,
    matrix_summary,
    sample_summary,
)


def make_test_data():
    return pd.DataFrame(
        {
            "Gene_A": [10, 20, 30],
            "Gene_B": [5, 10, 15],
            "Gene_C": [2, 4, 6]
        },
        index=["Sample_1", "Sample_2", "Sample_3"]
    )


def test_validate_expression():
    X = make_test_data()

    result = validate_expression(X)

    pd.testing.assert_frame_equal(X, result)


def test_gene_alignment():
    X = make_test_data()

    training_genes = [
        "Gene_C",
        "Gene_A",
        "Gene_B"
    ]

    result = align_genes(X, training_genes)

    assert list(result.columns) == training_genes

    assert result.loc["Sample_1", "Gene_C"] == 2


def test_missing_genes_error():
    X = make_test_data()

    with pytest.raises(ValueError):
        align_genes(
            X,
            ["Gene_A", "Gene_D"],
            missing_genes="error"
        )


def test_missing_genes_zero():
    X = make_test_data()

    result = align_genes(
        X,
        ["Gene_A", "Gene_D"],
        missing_genes="zero"
    )

    assert (result["Gene_D"] == 0).all()


def test_log_normalization():
    X = make_test_data()

    result = log_normalize_counts(X)

    assert result.shape == X.shape
    assert np.isfinite(result.to_numpy()).all()
    assert (result.to_numpy() >= 0).all()


def test_matrix_summary():
    X = make_test_data()

    result = matrix_summary(X, "test")

    assert result.shape[0] == 1
    assert result.loc[0, "dataset"] == "test"


def test_sample_summary():
    X = make_test_data()

    result = sample_summary(X)

    assert result.shape[0] == 3
    assert "library_size" in result.columns


def test_negative_expression():
    X = make_test_data()
    X.loc["Sample_1", "Gene_A"] = -1

    with pytest.raises(ValueError):
        validate_expression(X)
