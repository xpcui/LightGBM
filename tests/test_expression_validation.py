
import numpy as np
import pytest
from scipy import sparse

from ledc.preprocessing import validate_expression_scale


def test_valid_raw_counts():
    X = np.array([[0, 2, 3], [1, 0, 5]])
    assert validate_expression_scale(X, "raw_counts")


def test_noninteger_raw_counts_warning():
    X = np.array([[0.5, 2.3], [1.0, 0.0]])

    with pytest.warns(UserWarning, match="non-integer"):
        validate_expression_scale(X, "raw_counts")


def test_negative_expression():
    X = np.array([[1, -2]])

    with pytest.raises(ValueError, match="nonnegative"):
        validate_expression_scale(X, "raw_counts")


def test_nonfinite_expression():
    X = np.array([[1, np.nan]])

    with pytest.raises(ValueError, match="NaN or infinity"):
        validate_expression_scale(X)


def test_sparse_expression():
    X = sparse.csr_matrix([[0, 2, 3], [1, 0, 5]])
    assert validate_expression_scale(X, "raw_counts")


def test_unknown_scale():
    X = np.array([[0.5, 2.3]])
    assert validate_expression_scale(X, "unknown")
