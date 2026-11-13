import numpy as np
from scipy import sparse

from ledc.preprocessing import summarize_expression


def test_expression_summary_dense_and_sparse():
    X = np.array([
        [0, 2, 3],
        [0, 0, 5],
    ])

    dense = summarize_expression(X)
    sparse_result = summarize_expression(sparse.csr_matrix(X))

    assert dense == sparse_result
    assert dense["n_samples"] == 2
    assert dense["n_genes"] == 3
    assert dense["zero_fraction"] == 0.5
    assert dense["median_total_counts"] == 5.0