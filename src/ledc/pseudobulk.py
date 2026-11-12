
import os
import numpy as np
import pandas as pd
from scipy import sparse as sp


def generate_pseudobulk(
    adata,
    n_samples,
    cell_types,
    rng,
    mode="count",
    pb_cells=10,
    poisson_mean=10,
    alpha=1.0
):
    """
    Generate pseudobulk mixtures from annotated single-cell data.

    Parameters
    ----------
    adata : anndata.AnnData
        Single-cell expression data containing raw counts.
        Rows are cells and columns are genes.
        Cell-type labels must be stored in adata.obs["cell_type"].

    n_samples : int
        Number of pseudobulk samples to generate.

    cell_types : list of str
        Cell types included in pseudobulk mixtures.

    rng : numpy.random.Generator
        Random number generator for reproducibility.

    mode : {"proportion", "count"}
        "proportion": fixed number of cells per pseudobulk sample.
        "count": Poisson-distributed number of cells per sample,
        with a minimum of one cell.

    pb_cells : int, default=10
        Number of cells per sample in proportion mode.

    poisson_mean : float, default=10
        Mean of the Poisson distribution in count mode.

    alpha : float, default=1.0
        Symmetric Dirichlet concentration parameter.

    Returns
    -------
    T : scipy.sparse.csr_matrix
        Pseudobulk gene expression matrix (samples x genes).

    C : pandas.DataFrame
        True cell counts (samples x cell types).

    P : pandas.DataFrame
        True cell proportions (samples x cell types).
    """

    # --------------------------------------------------
    # 1. Validate inputs
    # --------------------------------------------------

    if not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")

    cell_types = list(cell_types)

    if len(cell_types) == 0:
        raise ValueError("cell_types cannot be empty")

    if len(set(cell_types)) != len(cell_types):
        raise ValueError("cell_types must not contain duplicates")

    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be positive and finite")

    if mode not in ("proportion", "count"):
        raise ValueError(
            "mode must be 'proportion' or 'count'"
        )

    if mode == "proportion":
        if not isinstance(pb_cells, (int, np.integer)) or pb_cells < 1:
            raise ValueError("pb_cells must be a positive integer")

    if mode == "count":
        if not np.isfinite(poisson_mean) or poisson_mean <= 0:
            raise ValueError(
                "poisson_mean must be positive and finite"
            )

    if "cell_type" not in adata.obs.columns:
        raise ValueError(
            'adata.obs must contain a "cell_type" column'
        )

    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            "rng must be a numpy.random.Generator"
        )

    # --------------------------------------------------
    # 2. Pool cell indices by cell type
    # --------------------------------------------------

    labels = adata.obs["cell_type"].astype(str).values

    cell_pools = {
        ct: np.where(labels == ct)[0]
        for ct in cell_types
    }

    # Ensure all requested cell types are available
    for ct, pool in cell_pools.items():
        if len(pool) == 0:
            raise ValueError(
                f"No cells found for cell type: {ct}"
            )

    K = len(cell_types)

    T_rows = []
    C_rows = []
    P_rows = []

    # --------------------------------------------------
    # 3. Generate pseudobulk samples
    # --------------------------------------------------

    for i in range(n_samples):

        # Generate random cell-type mixture probabilities
        p = rng.dirichlet(
            np.repeat(alpha, K)
        )

        # Determine total number of cells in the sample
        if mode == "proportion":
            n_total = pb_cells

        else:
            n_total = max(
                1,
                rng.poisson(poisson_mean)
            )

        # Generate cell counts for each cell type
        counts = rng.multinomial(
            n_total,
            p
        )

        # --------------------------------------------------
        # 4. Sample individual cells
        # --------------------------------------------------

        sampled_indices = []

        for ct, n_ct in zip(cell_types, counts):

            if n_ct == 0:
                continue

            pool = cell_pools[ct]

            sampled = rng.choice(
                pool,
                size=n_ct,
                replace=True
            )

            sampled_indices.extend(
                sampled.tolist()
            )

        # --------------------------------------------------
        # 5. Aggregate raw gene counts
        # --------------------------------------------------

        X_sample = adata.X[
            sampled_indices,
            :
        ]

        if not sp.issparse(X_sample):
            X_sample = sp.csr_matrix(
                X_sample
            )

        bulk_row = sp.csr_matrix(
            X_sample.sum(axis=0)
        )

        # Store gene expression
        T_rows.append(bulk_row)

        # Store true cell counts
        C_rows.append(counts)

        # Store true cell proportions
        P_rows.append(
            counts / counts.sum()
        )

    # --------------------------------------------------
    # 6. Combine generated samples
    # --------------------------------------------------

    T = sp.vstack(
        T_rows,
        format="csr"
    )

    C = pd.DataFrame(
        C_rows,
        columns=cell_types
    )

    P = pd.DataFrame(
        P_rows,
        columns=cell_types
    )

    return T, C, P


def save_T_csv(
    T,
    path,
    gene_names,
    chunk_size=100
):
    """
    Save a pseudobulk expression matrix to CSV in chunks.

    Parameters
    ----------
    T : scipy.sparse matrix or numpy.ndarray
        Pseudobulk expression matrix (samples x genes).

    path : str or pathlib.Path
        Output CSV file path.

    gene_names : list of str
        Gene names corresponding to columns of T.

    chunk_size : int, default=100
        Number of rows written at a time.

    Returns
    -------
    None
    """

    # --------------------------------------------------
    # 1. Validate inputs
    # --------------------------------------------------

    if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError(
            "chunk_size must be a positive integer"
        )

    if len(gene_names) != T.shape[1]:
        raise ValueError(
            "Number of gene names must match T columns"
        )

    if T.shape[0] == 0:
        raise ValueError("T must contain at least one sample")

    # --------------------------------------------------
    # 2. Prepare output directory
    # --------------------------------------------------

    path = os.fspath(path)

    output_dir = os.path.dirname(
        os.path.abspath(path)
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # Remove an old partial file if it exists
    if os.path.exists(path):
        os.remove(path)

    # --------------------------------------------------
    # 3. Save matrix in chunks
    # --------------------------------------------------

    n_rows = T.shape[0]

    for start in range(
        0,
        n_rows,
        chunk_size
    ):

        end = min(
            start + chunk_size,
            n_rows
        )

        chunk = T[
            start:end,
            :
        ]

        if sp.issparse(chunk):
            chunk = chunk.toarray()

        chunk_df = pd.DataFrame(
            chunk,
            columns=gene_names
        )

        chunk_df.to_csv(
            path,
            mode="w" if start == 0 else "a",
            header=(start == 0),
            index=False
        )

        print(
            f"Saved rows {start} to {end}"
        )
