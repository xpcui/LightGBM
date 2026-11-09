# LEDC: LightGBM Ensemble DeConvolution

**LEDC** is a supervised machine-learning method for estimating cell-type abundance from bulk RNA sequencing and spatial transcriptomics data.

LEDC trains independent LightGBM regression models for each cell type using pseudobulk mixtures generated from annotated single-cell RNA-seq data. It supports both cell-type count and proportion prediction, with an emphasis on count-based deconvolution for spatial transcriptomics.

## Installation

Clone the repository and install LEDC:

```bash
git clone https://github.com/YOUR_USERNAME/LEDC.git
cd LEDC
python -m pip install -e .
```

Replace `YOUR_USERNAME` with the actual GitHub account name after publication.

## Main features

- Generate pseudobulk training mixtures from annotated single-cell RNA-seq data.
- Predict cell-type counts or proportions using independent LightGBM regression models.
- Select boosting rounds using cross-validation with early stopping.
- Save and load trained models for reuse.
- Evaluate predictions using Pearson correlation and RMSE.
- Visualize predicted cell-type abundance in spatial transcriptomics data.

## Quick start

### 1. Generate pseudobulk training data

The input is an AnnData object containing raw single-cell counts and cell-type annotations stored in `adata.obs["cell_type"]`.

```python
import numpy as np
import pandas as pd
from ledc import generate_pseudobulk

T, C, P = generate_pseudobulk(
    adata=adata,
    n_samples=500,
    cell_types=cell_types,
    rng=np.random.default_rng(42),
    mode="count",
    poisson_mean=10
)

X_train = pd.DataFrame(
    T.toarray(),
    columns=adata.var_names
)

Y_train = C
```

The function returns:

- `T`: pseudobulk gene expression (samples × genes)
- `C`: cell-type counts (samples × cell types)
- `P`: cell-type proportions (