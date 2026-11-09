# LEDC: LightGBM Ensemble DeConvolution

**LEDC** is a supervised machine-learning method for estimating cell-type abundance from bulk RNA sequencing and spatial transcriptomics data.

LEDC trains independent LightGBM regression models for each cell type using pseudobulk mixtures generated from annotated single-cell RNA-seq data. It supports both cell-type count and proportion prediction, with an emphasis on count-based deconvolution for spatial transcriptomics.

## Installation

Clone the repository and install LEDC:

```bash
git clone https://github.com/yijiaxue1202/LEDC
cd LEDC
python -m pip install -e .
```

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
- `P`: cell-type proportions (samples × cell types)

The DataFrame conversion shown above is suitable for small datasets. Large datasets may require a memory-efficient sparse workflow.

### 2. Train LEDC

```python
from ledc import LEDC

model = LEDC(target="count")
model.fit(X_train, Y_train)
```

LEDC trains a separate LightGBM regression model for each cell type.

### 3. Predict cell-type abundance

```python
predictions = model.predict(X_test)
print(predictions.head())
```

`X_test` must contain compatible expression measurements and the training genes. LEDC aligns gene columns to the training order.

### 4. Save and load trained models

```python
model.save("saved_models")

loaded_model = LEDC.load("saved_models")
predictions = loaded_model.predict(X_test)
```

### 5. Evaluate predictions

```python
from ledc import evaluate_predictions

scores = evaluate_predictions(Y_test, predictions)
print(scores)
```

Evaluation includes Pearson correlation and RMSE by cell type and mean sample-level metrics. Count predictions are evaluated without proportion normalization.

### 6. Visualize spatial predictions

```python
from ledc import plot_spatial

fig = plot_spatial(
    predictions=predictions,
    positions=spatial_positions,
    cell_types=["B cell", "T cell"],
    shared_scale=True,
    save_path="spatial_maps.png"
)
```

The spatial positions DataFrame must contain `pxl_col_in_fullres` and `pxl_row_in_fullres`. The optional `in_tissue` column identifies tissue spots. Spot indices should match prediction indices.

## Input data requirements

LEDC expects gene-expression matrices with samples or spatial spots as rows and genes as columns. Training targets contain cell-type counts or proportions, with one column per cell type.

The training and prediction datasets should use compatible gene identifiers and expression preprocessing. Gene alignment alone does not resolve differences between experimental platforms or tissue distributions.

## Reproducibility

LEDC supports configurable random seeds, cross-validation, and model saving. For donor-aware evaluation, separate donors into training and testing datasets before pseudobulk generation or model training.

## Development

To install development dependencies and run the tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

To build distribution files:

```bash
python -m build
```

## Citation

A manuscript describing LEDC is in preparation. Citation details will be added when available.

## License

License information will be finalized before public release.
