# LEDC: LightGBM Ensembled DeConvolution

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

## Input and Output

LEDC is a supervised deconvolution package for estimating cell-type abundance from bulk RNA-seq or spatial transcriptomics data. The workflow consists of three main steps: generating pseudobulk training data, training cell-type-specific LightGBM models, and predicting cell-type abundance in new samples.

### Workflow overview

| Step | User input | LEDC output |
|---|---|---|
| 1. Pseudobulk generation | Annotated scRNA-seq reference data | Synthetic expression matrix (`T`), cell-count matrix (`C`), and prediction targets (`P`) |
| 2. Model training | Gene-expression matrix (`X_train`) and known cell-type targets (`Y_train`) | Trained LightGBM regression models, one per cell type |
| 3. Prediction | Gene-expression matrix from new bulk samples or spatial spots (`X_new`) | Predicted cell-type counts or proportions for each sample |

Users who already have labeled training data can skip pseudobulk generation and train LEDC directly using their own expression and cell-type abundance matrices.

Once a model has been trained, it can be applied to new datasets without requiring the original single-cell reference again, provided that the gene identifiers and expression preprocessing are compatible.

### 1. Pseudobulk generation

**Input**

LEDC's `generate_pseudobulk()` function accepts an annotated single-cell RNA-seq dataset stored as an `AnnData` object.

The input should contain:

- `adata.X`: Single-cell gene-expression matrix, with cells as rows and genes as columns. Raw counts are recommended for count-based pseudobulk generation.
- `adata.obs["cell_type"]`: Cell-type annotation for each cell.
- `adata.var_names`: Gene identifiers, such as Ensembl IDs or gene symbols.
- `cell_types`: List of cell types to include in the simulated mixtures.

Users also specify the number of pseudobulk samples, the prediction mode, and the random-number generator.

**Output**

The function returns three matrices:

| Matrix | Dimensions | Description |
|---|---|---|
| `T` | Samples × Genes | Synthetic pseudobulk gene expression, obtained by summing the expression of sampled cells |
| `C` | Samples × Cell types | Actual number of sampled cells from each cell type |
| `P` | Samples × Cell types | Prediction targets determined by the selected generation mode |

The expression matrix `T` is returned as a SciPy sparse matrix, while `C` and `P` are pandas DataFrames.

Two prediction modes are supported:

- **Count mode (`mode="count"`):** The total number of cells is sampled from a Poisson distribution, and `P = C`.
- **Proportion mode (`mode="proportion"`):** Each mixture contains a fixed number of cells, and `P` contains realized cell-type proportions, calculated as `C / row_sum(C)`.

### 2. LEDC model training

**Input**

Training requires two aligned pandas DataFrames:

- `X_train`: Gene-expression matrix with training samples as rows and genes as columns.
- `Y_train`: Ground-truth cell-type counts or proportions, with the same samples as rows and cell types as columns.

The rows of `X_train` and `Y_train` must correspond to the same samples.

When training on LEDC-generated pseudobulk data, the expression matrix `T` is used as `X_train`, and `P` is used as `Y_train`.

The prediction target is selected using:

- `LEDC(target="count")` for cell-type count prediction.
- `LEDC(target="proportion")` for cell-type proportion prediction.

**Output**

LEDC fits one LightGBM regression model per cell type. Each model learns the relationship between mixed gene expression and the abundance of a particular cell type.

The trained models can be saved to disk and subsequently loaded for prediction without retraining.

### 3. Cell-type abundance prediction

**Input**

For prediction, users provide a gene-expression matrix from new bulk RNA-seq samples or spatial transcriptomics spots.

The input can be:

- A pandas DataFrame (`X_new`), with samples or spots as rows and genes as columns.
- An AnnData object, using the `predict_adata()` method.

Gene identifiers should be consistent with those used for model training. LEDC aligns input genes with the training features. Missing genes can either raise an error or be explicitly filled with zeros.

The new expression data should use preprocessing compatible with the training data.

**Output**

LEDC returns a pandas DataFrame containing predicted cell-type abundances.

| Sample | B cell | T cell | Macrophage |
|---|---:|---:|---:|
| Sample 1 | 2.4 | 4.8 | 1.2 |
| Sample 2 | 1.1 | 3.6 | 2.3 |
| Sample 3 | 3.2 | 2.1 | 0.8 |

*Illustrative count-mode predictions; these are not benchmark results.*

Each row corresponds to a sample or spatial spot, and each column corresponds to a cell type.

For count-based models, the values represent predicted cell counts. Because LEDC uses regression, predicted counts are continuous estimates and are not necessarily integers.

For proportion-based models, the values represent predicted cell-type proportions. Optional normalization can be applied when predictions must sum to one.

Predictions can be exported as CSV files for downstream analysis or visualized as spatial cell-type abundance maps.

### Important notes

- A single-cell reference is required for pseudobulk generation, but not for prediction with an already trained model.
- Ground-truth cell-type labels are required for supervised training, but not for prediction on new samples.
- LEDC supports both count and proportion prediction; users should select the target that matches their application.
- Donor-disjoint training and testing require separate reference datasets or donor-level filtering before pseudobulk generation.
- LEDC does not automatically normalize gene expression to match the training data. Users should ensure compatible input preprocessing.

### Quick Example: Using LEDC with Your Own Data

```python
import anndata as ad
import numpy as np
import pandas as pd
from ledc import LEDC, generate_pseudobulk

# 1. Input: annotated single-cell reference
adata = ad.read_h5ad("reference.h5ad")
cell_types = sorted(adata.obs["cell_type"].unique())

# 2. Generate pseudobulk training data
T, C, P = generate_pseudobulk(
    adata=adata,
    n_samples=500,
    cell_types=cell_types,
    rng=np.random.default_rng(42),
    mode="count",
    poisson_mean=10
)

# Convert T to a DataFrame (suitable for small examples)
X_train = pd.DataFrame(
    T.toarray(),
    columns=adata.var_names
)

# 3. Train LEDC
model = LEDC(target="count", expression_scale="raw_counts")
model.fit(X_train, P)

# 4. Save the trained model
model.save("ledc_count_model")

# 5. Input: expression data from new samples
# Rows = samples, columns = genes
X_new = pd.read_csv("new_expression.csv", index_col=0)

# 6. Output: predicted cell-type counts
predictions = model.predict(X_new)

predictions.to_csv("predicted_cell_counts.csv")

```

## Example: Running LEDC on Synthetic Data

A reproducible example is available in [`examples/demo_ledc.py`](examples/demo_ledc.py). The demo uses simulated single-cell RNA-seq data containing three cell types (B cells, T cells, and macrophages) to demonstrate the complete LEDC workflow.

The example generates pseudobulk training data, fits LightGBM models for cell-type count prediction, and evaluates predictions on synthetic test samples.

To run the example after installing LEDC:

```bash
python examples/demo_ledc.py
```

See the example script for the complete implementation.

### Run the demo

After installing LEDC, run the following command from the repository root:

```bash
python examples/demo_ledc.py
```

### Example code

```python
from ledc import LEDC, generate_pseudobulk, evaluate_predictions
import numpy as np
import pandas as pd

# adata: simulated AnnData containing raw counts and cell-type labels
# cell_types: ["B cell", "T cell", "Macrophage"]

T_train, C_train, P_train = generate_pseudobulk(
    adata,
    n_samples=200,
    cell_types=cell_types,
    rng=np.random.default_rng(42),
    mode="count"
)

X_train = pd.DataFrame(T_train.toarray(), columns=adata.var_names)

# Train LEDC
model = LEDC(target="count", n_estimators=100)
model.fit(X_train, P_train)

# Predict on a separately generated test mixture dataset
predictions = model.predict(X_test)

# Evaluate predictions
scores = evaluate_predictions(C_test, predictions)
print(scores.round(3))
```

The complete script includes the creation of `adata`, `X_test`, and `C_test`.

### Example output

**Predicted cell-type counts (first five samples):**

| Sample | B cell | T cell | Macrophage |
|---|---:|---:|---:|
| 0 | 2.94 | 3.53 | 0.41 |
| 1 | 4.14 | 6.58 | 1.13 |
| 2 | 4.74 | 0.45 | 9.35 |
| 3 | 0.03 | 9.65 | 0.25 |
| 4 | 1.93 | 2.66 | 3.23 |

**Prediction performance:**

| Cell type | Pearson | RMSE |
|---|---:|---:|
| B cell | 0.978 | 0.481 |
| T cell | 0.984 | 0.456 |
| Macrophage | 0.978 | 0.805 |
| Mean by sample | 0.993 | 0.452 |

### Visualization

The figure compares predicted and true cell-type counts for each cell type. The dashed line represents perfect agreement.

![LEDC demo prediction results](examples/demo_output.png)

*Note: The example uses synthetic data and is intended to demonstrate package functionality, not biological generalization. Training and test mixtures are generated separately from the same underlying single-cell reference.*

For the complete executable example, see [`examples/demo_ledc.py`](examples/demo_ledc.py).

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

# In count mode, P is identical to C
Y_train = P
```

The function returns three matrices:

- `T`: Pseudobulk gene expression (samples × genes).
- `C`: Actual cell-type counts (samples × cell types).
- `P`: Cell-type prediction targets (samples × cell types), defined according to the selected mode.

When `mode="count"`, `P` contains absolute cell-type counts and is identical to `C`.

When `mode="proportion"`, `P` contains realized cell-type proportions, calculated by dividing each cell-type count by the total number of sampled cells.

Mathematically,

\[
P_{ik} =
\begin{cases}
C_{ik}, & \text{if mode = count},\\
\frac{C_{ik}}{\sum_{j=1}^{K} C_{ij}}, & \text{if mode = proportion}.
\end{cases}
\]

The default mode is `"count"`, consistent with the count-based deconvolution setting emphasized in the LEDC manuscript.

#### Pseudobulk generation parameters

The `generate_pseudobulk()` function generates pseudobulk mixtures from annotated single-cell RNA-seq data. It supports simulations with either fixed or variable total cell counts.

| Parameter | Default | Description |
|---|---|---|
| `adata` | Required | AnnData object containing single-cell expression and cell-type annotations. |
| `n_samples` | Required | Number of pseudobulk mixtures to generate. |
| `cell_types` | Required | Cell types included in the mixtures. |
| `rng` | Required | NumPy random number generator for reproducibility. |
| `mode` | `"count"` | `"count"` generates Poisson-distributed total cell counts and sets `P = C`. `"proportion"` uses a fixed number of cells and sets `P = C / row_sum(C)`. |
| `pb_cells` | `10` | Number of cells per mixture when `mode="proportion"`. |
| `poisson_mean` | `10` | Mean of the Poisson distribution determining total cell counts when `mode="count"`. |
| `alpha` | `1.0` | Concentration parameter for the Dirichlet distribution used to generate cell-type mixing weights. |

The function returns three objects: `T` (pseudobulk gene expression), `C` (actual cell-type counts), and `P` (cell-type prediction targets). In count mode, `P` is identical to `C`. In proportion mode, `P` contains the realized cell-type proportions obtained by normalizing each row of `C`.

The DataFrame conversion shown above is suitable for small datasets. Large datasets may require a memory-efficient sparse workflow.

### 2. Train LEDC

```python
from ledc import LEDC

model = LEDC(target="count")
model.fit(X_train, Y_train)
```
The prediction target must match the pseudobulk generation mode. For count-based deconvolution, use `mode="count"` and train the model with the count-valued `P` matrix:

```python
model = LEDC(target="count")
model.fit(X_train, P)
```

For proportion-based deconvolution, generate pseudobulk mixtures using `mode="proportion"` and train with the proportion-valued `P` matrix:

```python
model = LEDC(target="proportion")
model.fit(X_train, P)
```

The `target` argument specifies the outcome type, while the matrix passed to `fit()` supplies the actual training labels.

#### LEDC model parameters

LEDC fits an independent LightGBM regression model for each cell type. The number of boosting rounds is selected through cross-validation with early stopping.

| Parameter | Default | Description |
|---|---|---|
| `target` | `"count"` | Prediction target: `"count"` or `"proportion"`. |
| `n_estimators` | `200` | Maximum number of boosting rounds. |
| `learning_rate` | `0.1` | Learning rate for boosting. |
| `num_leaves` | `31` | Maximum number of leaves per tree. |
| `random_state` | `42` | Random seed for reproducibility. |
| `n_jobs` | `1` | Number of threads used for LightGBM computations. |
| `nfold` | `3` | Number of cross-validation folds. |
| `early_stopping_rounds` | `20` | Number of rounds without sufficient validation improvement before stopping. |
| `min_delta` | `0.0001` | Minimum improvement required for early stopping. |
| `verbose` | `False` | Whether to display training progress. |
| `expression_scale` | `"unknown"` | Training expression representation: `"unknown"`, `"raw_counts"`, `"cpm"`, or `"log_cpm"`. Stored as metadata without transforming the data. |

Example:

```python
from ledc import LEDC

model = LEDC(
    target="count",
    n_estimators=200,
    learning_rate=0.1,
    nfold=3,
    random_state=42
)

model.fit(X_train, C_train)

predictions = model.predict(X_test)
```

For proportion prediction, initialize the model with `target="proportion"` and train it using the cell-type proportion matrix `P_train`. Predictions can optionally be normalized to sum to one using `model.predict(X_test, normalize=True)`.

### 3. Predict cell-type abundance

```python
predictions = model.predict(X_test)
print(predictions.head())
```

`X_test` must contain compatible expression measurements and the training genes. LEDC aligns gene columns to the training order.

### Predict directly from AnnData

LEDC also supports prediction from AnnData objects, commonly used for single-cell RNA-seq and spatial transcriptomics datasets.

The `predict_adata()` method automatically aligns genes with the training data, handles sparse expression matrices, and performs predictions in batches.

```python
import anndata as ad
from ledc import LEDC

# Load spatial expression data
adata = ad.read_h5ad("spatial_data.h5ad")

# Load trained LEDC models
model = LEDC.load("saved_models")

# Predict cell-type abundances
predictions = model.predict_adata(
    adata,
    gene_id_column="gene_ids",
    batch_size=128,
    missing_genes="error",
    validate_scale=True
)

print(predictions.head())
```

The resulting DataFrame contains samples or spatial spots as rows and predicted cell types as columns. Sample identifiers from `adata.obs_names` are preserved.

The `gene_id_column` argument specifies the column in `adata.var` containing gene identifiers. If omitted, LEDC uses `adata.var_names`.

By default, all training genes must be present. Set `missing_genes="zero"` to explicitly fill missing training genes with zeros.

**Note:** Expression preprocessing should be compatible with the training data.


#### Prediction parameters

The `predict()` and `predict_adata()` methods provide options for controlling gene alignment, normalization, and batch prediction.

| Method | Parameter | Default | Description |
|---|---|---|---|
| `predict()` | `normalize` | `False` | If `True`, normalize predicted proportions to sum to one. Only supported when `target="proportion"`. |
| `predict()` | `missing_genes` | `"error"` | Raise an error if training genes are missing. Use `"zero"` to fill missing genes with zeros. |
| `predict_adata()` | `gene_id_column` | `None` | Use `adata.var_names` by default, or specify a column in `adata.var` containing gene IDs. |
| `predict_adata()` | `batch_size` | `128` | Number of samples or spatial spots processed per batch. |
| `predict_adata()` | `missing_genes` | `"error"` | Raise an error for missing training genes, or use `"zero"` to fill them with zeros. |
| `predict_adata()` | `validate_scale` | `False` | If `True`, checks input expression values against the model's declared `expression_scale` and warns about potential mismatches. Does not transform the data. |



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
    coordinates=spatial_positions,
    cell_types=["B cell", "T cell"],
    ncols=2,
    cmap="viridis",
    scale="shared",
    vmax="max",
    tissue_only=True,
    spot_size=13,
    save_path="spatial_maps.png",
    show=False,
)

```

The spatial positions DataFrame must contain `pxl_col_in_fullres` and `pxl_row_in_fullres`. The optional `in_tissue` column identifies tissue spots. Spot indices should match prediction indices.

### Spatial plotting parameters

The `plot_spatial()` function allows users to customize the visualization of predicted cell-type abundances.

| Parameter | Default | Description / Available options |
|---|---|---|
| `predictions` | Required | DataFrame containing predicted abundances (spots × cell types). |
| `coordinates` | Required | DataFrame containing spatial coordinates and spot barcodes. |
| `cell_types` | `None` | List of cell types to display. `None` plots all cell types. |
| `ncols` | `4` | Number of subplot columns. Rows are determined automatically. |
| `cmap` | `"viridis"` | Matplotlib colormap, such as `"viridis"`, `"plasma"`, or `"jet"`. |
| `scale` | `"individual"` | `"individual"` uses separate color scales; `"shared"` uses one common scale. |
| `vmax` | `"quantile"` | `"quantile"`, `"max"`, or a numeric upper limit. |
| `quantile` | `0.99` | Quantile used when `vmax="quantile"`. |
| `tissue_only` | `True` | If available, display only spots where `in_tissue == 1`. |
| `spot_size` | `10` | Size of the plotted spatial markers. |
| `figsize` | `None` | Optional figure size `(width, height)` in inches. |
| `image` | `None` | Optional H&E tissue image displayed in the background. |
| `image_extent` | `None` | Image boundaries in the same coordinate system as the spots. Required when using `image`. |
| `image_alpha` | `0.6` | Background image opacity, from 0 to 1. |
| `show` | `False` | Whether to display the figure. |
| `save_path` | `None` | Optional output path for saving the figure. |

The function returns a Matplotlib `Figure`, allowing further customization or saving with `fig.savefig()`.

### Example: Spatial deconvolution with LEDC

Once LEDC models have been trained, they can be applied to spatial transcriptomics data to estimate cell-type abundances across tissue locations.

```python
import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt

from ledc import LEDC, plot_spatial

# Load trained models
model = LEDC.load("saved_models")

# Load spatial gene expression
adata = ad.read_h5ad("visium_data.h5ad")

# Predict cell-type abundances
predictions = model.predict_adata(
    adata,
    gene_id_column="gene_ids"
)

# Load spatial coordinates
positions = pd.read_csv("tissue_positions.csv")

# Plot predicted cell-type abundances
fig = plot_spatial(
    predictions=predictions,
    coordinates=positions,
    ncols=4,
    cmap="viridis",
    scale="individual",
    vmax="quantile",
    quantile=0.99,
    tissue_only=True,
    spot_size=20
)

fig.savefig(
    "spatial_predictions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)
```

The spatial coordinate table should contain spot barcodes and the columns `pxl_col_in_fullres` and `pxl_row_in_fullres`. LEDC aligns predictions and coordinates using their spot identifiers.

The `scale` parameter supports:

- `"individual"`: separate color ranges for each cell type.
- `"shared"`: a common color range for comparing predicted abundances between cell types.

Users may also specify individual cell types, change the colormap, filter tissue spots, or provide a histology background image with an appropriate coordinate transformation.

### Run spatial prediction from the command line

LEDC provides a command-line example for applying trained models to spatial transcriptomics data.

```bash
python examples/predict_lymph.py \
  --adata visium_data.h5ad \
  --coordinates tissue_positions.csv \
  --models saved_models \
  --output results \
  --gene-id-column gene_ids
```

By default, the script generates spatial maps for **all predicted cell types**. Users can optionally select specific cell types using `--cell-types`.

For example, to plot only B cells and T cells:

```bash
python examples/predict_lymph.py \
  --adata visium_data.h5ad \
  --coordinates tissue_positions.csv \
  --models saved_models \
  --output results \
  --gene-id-column gene_ids \
  --cell-types "B cell" "T cell"
```

Other options include `--scale`, `--cmap`, `--spot-size`, and `--ncols`. Run `python examples/predict_lymph.py --help` for the complete list.

The script saves predicted cell-type abundances to `spatial_predictions.csv` and spatial maps to `spatial_maps.png` in the specified output directory.

### Example: Human lymph-node Visium data

The figure below illustrates the application of LEDC to a human lymph-node Visium dataset. Using trained LightGBM models, LEDC predicts cell-type abundances for 4,015 spatial spots across 39 cell types.

Eight cell types are shown to demonstrate the spatial distribution of predicted abundances across the tissue. The complete LEDC workflow supports visualization of all 39 predicted cell types; eight representative panels are displayed here for readability.

![LEDC spatial predictions for human lymph-node Visium data](examples/lymph_spatial_predictions.png)

A complete application using a human lymph-node Visium dataset is provided in [`examples/predict_lymph.py`](examples/predict_lymph.py). This example requires separately obtained data and trained model files.
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
