import numpy as np
import pandas as pd
import anndata as ad

from ledc import LEDC, generate_pseudobulk

import numpy as np
import pandas as pd
import anndata as ad

from ledc import LEDC, generate_pseudobulk

# Step 1: Create synthetic single-cell data
rng = np.random.default_rng(42)

adata = ad.AnnData(
    rng.poisson(5, size=(100, 20))
)

adata.obs["cell_type"] = (
    ["B cell"] * 50 + ["T cell"] * 50
)

adata.var_names = [f"Gene_{i}" for i in range(20)]

print("Step 1: Single-cell data created")

# Step 2: Generate pseudobulk data
T, C, P = generate_pseudobulk(
    adata=adata,
    n_samples=100,
    cell_types=["B cell", "T cell"],
    rng=np.random.default_rng(42),
    mode="count",
    poisson_mean=10
)

print("Step 2: Pseudobulk generated")
print("Expression shape:", T.shape)
print("Count shape:", C.shape)

# Step 3: Prepare training data
X_train = pd.DataFrame(
    T.toarray(),
    columns=adata.var_names
)

Y_train = C.copy()

# Step 4: Train LEDC
model = LEDC(target="count")
model.fit(X_train, Y_train)

print("Step 3: Model training complete")

# Step 5: Predict
predictions = model.predict(X_train)

print("\nPredicted cell counts:")
print(predictions.head())

# Step 6: Save models
model.save("saved_models")

print("\nLEDC pipeline completed successfully!")

model.save("saved_models")
# Load the saved LEDC models
loaded_model = LEDC.load("saved_models")

# Predict using the loaded models
loaded_predictions = loaded_model.predict(X_train)

# Compare original and loaded predictions
np.testing.assert_allclose(
    predictions.to_numpy(),
    loaded_predictions.to_numpy(),
    rtol=1e-6,
    atol=1e-8
)

print("Save/load test passed!")