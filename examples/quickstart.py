"""Run the actual CV-based LEDC training interface on synthetic data."""
import numpy as np
import pandas as pd
from ledc import LEDC, evaluate
rng = np.random.default_rng(7)
X = pd.DataFrame(rng.poisson(5, size=(45, 5)).astype(float), columns=[f'gene_{i}' for i in range(5)])
Y = pd.DataFrame({'B_cell': 2*X['gene_0'], 'T_cell': X['gene_1']})
model = LEDC(n_estimators=20, early_stopping_rounds=5).fit(X.iloc[:40], Y.iloc[:40])
pred = model.predict(X.iloc[40:])
print(pred)
print(evaluate(Y.iloc[40:], pred))
