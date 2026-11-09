import numpy as np
import pandas as pd
import pytest
from ledc import LEDC, evaluate_predictions

def toy_data():
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.poisson(5, size=(35, 5)).astype(float), columns=[f'gene_{i}' for i in range(5)])
    Y = pd.DataFrame({'B_cell': 2*X['gene_0'], 'T_cell': X['gene_1']})
    return X, Y

def test_train_predict_and_reload(tmp_path):
    X, Y = toy_data()
    model = LEDC(n_estimators=15, early_stopping_rounds=5).fit(X, Y)
    pred = model.predict(X)
    assert pred.shape == Y.shape
    assert (pred.to_numpy() >= 0).all()
    model.save(tmp_path / 'saved')
    restored = LEDC.load(tmp_path / 'saved')
    pd.testing.assert_frame_equal(pred, restored.predict(X))
    pd.testing.assert_frame_equal(pred, restored.predict(X[X.columns[::-1]]))
    assert list(evaluate_predictions(Y, pred).columns) == ["Pearson", "RMSE"]

def test_missing_gene_rejected():
    X, Y = toy_data()
    model = LEDC(n_estimators=10, early_stopping_rounds=5).fit(X, Y)
    with pytest.raises(ValueError, match='Missing'):
        model.predict(X.drop(columns='gene_0'))
    assert model.predict(X.drop(columns='gene_0'), missing_genes='zero').shape == Y.shape
