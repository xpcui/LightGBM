"""
Evaluate prediction performance
"""

import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error


def get_scores(P_hat, P_true, celltypes):
    """
    Evaluate prediction performance by cell type and by sample.

    Parameters
    ----------
    P_hat : numpy.ndarray
        Predicted proportions with shape samples x celltypes.
    P_true : numpy.ndarray
        True proportions with shape samples x celltypes.
    celltypes : list of str
        Cell type names.

    Returns
    -------
    pandas.DataFrame
        Summary table with Pearson correlation and RMSE for each cell type,
        plus average sample-level Pearson correlation and RMSE.
    """
    # By cell type
    pearson_c = [
        pearsonr(P_hat[:, j], P_true[:, j])[0]
        for j in range(P_true.shape[1])
    ]
    rmse_c = [
        np.sqrt(mean_squared_error(P_true[:, j], P_hat[:, j]))
        for j in range(P_true.shape[1])
    ]

    # By sample
    pearson_s = [
        pearsonr(P_hat[i, :], P_true[i, :])[0]
        for i in range(P_true.shape[0])
    ]
    rmse_s = [
        np.sqrt(mean_squared_error(P_true[i, :], P_hat[i, :]))
        for i in range(P_true.shape[0])
    ]

    # Assemble into DataFrame
    df = pd.DataFrame({
        "Mean_Pearson": pearson_c + [np.mean(pearson_s)],
        "Mean_RMSE": rmse_c + [np.mean(rmse_s)]
    }, index=celltypes + ["-- By sample --"])

    return df



def evaluate_lgbm_decon_models(
    models,
    X_train,
    P_train,
    X_test,
    P_test,
    row_normalize_func,
    get_scores_func,
    model_name="model"
):
    """
    Evaluate trained LightGBM deconvolution models on training and test data.

    Parameters
    ----------
    models : dict
        Dictionary of trained models.
        Keys should be cell type names.

    X_train : pd.DataFrame
        Training expression matrix.
        Rows = samples, columns = genes.

    P_train : pd.DataFrame
        Training true proportion matrix.
        Rows = samples, columns = cell types.

    X_test : pd.DataFrame
        Test expression matrix.
        Rows = samples, columns = genes.

    P_test : pd.DataFrame
        Test true proportion matrix.
        Rows = samples, columns = cell types.

    row_normalize_func : function
        Function used to normalize predicted proportions so each row sums to 1.

    get_scores_func : function
        Function used to compute the score matrix.

    model_name : str
        Name used for printing only.

    Returns
    -------
    scores_train : pd.DataFrame
        Score matrix for training data.

    scores_test : pd.DataFrame
        Score matrix for test data.

    train_preds_norm_df : pd.DataFrame
        Normalized training predictions.

    test_preds_norm_df : pd.DataFrame
        Normalized test predictions.
    """

    # Make sure test genes are in the same order as training genes
    X_test_aligned = X_test[X_train.columns]

    # Make sure cell types are consistent
    cell_types = list(P_train.columns)

    missing_models = set(cell_types) - set(models.keys())
    if len(missing_models) > 0:
        raise ValueError(f"Missing models for cell types: {missing_models}")

    # Empty prediction tables
    train_preds = pd.DataFrame(
        index=X_train.index,
        columns=cell_types
    )

    test_preds = pd.DataFrame(
        index=X_test_aligned.index,
        columns=cell_types
    )

    # Predict for each cell type
    for cell_type in cell_types:
        model = models[cell_type]

        train_preds[cell_type] = model.predict(X_train)
        test_preds[cell_type] = model.predict(X_test_aligned)

    # Row-normalize predictions
    train_preds_norm = row_normalize_func(train_preds.to_numpy())
    test_preds_norm = row_normalize_func(test_preds.to_numpy())

    # Convert normalized predictions back to DataFrame
    train_preds_norm_df = pd.DataFrame(
        train_preds_norm,
        index=X_train.index,
        columns=cell_types
    )

    test_preds_norm_df = pd.DataFrame(
        test_preds_norm,
        index=X_test_aligned.index,
        columns=cell_types
    )

    # Compute score matrices
    scores_train = get_scores_func(
        train_preds_norm,
        P_train.to_numpy(),
        cell_types
    )

    scores_test = get_scores_func(
        test_preds_norm,
        P_test[cell_types].to_numpy(),
        cell_types
    )

    print(f"{model_name} evaluation complete.")
    print("Train scores shape:", scores_train.shape)
    print("Test scores shape:", scores_test.shape)

    return scores_train, scores_test, train_preds_norm_df, test_preds_norm_df