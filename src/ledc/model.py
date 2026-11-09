"""LEDC: one LightGBM regression booster per cell type, with training-only CV."""
from pathlib import Path
import json
import lightgbm as lgb
import numpy as np
import pandas as pd

class LEDC:
    """LightGBM Ensemble DeConvolution.

    Parameters match the research training code (200 rounds, 3 folds,
    20-round early stopping, RMSE, learning rate 0.1 and 31 leaves).
    X: samples by genes; Y: samples by cell types, matched by row index.
    """
    def __init__(self, target="count", n_estimators=200, learning_rate=0.1,
                 num_leaves=31, random_state=42, n_jobs=1, nfold=3,
                 early_stopping_rounds=20, min_delta=1e-4, verbose=False):
        if target not in ("count", "proportion"):
            raise ValueError("target must be 'count' or 'proportion'")
        self.target = target
        self.params = dict(n_estimators=n_estimators, learning_rate=learning_rate,
                           num_leaves=num_leaves, random_state=random_state,
                           n_jobs=n_jobs, nfold=nfold,
                           early_stopping_rounds=early_stopping_rounds,
                           min_delta=min_delta, verbose=verbose)
        self.models_ = None
        self.genes_ = None
        self.cell_types_ = None
        self.cv_results_ = None

    @staticmethod
    def _check_df(value, name):
        if not isinstance(value, pd.DataFrame) or value.empty:
            raise ValueError(f"{name} must be a nonempty pandas DataFrame")
        if not value.columns.is_unique or not value.index.is_unique:
            raise ValueError(f"{name} must have unique indices and columns")
        if not all(isinstance(c, str) for c in value.columns):
            raise ValueError(f"{name} must have string column names")
        if not np.issubdtype(value.to_numpy().dtype, np.number) or not np.isfinite(value.to_numpy()).all():
            raise ValueError(f"{name} must contain finite numeric values")

    def fit(self, X, Y):
        self._check_df(X, 'X')
        self._check_df(Y, 'Y')
        if not X.index.equals(Y.index):
            raise ValueError('X and Y must have matching sample indices in the same order')
        if (Y.to_numpy() < 0).any():
            raise ValueError('Cell-type targets must be nonnegative')
        if self.target == 'proportion' and (Y.to_numpy() > 1 + 1e-8).any():
            raise ValueError('Proportion targets must be <= 1')
        p = self.params
        if len(X) < p['nfold']:
            raise ValueError('Number of samples must be at least nfold')
        self.genes_, self.cell_types_ = list(X.columns), list(Y.columns)
        self.models_, self.cv_results_ = {}, {}
        params = dict(objective='regression', metric='rmse', verbosity=-1,
                      seed=p['random_state'], learning_rate=p['learning_rate'],
                      num_leaves=p['num_leaves'], num_threads=p['n_jobs'])
        for ct in self.cell_types_:
            dataset = lgb.Dataset(X, label=Y[ct], feature_name='auto', free_raw_data=False)
            cv = lgb.cv(params, dataset, num_boost_round=p['n_estimators'],
                        nfold=p['nfold'], stratified=False, seed=p['random_state'],
                        callbacks=[lgb.early_stopping(p['early_stopping_rounds'],
                                                      min_delta=p['min_delta'], verbose=p['verbose'])])
            key = 'valid rmse-mean'
            rounds = len(cv[key])
            self.models_[ct] = lgb.train(params, dataset, num_boost_round=rounds)
            self.cv_results_[ct] = dict(best_iteration=rounds, cv_rmse=cv[key][-1])
        return self

    def predict(self, X, normalize=False, missing_genes='error'):
        if self.models_ is None:
            raise RuntimeError('Fit or load a model before predicting')
        self._check_df(X, 'X')
        missing = [gene for gene in self.genes_ if gene not in X.columns]
        if missing and missing_genes == 'error':
            raise ValueError(f'Missing {len(missing)} training genes (e.g. {missing[:5]}); '
                             "set missing_genes='zero' only if intentional")
        if missing_genes not in ('error', 'zero'):
            raise ValueError("missing_genes must be 'error' or 'zero'")
        aligned = X.reindex(columns=self.genes_, fill_value=0)
        preds = np.column_stack([np.clip(self.models_[ct].predict(aligned), 0, None)
                                 for ct in self.cell_types_])
        if normalize:
            if self.target != 'proportion':
                raise ValueError('normalize=True applies only to proportion predictions')
            totals = preds.sum(axis=1, keepdims=True)
            preds = np.divide(preds, totals, out=np.zeros_like(preds), where=totals > 0)
        return pd.DataFrame(preds, index=X.index, columns=self.cell_types_)

    def save(self, path):
        if self.models_ is None:
            raise RuntimeError('No models to save')
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        metadata = dict(target=self.target, params=self.params, genes=self.genes_,
                        cell_types=self.cell_types_, cv_results=self.cv_results_)
        (path/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        for i, ct in enumerate(self.cell_types_):
            self.models_[ct].save_model(str(path/f'model_{i}.txt'))
        return self

    @classmethod
    def load(cls, path):
        path = Path(path)
        meta = json.loads((path/'metadata.json').read_text(encoding='utf-8'))
        obj = cls(target=meta['target'], **meta['params'])
        obj.genes_, obj.cell_types_ = meta['genes'], meta['cell_types']
        obj.cv_results_ = meta.get('cv_results')
        obj.models_ = {ct: lgb.Booster(model_file=str(path/f'model_{i}.txt'))
                       for i, ct in enumerate(obj.cell_types_)}
        return obj

    @classmethod
    def from_legacy_models(cls, model_dir, model_prefix, cell_types, genes, target='count'):
        """Load the individual .txt models written by the original HPCC routine.

        Gene order and cell-type labels must be supplied from the training reference.
        """
        obj = cls(target=target)
        obj.genes_ = list(genes)
        obj.cell_types_ = list(cell_types)
        obj.models_ = {}
        for ct in obj.cell_types_:
            safe = str(ct).replace('/', '_').replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')
            file = Path(model_dir)/f'{model_prefix}_{safe}.txt'
            if not file.is_file():
                raise FileNotFoundError(file)
            booster = lgb.Booster(model_file=str(file))
            if booster.num_feature() != len(obj.genes_):
                raise ValueError(f'Feature count mismatch for {ct}: {booster.num_feature()} != {len(obj.genes_)}')
            obj.models_[ct] = booster
        return obj
