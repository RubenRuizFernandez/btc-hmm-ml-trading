"""LightGBM signal classifier wrapper."""
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap

from src.config import LGBM_PARAMS, LGBM_EARLY_STOPPING_ROUNDS, LGBM_VAL_FRACTION


class LGBMSignalModel:
    """Binary classifier: predicts P(up-move) given the feature matrix."""

    def __init__(self, params: dict = None):
        self.params = params or dict(LGBM_PARAMS)
        self.model: lgb.Booster | None = None
        self.feature_names: list = []
        self._shap_values: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LGBMSignalModel":
        """
        Fit on in-sample data.

        Parameters
        ----------
        X : feature DataFrame (IS rows only)
        y : binary target {0, 1}  (1 = long profitable, 0 = otherwise)
        """
        self.feature_names = list(X.columns)

        # Validation split from the tail of IS (time-ordered, no shuffle)
        n_val = max(int(len(X) * LGBM_VAL_FRACTION), 1)
        X_tr, X_val = X.iloc[:-n_val], X.iloc[-n_val:]
        y_tr, y_val = y.iloc[:-n_val], y.iloc[-n_val:]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        callbacks = [
            lgb.early_stopping(LGBM_EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=-1),
        ]

        self.model = lgb.train(
            self.params,
            dtrain,
            valid_sets=[dval],
            callbacks=callbacks,
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(up-move) for each row."""
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        return self.model.predict(X[self.feature_names])

    def compute_shap(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values (cached)."""
        explainer = shap.TreeExplainer(self.model)
        self._shap_values = explainer.shap_values(X[self.feature_names])
        return self._shap_values

    @property
    def feature_importance(self) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        imp = self.model.feature_importance(importance_type="gain")
        return pd.Series(imp, index=self.feature_names).sort_values(ascending=False)
