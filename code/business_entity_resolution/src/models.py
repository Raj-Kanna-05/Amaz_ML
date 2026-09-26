"""
Model Training and Wrapper Module for Business Entity Resolution.
Supports LightGBM and XGBoost binary classifiers (MIT / Apache 2.0 licensed),
with a pure-NumPy fallback classifier for zero-dependency portability.
"""

from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd

# Safe dynamic imports for gradient boosting libraries
HAS_LIGHTGBM = False
HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    from .utils import get_logger
except (ImportError, ValueError):
    from src.utils import get_logger

logger = get_logger("Models")


class PureNumpyClassifier:
    """
    Zero-dependency pure-NumPy fallback binary classifier using regularized logistic regression.
    Guarantees execution even if external C++ libraries fail to load.
    """

    def __init__(self, learning_rate: float = 0.05, n_epochs: int = 50, random_state: int = 42):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        X_arr = np.nan_to_num(X.values.astype(float), nan=0.0)
        y_arr = y.astype(float)
        n_samples, n_features = X_arr.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_epochs):
            linear = np.dot(X_arr, self.weights) + self.bias
            preds = 1.0 / (1.0 + np.exp(-np.clip(linear, -15.0, 15.0)))
            errors = preds - y_arr
            dw = (np.dot(X_arr.T, errors) / n_samples) + (0.01 * self.weights)
            db = float(np.sum(errors) / n_samples)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        self.feature_importances_ = np.abs(self.weights)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if len(X) == 0:
            return np.array([], dtype=float)
        X_arr = np.nan_to_num(X.values.astype(float), nan=0.0)
        if self.weights is None:
            self.weights = np.zeros(X_arr.shape[1], dtype=float)
        linear = np.dot(X_arr, self.weights) + self.bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(linear, -15.0, 15.0)))
        return np.column_stack([1.0 - probs, probs])


class EntityMatchingModel:
    """
    Unified binary classifier for pairwise entity matching.
    Supports LightGBM and XGBoost, with automatic fallback to PureNumpyClassifier.
    """

    def __init__(
        self,
        model_type: str = "lightgbm",
        params: Optional[Dict[str, Any]] = None,
        init_model: bool = True,
    ):
        self.model_type = model_type.lower()
        self.params = params or {}
        self.model = None
        self.feature_names: List[str] = []
        self.single_class_prediction: Optional[float] = None
        if init_model:
            self._init_model()

    def _init_model(self):
        # 1. Try LightGBM if requested and available
        if self.model_type == "lightgbm" and HAS_LIGHTGBM:
            try:
                default_params = {
                    "n_estimators": 300,
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "max_depth": -1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,
                    "n_jobs": -1,
                    "verbose": -1,
                }
                default_params.update(self.params)
                self.model = lgb.LGBMClassifier(**default_params)
                return
            except Exception as e:
                logger.warning(f"LightGBM initialization failed ({e}); checking XGBoost fallback.")

        # 2. Try XGBoost if requested or as fallback
        if (self.model_type in ("xgboost", "lightgbm")) and HAS_XGBOOST:
            try:
                self.model_type = "xgboost"
                default_params = {
                    "n_estimators": 300,
                    "learning_rate": 0.05,
                    "max_depth": 6,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,
                    "n_jobs": -1,
                    "eval_metric": "logloss",
                }
                default_params.update(self.params)
                self.model = xgb.XGBClassifier(**default_params)
                return
            except Exception as e:
                logger.warning(f"XGBoost initialization failed ({e}); checking other options.")

        # 3. If LightGBM is available, use it
        if HAS_LIGHTGBM:
            try:
                self.model_type = "lightgbm"
                self.model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42)
                return
            except Exception:
                pass

        # 4. Fallback to PureNumpyClassifier
        logger.warning("Neither LightGBM nor XGBoost could be loaded. Using zero-dependency PureNumpyClassifier.")
        self.model_type = "numpy"
        self.model = PureNumpyClassifier()

    def fit(self, X: pd.DataFrame, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "EntityMatchingModel":
        """Fits binary classifier on training features and labels."""
        if feature_names:
            self.feature_names = feature_names
        elif isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)

        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            self.single_class_prediction = float(unique_classes[0]) if len(unique_classes) > 0 else 0.0
            logger.warning(f"Training data has single class ({self.single_class_prediction}); setting constant prediction.")
            return self

        self.single_class_prediction = None
        if self.model is None:
            self._init_model()
        assert self.model is not None
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns predicted probability for positive class (match)."""
        if len(X) == 0:
            return np.array([], dtype=float)

        if self.single_class_prediction is not None:
            return np.full(len(X), float(self.single_class_prediction))

        if isinstance(X, pd.DataFrame) and self.feature_names:
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[self.feature_names]

        if self.model is None:
            self._init_model()
        assert self.model is not None
        probs = self.model.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return probs[:, 1]
        elif probs.ndim == 2 and probs.shape[1] == 1:
            return probs[:, 0]
        return probs.ravel()

    def get_feature_importances(self) -> Dict[str, float]:
        """Returns sorted dictionary of feature importances."""
        if hasattr(self.model, "feature_importances_") and self.feature_names:
            importances = self.model.feature_importances_
            total = sum(importances) if sum(importances) > 0 else 1.0
            norm_imp = {
                name: float(imp / total)
                for name, imp in zip(self.feature_names, importances)
            }
            return dict(sorted(norm_imp.items(), key=lambda x: x[1], reverse=True))
        return {}

    def save(self, filepath: str) -> None:
        """Serializes model and metadata to disk."""
        payload = {
            "model_type": self.model_type,
            "params": self.params,
            "feature_names": self.feature_names,
            "single_class_prediction": self.single_class_prediction,
            "model": self.model,
        }
        joblib.dump(payload, filepath)

    @classmethod
    def load(cls, filepath: str) -> "EntityMatchingModel":
        """Loads serialized model artifact from disk."""
        payload = joblib.load(filepath)
        instance = cls(
            model_type=payload.get("model_type", "lightgbm"),
            params=payload.get("params", {}),
            init_model=False,
        )
        instance.feature_names = payload.get("feature_names", [])
        instance.single_class_prediction = payload.get("single_class_prediction", None)
        instance.model = payload.get("model", None)
        return instance
