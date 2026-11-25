"""Tabular baseline and gradient-boosted models (Logistic Regression, Random Forest, XGBoost)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from receivables_ml.config import TabularModelsConfig
from receivables_ml.evaluation.metrics import ModelPrediction


class BaseDualTabularModel:
    """Base interface pairing a risk classifier with a continuous delay regressor."""

    name: str = "base"

    def fit(
        self,
        X_train: np.ndarray,
        y_late_train: np.ndarray,
        y_delay_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_late_val: np.ndarray | None = None,
        y_delay_val: np.ndarray | None = None,
    ) -> "BaseDualTabularModel":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> ModelPrediction:
        raise NotImplementedError

    def save(self, directory: str | Path) -> Path:
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.name}.joblib"
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BaseDualTabularModel":
        return joblib.load(path)


class LogisticRegressionSuite(BaseDualTabularModel):
    """Standardized Logistic Regression classifier + Ridge delay regressor."""

    name = "logistic_regression"

    def __init__(self, config: TabularModelsConfig, seed: int = 42) -> None:
        self.config = config
        self.seed = seed
        self.classifier = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=config.logreg_c,
                        max_iter=2000,
                        random_state=seed,
                    ),
                ),
            ]
        )
        self.regressor = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=config.ridge_alpha, random_state=seed)),
            ]
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_late_train: np.ndarray,
        y_delay_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_late_val: np.ndarray | None = None,
        y_delay_val: np.ndarray | None = None,
    ) -> "LogisticRegressionSuite":
        self.classifier.fit(X_train, y_late_train)
        self.regressor.fit(X_train, y_delay_train)
        return self

    def predict(self, X: np.ndarray) -> ModelPrediction:
        probs = self.classifier.predict_proba(X)[:, 1]
        delay_preds = np.clip(self.regressor.predict(X), -15.0, 90.0)
        return ModelPrediction(probs=probs, delay_preds=delay_preds)


class RandomForestSuite(BaseDualTabularModel):
    """Random Forest classifier + Random Forest delay regressor."""

    name = "random_forest"

    def __init__(self, config: TabularModelsConfig, seed: int = 42) -> None:
        self.config = config
        self.seed = seed
        self.imputer = SimpleImputer(strategy="median")
        self.classifier = RandomForestClassifier(
            n_estimators=config.rf_n_estimators,
            max_depth=config.rf_max_depth,
            min_samples_leaf=config.rf_min_samples_leaf,
            n_jobs=-1,
            random_state=seed,
        )
        self.regressor = RandomForestRegressor(
            n_estimators=config.rf_n_estimators,
            max_depth=config.rf_max_depth,
            min_samples_leaf=config.rf_min_samples_leaf,
            n_jobs=-1,
            random_state=seed,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_late_train: np.ndarray,
        y_delay_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_late_val: np.ndarray | None = None,
        y_delay_val: np.ndarray | None = None,
    ) -> "RandomForestSuite":
        X_tr = self.imputer.fit_transform(X_train)
        self.classifier.fit(X_tr, y_late_train)
        self.regressor.fit(X_tr, y_delay_train)
        return self

    def predict(self, X: np.ndarray) -> ModelPrediction:
        X_imp = self.imputer.transform(X)
        probs = self.classifier.predict_proba(X_imp)[:, 1]
        delay_preds = np.clip(self.regressor.predict(X_imp), -15.0, 90.0)
        return ModelPrediction(probs=probs, delay_preds=delay_preds)


class XGBoostSuite(BaseDualTabularModel):
    """XGBoost classifier + XGBoost delay regressor."""

    name = "xgboost"

    def __init__(self, config: TabularModelsConfig, seed: int = 42) -> None:
        self.config = config
        self.seed = seed
        self.classifier = XGBClassifier(
            n_estimators=config.xgb_n_estimators,
            max_depth=config.xgb_max_depth,
            learning_rate=config.xgb_learning_rate,
            subsample=config.xgb_subsample,
            colsample_bytree=config.xgb_colsample_bytree,
            eval_metric="logloss",
            early_stopping_rounds=25,
            random_state=seed,
            n_jobs=4,
            tree_method="hist",
        )
        self.regressor = XGBRegressor(
            n_estimators=config.xgb_n_estimators,
            max_depth=config.xgb_max_depth,
            learning_rate=config.xgb_learning_rate,
            subsample=config.xgb_subsample,
            colsample_bytree=config.xgb_colsample_bytree,
            eval_metric="rmse",
            early_stopping_rounds=25,
            random_state=seed,
            n_jobs=4,
            tree_method="hist",
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_late_train: np.ndarray,
        y_delay_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_late_val: np.ndarray | None = None,
        y_delay_val: np.ndarray | None = None,
    ) -> "XGBoostSuite":
        if X_val is not None and y_late_val is not None:
            self.classifier.fit(
                X_train,
                y_late_train,
                eval_set=[(X_val, y_late_val)],
                verbose=False,
            )
        else:
            self.classifier.set_params(early_stopping_rounds=None)
            self.classifier.fit(X_train, y_late_train, verbose=False)

        if X_val is not None and y_delay_val is not None:
            self.regressor.fit(
                X_train,
                y_delay_train,
                eval_set=[(X_val, y_delay_val)],
                verbose=False,
            )
        else:
            self.regressor.set_params(early_stopping_rounds=None)
            self.regressor.fit(X_train, y_delay_train, verbose=False)
        return self

    def predict(self, X: np.ndarray) -> ModelPrediction:
        probs = self.classifier.predict_proba(X)[:, 1]
        delay_preds = np.clip(self.regressor.predict(X), -15.0, 90.0)
        return ModelPrediction(probs=probs, delay_preds=delay_preds)

    def save(self, directory: str | Path) -> Path:
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        clf_path = out_dir / "xgboost_classifier.json"
        reg_path = out_dir / "xgboost_regressor.json"
        self.classifier.save_model(str(clf_path))
        self.regressor.save_model(str(reg_path))
        meta_path = out_dir / "xgboost.joblib"
        joblib.dump(self, meta_path)
        return meta_path


def build_tabular_models(
    config: TabularModelsConfig, seed: int = 42
) -> dict[str, BaseDualTabularModel]:
    """Instantiate all 3 tabular model families with consistent config and seed."""
    return {
        "logistic_regression": LogisticRegressionSuite(config=config, seed=seed),
        "random_forest": RandomForestSuite(config=config, seed=seed),
        "xgboost": XGBoostSuite(config=config, seed=seed),
    }
