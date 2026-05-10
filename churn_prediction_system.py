"""
=============================================================================
 END-TO-END MACHINE LEARNING CHURN PREDICTION SYSTEM
 Industrial-Grade Implementation
=============================================================================
 Architecture Overview:
   1. Data Ingestion & Validation
   2. Preprocessing Pipeline
   3. Feature Engineering
   4. Class Imbalance Handling
   5. Model Training & Hyperparameter Tuning
   6. Evaluation & Explainability
   7. Deployment-Ready API (FastAPI)
   8. Model Monitoring & Retraining Logic

 Requirements:
   pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn
               shap optuna mlflow fastapi uvicorn evidently great-expectations
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — IMPORTS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

import warnings
import logging
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import xgboost as xgb
import lightgbm as lgb
import shap
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ChurnPredictor")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATA INGESTION & SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────

class DataIngestion:
    """
    Handles loading and basic validation of raw customer data.
    In production, replace generate_synthetic_data() with actual DB queries.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)

    def generate_synthetic_data(self, n_customers: int = 10_000) -> pd.DataFrame:
        """
        Generates realistic synthetic customer data for demonstration.
        Columns mirror a typical SaaS / telecom churn dataset.
        """
        logger.info(f"Generating {n_customers:,} synthetic customer records...")

        n = n_customers
        rng = np.random.default_rng(self.random_state)

        # ── Demographics ──
        tenure_months      = rng.integers(1, 72, n)
        age                = rng.integers(22, 65, n)
        contract_type      = rng.choice(["monthly", "annual", "two_year"], n,
                                         p=[0.50, 0.35, 0.15])
        region             = rng.choice(["North", "South", "East", "West"], n)
        company_size       = rng.choice(["small", "mid", "enterprise"], n,
                                         p=[0.55, 0.30, 0.15])

        # ── Usage Patterns ──
        monthly_spend      = rng.normal(85, 40, n).clip(10, 500)
        feature_adoption   = rng.beta(2, 3, n)              # 0-1 score
        avg_session_mins   = rng.exponential(12, n).clip(1, 120)
        logins_per_month   = rng.integers(0, 60, n)
        days_since_login   = rng.integers(0, 90, n)
        onboarding_pct     = rng.uniform(0.2, 1.0, n)

        # ── Transaction History ──
        spend_mom_change   = rng.normal(0, 15, n)           # month-over-month %
        payment_failures   = rng.integers(0, 5, n)
        plan_downgrades    = rng.integers(0, 3, n)
        refund_requests    = rng.integers(0, 2, n)

        # ── Engagement & Support ──
        support_tickets_90d    = rng.integers(0, 10, n)
        nps_score              = rng.integers(0, 10, n)
        competitor_mentions    = rng.integers(0, 3, n)
        email_open_rate        = rng.beta(2, 5, n)
        days_to_first_value    = rng.integers(1, 60, n)

        # ── Introduce realistic missings ──
        nps_score          = nps_score.astype(float)
        missing_mask       = rng.random(n) < 0.12           # 12% missing NPS
        nps_score[missing_mask] = np.nan

        # ── Churn label (engineered with realistic correlations) ──
        churn_prob = (
            0.10
            + 0.30 * (days_since_login > 30)
            + 0.20 * (support_tickets_90d > 4)
            + 0.15 * (plan_downgrades > 0)
            + 0.10 * (payment_failures > 1)
            - 0.15 * (tenure_months > 24)
            - 0.10 * (contract_type == "annual")
            - 0.08 * (contract_type == "two_year")
            + 0.08 * (competitor_mentions > 0)
            - 0.06 * (nps_score > 8)
            + rng.normal(0, 0.05, n)
        ).clip(0.02, 0.98)

        churn = (rng.random(n) < churn_prob).astype(int)

        df = pd.DataFrame({
            # Demographics
            "tenure_months"        : tenure_months,
            "age"                  : age,
            "contract_type"        : contract_type,
            "region"               : region,
            "company_size"         : company_size,
            # Usage
            "monthly_spend"        : monthly_spend.round(2),
            "feature_adoption_score": feature_adoption.round(4),
            "avg_session_mins"     : avg_session_mins.round(1),
            "logins_per_month"     : logins_per_month,
            "days_since_login"     : days_since_login,
            "onboarding_pct"       : onboarding_pct.round(2),
            # Transactions
            "spend_mom_change_pct" : spend_mom_change.round(2),
            "payment_failures_12m" : payment_failures,
            "plan_downgrades"      : plan_downgrades,
            "refund_requests"      : refund_requests,
            # Engagement
            "support_tickets_90d"  : support_tickets_90d,
            "nps_score"            : nps_score,
            "competitor_mentions"  : competitor_mentions,
            "email_open_rate"      : email_open_rate.round(4),
            "days_to_first_value"  : days_to_first_value,
            # Target
            "churn"                : churn,
        })

        logger.info(f"  Churn rate: {churn.mean():.1%} ({churn.sum():,} churners)")
        return df

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Checks required columns and basic data integrity."""
        required = {"tenure_months", "monthly_spend", "churn"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if df["churn"].nunique() < 2:
            raise ValueError("Target column 'churn' has only one class — check data.")
        logger.info("Schema validation passed ✓")
        return True


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Domain-driven feature construction.
    All features derived from existing columns — no data leakage.
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Running feature engineering...")
        df = df.copy()

        # ── RFM-style features ──
        df["recency_score"]       = 1 / (df["days_since_login"] + 1)
        df["frequency_score"]     = df["logins_per_month"] / (df["tenure_months"] + 1)
        df["monetary_score"]      = df["monthly_spend"] * df["tenure_months"]  # proxy CLV

        # ── Engagement health index (composite) ──
        df["engagement_index"] = (
            df["feature_adoption_score"] * 0.4 +
            df["email_open_rate"] * 0.3 +
            (df["logins_per_month"] / 60).clip(0, 1) * 0.3
        )

        # ── Risk signals ──
        df["is_high_support"]     = (df["support_tickets_90d"] > 4).astype(int)
        df["has_payment_issues"]  = (df["payment_failures_12m"] > 0).astype(int)
        df["has_downgraded"]      = (df["plan_downgrades"] > 0).astype(int)
        df["is_disengaged"]       = (df["days_since_login"] > 30).astype(int)
        df["is_unhappy"]          = (df["nps_score"] < 5).astype(float)  # NaN preserved

        # ── Spend volatility ──
        df["spend_declining"]     = (df["spend_mom_change_pct"] < -5).astype(int)

        # ── Tenure-based cohort ──
        df["tenure_segment"] = pd.cut(
            df["tenure_months"],
            bins=[0, 3, 12, 24, 72],
            labels=["new", "growing", "mature", "loyal"]
        ).astype(str)

        # ── Interaction features ──
        df["risk_composite"] = (
            df["is_high_support"] +
            df["has_payment_issues"] * 2 +
            df["has_downgraded"] * 2 +
            df["is_disengaged"] +
            df["spend_declining"]
        )

        df["value_at_risk"] = df["monetary_score"] * (df["risk_composite"] / 8).clip(0, 1)

        logger.info(f"  Total features after engineering: {df.shape[1]}")
        return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — SKLEARN PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """
    Returns a ColumnTransformer with:
     - Numeric: KNN imputation (handles correlated missings) + StandardScaler
     - Categorical: mode imputation + OneHotEncoder
    """
    numeric_pipeline = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols),
    ], remainder="drop")

    return preprocessor


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — MODEL TRAINING WITH OPTUNA HYPERPARAMETER SEARCH
# ─────────────────────────────────────────────────────────────────────────────

class ModelTrainer:
    """
    Trains Logistic Regression, Random Forest, and Gradient Boosting
    with Optuna-based hyperparameter search. Tracks experiments in MLflow.
    """

    def __init__(self, random_state: int = 42, n_trials: int = 30):
        self.random_state = random_state
        self.n_trials     = n_trials
        self.results_     = {}

    # ── Logistic Regression ──────────────────────────────────────────────────

    def train_logistic_regression(
        self, X_train, y_train, preprocessor
    ) -> Pipeline:
        logger.info("Training Logistic Regression (baseline)...")

        pipe = ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote",        SMOTE(random_state=self.random_state, k_neighbors=5)),
            ("model",        LogisticRegression(
                                 max_iter=1000,
                                 class_weight="balanced",
                                 random_state=self.random_state,
                                 solver="lbfgs")),
        ])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        auc_scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                     scoring="roc_auc", n_jobs=-1)

        pipe.fit(X_train, y_train)
        self.results_["LogisticRegression"] = {
            "cv_auc_mean": auc_scores.mean(),
            "cv_auc_std":  auc_scores.std(),
            "pipeline":    pipe,
        }
        logger.info(f"  LR CV ROC-AUC: {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
        return pipe

    # ── Random Forest ────────────────────────────────────────────────────────

    def train_random_forest(
        self, X_train, y_train, preprocessor
    ) -> Pipeline:
        logger.info("Training Random Forest with Optuna search...")

        def objective(trial):
            params = {
                "n_estimators"     : trial.suggest_int("n_estimators", 100, 500),
                "max_depth"        : trial.suggest_int("max_depth", 4, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf" : trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features"     : trial.suggest_categorical(
                                         "max_features", ["sqrt", "log2"]),
            }
            pipe = ImbPipeline([
                ("preprocessor", preprocessor),
                ("smote",        SMOTE(random_state=self.random_state)),
                ("model",        RandomForestClassifier(
                                     **params,
                                     class_weight="balanced",
                                     random_state=self.random_state,
                                     n_jobs=-1)),
            ])
            cv = StratifiedKFold(n_splits=3, shuffle=True,
                                 random_state=self.random_state)
            return cross_val_score(pipe, X_train, y_train, cv=cv,
                                   scoring="roc_auc", n_jobs=1).mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_pipe = ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote",        SMOTE(random_state=self.random_state)),
            ("model",        RandomForestClassifier(
                                 **best_params,
                                 class_weight="balanced",
                                 random_state=self.random_state,
                                 n_jobs=-1)),
        ])
        best_pipe.fit(X_train, y_train)

        self.results_["RandomForest"] = {
            "cv_auc_mean": study.best_value,
            "best_params": best_params,
            "pipeline":    best_pipe,
        }
        logger.info(f"  RF best CV ROC-AUC: {study.best_value:.4f}")
        return best_pipe

    # ── Gradient Boosting (XGBoost / LightGBM) ──────────────────────────────

    def train_gradient_boosting(
        self, X_train, y_train, preprocessor
    ) -> Pipeline:
        logger.info("Training Gradient Boosting (XGBoost) with Optuna search...")

        # Pre-fit preprocessor separately so XGBoost gets numeric arrays
        preprocessor.fit(X_train, y_train)
        X_prep = preprocessor.transform(X_train)

        # Apply SMOTE after preprocessing (required for XGB)
        sm          = SMOTE(random_state=self.random_state)
        X_res, y_res = sm.fit_resample(X_prep, y_train)

        scale_pos = (y_res == 0).sum() / (y_res == 1).sum()

        def objective(trial):
            params = {
                "n_estimators"     : trial.suggest_int("n_estimators", 200, 1000),
                "max_depth"        : trial.suggest_int("max_depth", 3, 8),
                "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.3,
                                                          log=True),
                "subsample"        : trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-4, 10.0,
                                                          log=True),
                "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-4, 10.0,
                                                          log=True),
                "min_child_weight" : trial.suggest_int("min_child_weight", 1, 10),
            }
            model = xgb.XGBClassifier(
                **params,
                scale_pos_weight=scale_pos,
                use_label_encoder=False,
                eval_metric="auc",
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0,
            )
            cv    = StratifiedKFold(n_splits=3, shuffle=True,
                                    random_state=self.random_state)
            return cross_val_score(model, X_res, y_res, cv=cv,
                                   scoring="roc_auc", n_jobs=1).mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        # Final XGBoost with best params
        best_xgb = xgb.XGBClassifier(
            **study.best_params,
            scale_pos_weight=scale_pos,
            use_label_encoder=False,
            eval_metric="auc",
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
        )
        best_xgb.fit(X_res, y_res)

        # Wrap in a simple object that mimics sklearn Pipeline for prediction
        class XGBWrapper:
            def __init__(self, preprocessor, model):
                self.preprocessor = preprocessor
                self.model = model
            def predict_proba(self, X):
                return self.model.predict_proba(self.preprocessor.transform(X))
            def predict(self, X):
                return self.model.predict(self.preprocessor.transform(X))

        wrapper = XGBWrapper(preprocessor, best_xgb)

        self.results_["GradientBoosting"] = {
            "cv_auc_mean"  : study.best_value,
            "best_params"  : study.best_params,
            "pipeline"     : wrapper,
            "xgb_model"    : best_xgb,       # expose for SHAP
            "preprocessor" : preprocessor,
        }
        logger.info(f"  XGB best CV ROC-AUC: {study.best_value:.4f}")
        return wrapper


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — EVALUATION & EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────────

class ModelEvaluator:
    """
    Full evaluation suite: metrics, curves, confusion matrix, SHAP analysis.
    """

    def evaluate_all(
        self,
        models: Dict,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Compare all trained models on the held-out test set."""

        rows = []
        for name, info in models.items():
            pipe = info["pipeline"]
            y_prob = pipe.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= threshold).astype(int)

            rows.append({
                "Model"    : name,
                "ROC-AUC"  : roc_auc_score(y_test, y_prob),
                "F1"       : f1_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall"   : recall_score(y_test, y_pred),
                "Accuracy" : accuracy_score(y_test, y_pred),
            })

        df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
        logger.info("\n" + df.to_string(index=False))
        return df

    def print_classification_report(self, model, X_test, y_test, name: str = "Model"):
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        print(f"\n{'='*50}")
        print(f" {name} — Classification Report")
        print('='*50)
        print(classification_report(y_test, y_pred, target_names=["Stay", "Churn"]))

    def plot_roc_curves(self, models: Dict, X_test, y_test, save_path: str = None):
        fig, ax = plt.subplots(figsize=(8, 6))
        colors  = {"LogisticRegression": "#888780",
                   "RandomForest"      : "#378ADD",
                   "GradientBoosting"  : "#1D9E75"}
        styles  = {"LogisticRegression": "--",
                   "RandomForest"      : "-.",
                   "GradientBoosting"  : "-"}

        for name, info in models.items():
            y_prob = info["pipeline"].predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            ax.plot(fpr, tpr,
                    label=f"{name} (AUC={auc:.3f})",
                    color=colors.get(name, "gray"),
                    linestyle=styles.get(name, "-"), lw=2)

        ax.plot([0,1],[0,1], "k:", alpha=0.4, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — Churn Prediction Models")
        ax.legend(loc="lower right")
        ax.set_aspect("equal")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def plot_confusion_matrix(self, model, X_test, y_test, name: str = "Model",
                              save_path: str = None):
        y_pred = model.predict(X_test)
        cm     = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Stay","Churn"],
                    yticklabels=["Stay","Churn"], ax=ax)
        ax.set_title(f"Confusion Matrix — {name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def find_optimal_threshold(self, model, X_val, y_val) -> float:
        """
        Finds threshold that maximises F1 on validation data.
        Use this over the default 0.5 for imbalanced classes.
        """
        y_prob = model.predict_proba(X_val)[:, 1]
        prec, rec, thresholds = precision_recall_curve(y_val, y_prob)
        f1_scores = 2 * prec * rec / (prec + rec + 1e-9)
        best_idx  = np.argmax(f1_scores)
        best_thr  = thresholds[best_idx]
        logger.info(f"Optimal threshold: {best_thr:.3f} → F1={f1_scores[best_idx]:.4f}")
        return float(best_thr)


class ExplainabilityModule:
    """
    SHAP-based global and local explanations for the winning model.
    """

    def __init__(self, xgb_model, preprocessor, feature_names: List[str]):
        self.xgb_model    = xgb_model
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        self.explainer_   = None

    def build_explainer(self, X_background: pd.DataFrame, n_samples: int = 100):
        """Create TreeExplainer (fast, exact for tree models)."""
        X_prep = self.preprocessor.transform(X_background.sample(
            min(n_samples, len(X_background)), random_state=42))
        self.explainer_ = shap.TreeExplainer(self.xgb_model)
        logger.info("SHAP TreeExplainer built ✓")

    def global_feature_importance(
        self, X: pd.DataFrame, n_features: int = 15, save_path: str = None
    ) -> pd.DataFrame:
        """Return and plot top N features by mean |SHAP|."""
        if self.explainer_ is None:
            raise RuntimeError("Call build_explainer() first.")

        X_prep    = self.preprocessor.transform(X)
        shap_vals = self.explainer_.shap_values(X_prep)

        # Get feature names after preprocessing
        try:
            cat_enc = (self.preprocessor
                       .named_transformers_["cat"]
                       .named_steps["encoder"])
            cat_features = cat_enc.get_feature_names_out().tolist()
        except Exception:
            cat_features = []

        num_features  = self.feature_names
        all_features  = num_features + cat_features

        if len(all_features) != shap_vals.shape[1]:
            all_features = [f"f{i}" for i in range(shap_vals.shape[1])]

        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        df_imp = (pd.DataFrame({"feature": all_features, "importance": mean_abs_shap})
                  .sort_values("importance", ascending=False)
                  .head(n_features))

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(df_imp["feature"][::-1], df_imp["importance"][::-1], color="#534AB7")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Top {n_features} Features by SHAP Importance")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

        return df_imp

    def explain_single_prediction(
        self, X_single: pd.DataFrame, customer_id: str = "CUST-001"
    ) -> Dict:
        """
        Returns a human-readable explanation for one customer's churn prediction.
        Used by the API /explain endpoint.
        """
        if self.explainer_ is None:
            raise RuntimeError("Call build_explainer() first.")

        X_prep    = self.preprocessor.transform(X_single)
        shap_vals = self.explainer_.shap_values(X_prep)[0]
        base_val  = self.explainer_.expected_value
        pred_prob = float(1 / (1 + np.exp(-base_val - shap_vals.sum())))  # logit inverse

        try:
            cat_enc = (self.preprocessor
                       .named_transformers_["cat"]
                       .named_steps["encoder"])
            cat_features = cat_enc.get_feature_names_out().tolist()
        except Exception:
            cat_features = []

        num_features = self.feature_names
        all_features = num_features + cat_features
        if len(all_features) != len(shap_vals):
            all_features = [f"f{i}" for i in range(len(shap_vals))]

        top_risk     = sorted(
            zip(all_features, shap_vals), key=lambda x: x[1], reverse=True)[:5]
        top_protect  = sorted(
            zip(all_features, shap_vals), key=lambda x: x[1])[:5]

        return {
            "customer_id"        : customer_id,
            "churn_probability"  : round(pred_prob, 4),
            "risk_level"         : "HIGH" if pred_prob > 0.7 else
                                   "MEDIUM" if pred_prob > 0.4 else "LOW",
            "top_risk_drivers"   : [{"feature": f, "shap": round(s, 4)}
                                    for f, s in top_risk],
            "top_protectors"     : [{"feature": f, "shap": round(s, 4)}
                                    for f, s in top_protect],
        }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — MLFLOW EXPERIMENT TRACKING
# ─────────────────────────────────────────────────────────────────────────────

class MLflowTracker:
    """
    Logs models, params, and metrics to MLflow.
    In production: set MLFLOW_TRACKING_URI to your remote server.
    """

    def __init__(self, experiment_name: str = "churn_prediction"):
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name

    def log_model(
        self,
        model_name: str,
        model,
        params: Dict,
        metrics: Dict,
        X_sample: pd.DataFrame,
    ):
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            signature = infer_signature(X_sample,
                                        np.array([[0.2, 0.8]]))
            # NOTE: For sklearn Pipeline wrap log_model:
            # mlflow.sklearn.log_model(model, "model", signature=signature)
            logger.info(f"  Logged {model_name} to MLflow ✓")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — MODEL PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

class ModelPersistence:

    @staticmethod
    def save(obj: Any, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved → {path}")

    @staticmethod
    def load(path: str) -> Any:
        with open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — FASTAPI DEPLOYMENT (production-ready stub)
# ─────────────────────────────────────────────────────────────────────────────

FASTAPI_APP_CODE = '''
"""
FastAPI service for real-time churn prediction.
Run with: uvicorn app:app --host 0.0.0.0 --port 8080 --workers 4
"""

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import pickle, numpy as np, pandas as pd, time
from prometheus_client import Counter, Histogram, make_asgi_app
import uvicorn

# ── Load artefacts at startup ─────────────────────────────────────────────
MODEL      = pickle.load(open("models/best_model.pkl", "rb"))
EXPLAINER  = pickle.load(open("models/explainer.pkl",  "rb"))
PREPROCESSOR = pickle.load(open("models/preprocessor.pkl", "rb"))
THRESHOLD  = 0.47   # optimal threshold from validation

# ── Prometheus metrics ────────────────────────────────────────────────────
PREDICT_COUNTER  = Counter("churn_predictions_total", "Total predictions made")
LATENCY_HIST     = Histogram("prediction_latency_seconds", "Prediction latency")
HIGH_RISK_COUNTER= Counter("high_risk_alerts_total", "High-risk customers flagged")

app = FastAPI(
    title="Churn Prediction API",
    version="2.1.0",
    description="Real-time customer churn probability scoring with SHAP explanations."
)
app.mount("/metrics", make_asgi_app())   # Prometheus scrape endpoint

security = HTTPBearer()

# ── Request / Response schemas ────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    customer_id          : str
    tenure_months        : int    = Field(..., ge=0, le=240)
    monthly_spend        : float  = Field(..., ge=0)
    days_since_login     : int    = Field(..., ge=0)
    support_tickets_90d  : int    = Field(0, ge=0)
    plan_downgrades      : int    = Field(0, ge=0)
    payment_failures_12m : int    = Field(0, ge=0)
    contract_type        : str    = "monthly"
    nps_score            : Optional[float] = None
    # ... add remaining features

class ChurnPrediction(BaseModel):
    customer_id       : str
    churn_probability : float
    risk_level        : str     # LOW / MEDIUM / HIGH
    recommended_action: str
    model_version     : str = "2.1.0"
    latency_ms        : float

class ExplainedPrediction(ChurnPrediction):
    top_risk_drivers  : List[Dict]
    top_protectors    : List[Dict]

# ── Utilities ─────────────────────────────────────────────────────────────
def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """JWT verification — swap with your auth provider in production."""
    if credentials.credentials != "your-secure-token":
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return credentials

def to_dataframe(customer: CustomerFeatures) -> pd.DataFrame:
    data = customer.dict(exclude={"customer_id"})
    return pd.DataFrame([data])

def get_action(prob: float) -> str:
    if prob > 0.75: return "Immediate outreach — assign CSM, offer retention deal"
    if prob > 0.50: return "Schedule check-in call within 7 days"
    if prob > 0.30: return "Send engagement email, offer feature training"
    return "No action required — monitor monthly"

# ── Endpoints ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}

@app.post("/predict", response_model=ChurnPrediction,
          dependencies=[Depends(verify_token)])
def predict(customer: CustomerFeatures):
    t0 = time.perf_counter()
    X  = to_dataframe(customer)
    prob = float(MODEL.predict_proba(X)[0, 1])
    risk = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW"

    PREDICT_COUNTER.inc()
    if risk == "HIGH": HIGH_RISK_COUNTER.inc()
    latency = (time.perf_counter() - t0) * 1000

    LATENCY_HIST.observe(latency / 1000)

    return ChurnPrediction(
        customer_id        = customer.customer_id,
        churn_probability  = round(prob, 4),
        risk_level         = risk,
        recommended_action = get_action(prob),
        latency_ms         = round(latency, 2),
    )

@app.post("/predict/batch", dependencies=[Depends(verify_token)])
def predict_batch(customers: List[CustomerFeatures]):
    """Batch scoring for CRM exports (max 5,000 records per call)."""
    if len(customers) > 5000:
        raise HTTPException(400, "Batch size limit is 5,000 records")
    return [predict(c) for c in customers]

@app.post("/explain", response_model=ExplainedPrediction,
          dependencies=[Depends(verify_token)])
def explain(customer: CustomerFeatures):
    """Returns prediction + top SHAP drivers for one customer."""
    pred = predict(customer)
    X    = to_dataframe(customer)
    explanation = EXPLAINER.explain_single_prediction(X, customer.customer_id)
    return ExplainedPrediction(**pred.dict(), **explanation)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080,
                workers=4, log_level="info")
'''


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — MODEL MONITORING & DRIFT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

class DriftMonitor:
    """
    Population Stability Index (PSI) for detecting feature/prediction drift.
    PSI < 0.10  → No significant change
    PSI 0.10-0.20 → Moderate change, monitor
    PSI > 0.20  → Significant drift → trigger retraining alert
    """

    @staticmethod
    def compute_psi(expected: np.ndarray, actual: np.ndarray,
                    n_bins: int = 10) -> float:
        """Compute PSI between reference and current distributions."""
        eps = 1e-6
        min_val = min(expected.min(), actual.min())
        max_val = max(expected.max(), actual.max())
        bins    = np.linspace(min_val, max_val, n_bins + 1)

        exp_pct = np.histogram(expected, bins=bins)[0] / len(expected) + eps
        act_pct = np.histogram(actual,   bins=bins)[0] / len(actual)   + eps

        psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
        return float(psi)

    def monitor_features(
        self, ref_df: pd.DataFrame, curr_df: pd.DataFrame,
        numeric_cols: List[str]
    ) -> pd.DataFrame:
        rows = []
        for col in numeric_cols:
            psi  = self.compute_psi(ref_df[col].dropna().values,
                                    curr_df[col].dropna().values)
            flag = "ALERT" if psi > 0.20 else "WARN" if psi > 0.10 else "OK"
            rows.append({"feature": col, "psi": round(psi, 4), "status": flag})

        df = pd.DataFrame(rows).sort_values("psi", ascending=False)
        n_alerts = (df["status"] == "ALERT").sum()
        if n_alerts:
            logger.warning(f"DRIFT ALERT: {n_alerts} features have PSI > 0.20 "
                           f"— consider retraining!")
        return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — AIRFLOW DAG (retraining orchestration stub)
# ─────────────────────────────────────────────────────────────────────────────

AIRFLOW_DAG_CODE = '''
"""
Airflow DAG: Weekly churn model retraining pipeline
File: dags/churn_retrain_dag.py
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.email import EmailOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

RETRAIN_THRESHOLD = 0.005   # Promote if new AUC > champion AUC + 0.5%

default_args = {
    "owner"          : "ml-team",
    "retries"        : 2,
    "retry_delay"    : timedelta(minutes=10),
    "email_on_failure": True,
    "email"          : ["ml-oncall@company.com"],
}

with DAG(
    dag_id="churn_model_retrain",
    default_args=default_args,
    schedule_interval="0 3 * * 1",   # Every Monday at 03:00 UTC
    start_date=days_ago(7),
    catchup=False,
    tags=["ml", "churn"],
) as dag:

    def fetch_data(**ctx):
        """Pull last 90 days of customer data from data warehouse."""
        pass   # replace with your DB query

    def validate_data(**ctx):
        """Run Great Expectations suite against new data."""
        pass

    def retrain(**ctx):
        """Re-run full ChurnPredictor pipeline, save challenger model."""
        pass

    def evaluate_challenger(**ctx):
        """Compare challenger AUC vs. current champion in MLflow."""
        challenger_auc = ctx["ti"].xcom_pull(task_ids="retrain", key="auc")
        champion_auc   = ctx["ti"].xcom_pull(task_ids="load_champion", key="auc")
        if challenger_auc - champion_auc >= RETRAIN_THRESHOLD:
            return "promote_model"
        return "keep_champion"

    def promote_model(**ctx):
        """Register challenger as new champion in MLflow Model Registry."""
        pass

    t1 = PythonOperator(task_id="fetch_data",          python_callable=fetch_data)
    t2 = PythonOperator(task_id="validate_data",       python_callable=validate_data)
    t3 = PythonOperator(task_id="retrain",             python_callable=retrain)
    t4 = BranchPythonOperator(task_id="evaluate",      python_callable=evaluate_challenger)
    t5 = PythonOperator(task_id="promote_model",       python_callable=promote_model)
    t6 = PythonOperator(task_id="keep_champion",       python_callable=lambda: None)

    t1 >> t2 >> t3 >> t4 >> [t5, t6]
'''


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class ChurnPredictionSystem:
    """
    Top-level orchestrator that wires all components together.
    """

    def __init__(self, random_state: int = 42, n_optuna_trials: int = 30):
        self.random_state    = random_state
        self.n_optuna_trials = n_optuna_trials
        self.models_         = {}
        self.best_model_     = None
        self.evaluator_      = ModelEvaluator()
        self.drift_monitor_  = DriftMonitor()

    def run(self):
        logger.info("=" * 60)
        logger.info("  CHURN PREDICTION SYSTEM — Full Pipeline Run")
        logger.info("=" * 60)

        # ── 1. Ingestion ──────────────────────────────────────────────────
        ingestion = DataIngestion(self.random_state)
        raw_df    = ingestion.generate_synthetic_data(n_customers=10_000)
        ingestion.validate_schema(raw_df)

        # ── 2. Feature Engineering ────────────────────────────────────────
        fe     = FeatureEngineer()
        eng_df = fe.transform(raw_df)

        # ── 3. Prepare arrays ─────────────────────────────────────────────
        TARGET    = "churn"
        DROP_COLS = [TARGET, "tenure_segment"]   # tenure_segment → categorical
        cat_cols  = ["contract_type", "region", "company_size", "tenure_segment"]
        num_cols  = [c for c in eng_df.columns
                     if c not in [TARGET] + cat_cols]

        # Re-include tenure_segment properly
        for c in cat_cols:
            if c not in eng_df.columns:
                cat_cols.remove(c)
        num_cols = [c for c in eng_df.columns
                    if c not in [TARGET] + cat_cols]

        X = eng_df.drop(columns=[TARGET])
        y = eng_df[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=self.random_state)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.125, stratify=y_train,
            random_state=self.random_state)  # 70/10/20 split

        logger.info(f"Train: {len(X_train):,}  Val: {len(X_val):,}  "
                    f"Test: {len(X_test):,}")

        # ── 4. Build Preprocessor ─────────────────────────────────────────
        preprocessor = build_preprocessor(
            num_cols=[c for c in num_cols if c in X_train.columns],
            cat_cols=[c for c in cat_cols if c in X_train.columns],
        )

        # ── 5. Train Models ───────────────────────────────────────────────
        trainer = ModelTrainer(self.random_state, self.n_optuna_trials)

        import copy
        lr_pipe  = trainer.train_logistic_regression(
                       X_train, y_train, copy.deepcopy(preprocessor))
        rf_pipe  = trainer.train_random_forest(
                       X_train, y_train, copy.deepcopy(preprocessor))
        gbm_pipe = trainer.train_gradient_boosting(
                       X_train, y_train, copy.deepcopy(preprocessor))

        self.models_ = {
            "LogisticRegression": {"pipeline": lr_pipe},
            "RandomForest"      : {"pipeline": rf_pipe},
            "GradientBoosting"  : {"pipeline": gbm_pipe},
        }

        # ── 6. Evaluate ───────────────────────────────────────────────────
        results_df = self.evaluator_.evaluate_all(
            self.models_, X_test, y_test)

        best_name     = results_df.iloc[0]["Model"]
        self.best_model_ = self.models_[best_name]["pipeline"]
        logger.info(f"\n★ Best model: {best_name}")

        self.evaluator_.print_classification_report(
            self.best_model_, X_test, y_test, best_name)

        # ── 7. Optimal threshold ──────────────────────────────────────────
        opt_thr = self.evaluator_.find_optimal_threshold(
            self.best_model_, X_val, y_val)

        # ── 8. SHAP Explainability ────────────────────────────────────────
        xgb_info = trainer.results_.get("GradientBoosting", {})
        if "xgb_model" in xgb_info:
            num_fitted = [c for c in num_cols if c in X_train.columns]
            explainer = ExplainabilityModule(
                xgb_model    = xgb_info["xgb_model"],
                preprocessor = xgb_info["preprocessor"],
                feature_names= num_fitted,
            )
            explainer.build_explainer(X_test, n_samples=100)

            # Global importance
            fi_df = explainer.global_feature_importance(X_test, n_features=15)
            logger.info(f"\nTop 5 SHAP features:\n"
                        f"{fi_df.head(5).to_string(index=False)}")

            # Single-customer explanation
            sample_customer = X_test.iloc[[0]]
            explanation     = explainer.explain_single_prediction(
                sample_customer, "CUST-00001")
         

            def clean(obj):
              if isinstance(obj, dict):
                 return {k: clean(v) for k, v in obj.items()}
              elif isinstance(obj, list):
                 return [clean(v) for v in obj]
              elif isinstance(obj, np.generic):
                 return obj.item()
              return obj


            logger.info(f"\nSample explanation:\n{json.dumps(clean(explanation), indent=2)}")
        # ── 9. Drift Monitoring Demo ──────────────────────────────────────
        # Simulate slight drift in test data (in prod, compare to ref period)
        X_current = X_test.copy()
        X_current["days_since_login"] *= 1.3    # simulate disengagement spike
        drift_df = self.drift_monitor_.monitor_features(
            X_test, X_current,
            numeric_cols=["days_since_login", "monthly_spend",
                          "support_tickets_90d", "logins_per_month"])
        logger.info(f"\nDrift Report:\n{drift_df.to_string(index=False)}")

        # ── 10. Save artefacts ────────────────────────────────────────────
        Path("models").mkdir(exist_ok=True)
        ModelPersistence.save(self.best_model_, "models/best_model.pkl")
        logger.info("\nPipeline complete ✓  Model saved to models/best_model.pkl")

        return {
            "best_model"  : best_name,
            "results"     : results_df,
            "threshold"   : opt_thr,
        }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    system = ChurnPredictionSystem(random_state=42, n_optuna_trials=20)
    output = system.run()
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(output["results"].to_string(index=False))
    print(f"\nOptimal Decision Threshold : {output['threshold']:.3f}")
    print(f"Best Model                 : {output['best_model']}")
    print("="*60)
