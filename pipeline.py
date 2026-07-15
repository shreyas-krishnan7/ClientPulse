"""
ClientPulse — analytics + modelling pipeline.

Three responsibilities:
  1. load_data       : read the account table
  2. compute_kpis    : derive product KPIs (adoption, utilisation, health score)
  3. train_churn_model : train + validate a LightGBM churn model, score all accounts
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

TOTAL_FEATURES = 12

FEATURE_COLS = [
    "industry", "plan_tier", "mrr", "tenure_months", "seats_purchased",
    "active_seats_last_30d", "logins_last_30d", "features_adopted",
    "avg_session_min", "support_tickets_last_90d", "unresolved_tickets",
    "avg_resolution_days", "nps_last_survey",
]
CATEGORICAL = ["industry", "plan_tier"]


def load_data(path: str = "data/accounts.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the product metrics a PM actually tracks."""
    df = df.copy()
    df["adoption_rate"] = df["features_adopted"] / TOTAL_FEATURES
    df["seat_utilization"] = (df["active_seats_last_30d"] / df["seats_purchased"]).clip(0, 1)
    df["login_intensity"] = (df["logins_last_30d"] / 30).clip(0, 1)  # stickiness proxy

    # composite health score (0-100), weighted across engagement + support + sentiment
    df["health_score"] = (
        0.35 * df["adoption_rate"]
        + 0.25 * df["seat_utilization"]
        + 0.20 * df["login_intensity"]
        + 0.10 * (df["nps_last_survey"].clip(0, 10) / 10)
        + 0.10 * (1 - (df["unresolved_tickets"] / (df["support_tickets_last_90d"] + 1)))
    ) * 100
    df["health_score"] = df["health_score"].round(1)
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLS].copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    return X


def train_churn_model(df: pd.DataFrame, seed: int = 42):
    """Returns (model, validation_auc, scored_df, splits)."""
    X = prepare_features(df)
    y = df["churned"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        subsample=0.9, colsample_bytree=0.9, random_state=seed, verbose=-1,
    )
    model.fit(X_train, y_train, categorical_feature=CATEGORICAL)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    scored = df.copy()
    scored["churn_risk"] = model.predict_proba(X)[:, 1]
    return model, auc, scored, (X_train, X_test, y_train, y_test)
