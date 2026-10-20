"""Reason codes for triage scores: copied from backend/app/ml/triage_score.py (reason_codes only).

Top features by XGBoost's own per-row contributions (pred_contribs, TreeSHAP). No extra dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

REASONS = 3


def reason_codes(booster: xgb.Booster, df: pd.DataFrame, features: list[str], top_n: int = REASONS) -> list[list[str]]:
    dm = xgb.DMatrix(df[features].to_numpy(np.float32), feature_names=features)
    contrib = booster.predict(dm, pred_contribs=True, iteration_range=(0, booster.best_iteration + 1))[:, :-1]
    order = np.argsort(-contrib, axis=1)[:, :top_n]
    picked = np.take_along_axis(contrib, order, axis=1)
    names = np.asarray(features)
    return [[str(names[j]) for j, v in zip(row_idx, row_val) if v > 0] for row_idx, row_val in zip(order, picked)]
