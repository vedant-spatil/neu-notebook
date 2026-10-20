"""Train and evaluate the XGBoost triage model (plan.md section 24). Reads the output of triage_features.

Split: by window-end day t. After each cut, a gap of HORIZON days is dropped so no label window
crosses from one split into the next. Negatives are downsampled for training only; val and test
keep the natural laundering rate.

Gate metrics are at a fixed alert budget: the top BUDGET account-days per day are flagged. The
baseline is the app's rule score (`rule_score_app`, time-gated), scored the same way.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

from triage_lib.triage_features import FEATURES_FILE, MANIFEST_FILE

NEG_RATIO = 10
ROUNDS = 1000
EARLY_STOP = 50
SEED = 7
GRID = [{"max_depth": d, "eta": e, "min_child_weight": m} for d, e, m in product((4, 6, 8), (0.05, 0.1), (1, 5))]
BASE_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": SEED,
}
PATTERN_HEADER = "BEGIN LAUNDERING ATTEMPT - "
PATTERN_TS = "%Y/%m/%d %H:%M"


def assign_splits(t: np.ndarray, horizon: int, train_end: int, val_end: int) -> np.ndarray:
    """train: t <= train_end. val: train_end + horizon < t <= val_end. test: t > val_end + horizon. Else gap."""
    t = np.asarray(t)
    split = np.full(t.shape, "gap", dtype=object)
    split[t <= train_end] = "train"
    split[(t > train_end + horizon) & (t <= val_end)] = "val"
    split[t > val_end + horizon] = "test"
    return split


def default_cuts(max_t: int, horizon: int) -> tuple[int, int]:
    """60% train, 20% val, 20% test by day, with the horizon gap after each cut."""
    train_end = int(max_t * 0.6)
    val_end = int(max_t * 0.8)
    if val_end - (train_end + horizon) < 1 or max_t - (val_end + horizon) < 1:
        raise ValueError(f"not enough days for val and test with horizon {horizon}; pass --train-end and --val-end")
    return train_end, val_end


def downsample(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    pos = df[df["y"] == 1]
    neg = df[df["y"] == 0]
    if pos.empty:
        raise ValueError("no positive rows in the training split")
    keep = min(len(neg), NEG_RATIO * len(pos))
    return pd.concat([pos, neg.sample(n=keep, random_state=seed)], ignore_index=True)


def flag_top(df: pd.DataFrame, score_col: str, budget: int) -> pd.Series:
    """True for the top `budget` rows of each day t (ties keep file order)."""
    order = df.sort_values(["t", score_col], ascending=[True, False], kind="mergesort")
    rank = order.groupby("t").cumcount().to_numpy()
    return pd.Series(rank < budget, index=order.index).reindex(df.index)


def budget_metrics(df: pd.DataFrame, score_col: str, budget: int) -> dict[str, Any]:
    flagged = flag_top(df, score_col, budget).to_numpy()
    y = df["y"].to_numpy()
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    positives = int(y.sum())
    return {
        "budget_per_day": budget,
        "days": int(df["t"].nunique()),
        "flagged": int(flagged.sum()),
        "tp": tp,
        "fp": fp,
        "positives": positives,
        "precision_at_budget": round(tp / (tp + fp), 5) if tp + fp else 0.0,
        "recall_at_budget": round(tp / positives, 5) if positives else 0.0,
        "pr_auc": round(float(average_precision_score(y, df[score_col])), 5) if positives else None,
    }


def predict(booster: xgb.Booster, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    dm = xgb.DMatrix(df[features].to_numpy(np.float32), feature_names=features)
    return booster.predict(dm, iteration_range=(0, booster.best_iteration + 1))


def fit_best(train: pd.DataFrame, val: pd.DataFrame, features: list[str]) -> tuple[tuple[float, dict, xgb.Booster], list[dict]]:
    """Early-stop each grid point on validation PR-AUC; keep the best."""
    dtr = xgb.DMatrix(train[features].to_numpy(np.float32), label=train["y"].to_numpy(), feature_names=features)
    dva = xgb.DMatrix(val[features].to_numpy(np.float32), label=val["y"].to_numpy(), feature_names=features)
    best: tuple[float, dict, xgb.Booster] | None = None
    runs: list[dict] = []
    for point in GRID:
        params = {**BASE_PARAMS, **point}
        booster = xgb.train(params, dtr, num_boost_round=ROUNDS, evals=[(dva, "val")],
                            early_stopping_rounds=EARLY_STOP, verbose_eval=False)
        score = float(booster.best_score)
        runs.append({**point, "val_aucpr": round(score, 5), "best_iteration": int(booster.best_iteration)})
        if best is None or score > best[0]:
            best = (score, params, booster)
    assert best is not None
    return best, runs


def parse_patterns(path: Path, day0: pd.Timestamp) -> pd.DataFrame:
    """HI-*_Patterns.txt: BEGIN/END blocks of laundering transactions. Returns (acct, day, typology) for both sides."""
    rows: list[tuple[str, int, str]] = []
    typ: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(PATTERN_HEADER):
            typ = line[len(PATTERN_HEADER):].split(":", 1)[0].strip()
            continue
        if line.startswith("END"):
            typ = None
            continue
        if typ is None or "," not in line:
            continue
        fields = line.split(",")
        day = int((pd.to_datetime(fields[0], format=PATTERN_TS).normalize() - day0).days) + 1
        for acct in (fields[2], fields[4]):
            rows.append((acct, day, typ))
    return pd.DataFrame(rows, columns=["acct", "day", "typology"]).drop_duplicates()


def typology_recall(test: pd.DataFrame, flagged: pd.Series, patterns: pd.DataFrame, horizon: int) -> dict[str, Any]:
    """Recall per typology: share of (account, t) rows in the test split with a pattern transaction in
    t+1..t+horizon that were flagged at the budget."""
    pos = pd.concat([patterns.assign(t=patterns["day"] - k) for k in range(1, horizon + 1)], ignore_index=True)
    pos = pos[["acct", "t", "typology"]].drop_duplicates()
    keyed = test[["acct", "t"]].assign(flagged=flagged.to_numpy())
    hits = pos.merge(keyed, on=["acct", "t"], how="inner")
    total = hits.groupby("typology").size()
    caught = hits[hits["flagged"]].groupby("typology").size()
    return {typ: {"positives": int(n), "flagged": int(caught.get(typ, 0)),
                  "recall_at_budget": round(float(caught.get(typ, 0)) / n, 5)}
            for typ, n in total.items()}


def run_train(features_dir: Path, run_dir: Path, budget: int, train_end: int | None = None,
              val_end: int | None = None, patterns_path: Path | None = None) -> dict[str, Any]:
    manifest = json.loads((features_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    frame = pd.read_parquet(features_dir / FEATURES_FILE)
    features: list[str] = manifest["features"]
    horizon: int = manifest["horizon"]
    day0 = pd.Timestamp(manifest["day0"])

    if train_end is None or val_end is None:
        train_end, val_end = default_cuts(int(frame["t"].max()), horizon)
    frame["split"] = assign_splits(frame["t"].to_numpy(), horizon, train_end, val_end)
    train = downsample(frame[frame["split"] == "train"])
    val = frame[frame["split"] == "val"].copy()
    test = frame[frame["split"] == "test"].copy()
    if val.empty or test.empty or val["y"].sum() == 0 or test["y"].sum() == 0:
        raise ValueError("val or test split has no rows or no positives; adjust --train-end / --val-end")

    (best_val, best_params, booster), runs = fit_best(train, val, features)
    run_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(run_dir / "model.json"))

    val["score"] = predict(booster, val, features)
    test["score"] = predict(booster, test, features)
    test["rule_app"] = test["rule_score_app"]

    # Threshold for the app: the score that flags about `budget` account-days per day on validation.
    per_day = val.groupby("t").size()
    frac = float(np.minimum(1.0, budget / per_day).mean())
    suggested_threshold = float(np.quantile(val["score"], 1 - frac))

    flagged = flag_top(test, "score", budget)
    typology = {}
    if patterns_path is not None:
        typology = typology_recall(test, flagged, parse_patterns(patterns_path, day0), horizon)

    importance = booster.get_score(importance_type="gain")
    top = dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:20])
    metrics = {
        "horizon": horizon,
        "train_end": train_end,
        "val_end": val_end,
        "budget_per_day": budget,
        "neg_ratio": NEG_RATIO,
        "rows": {"train_after_downsample": len(train), "val": len(val), "test": len(test),
                 "gap": int((frame["split"] == "gap").sum())},
        "positives": {"train": int(train["y"].sum()), "val": int(val["y"].sum()), "test": int(test["y"].sum())},
        "grid": runs,
        "best_params": {k: best_params[k] for k in GRID[0]},
        "best_iteration": int(booster.best_iteration),
        "val_aucpr": round(best_val, 5),
        "suggested_threshold_val": round(suggested_threshold, 5),
        "test": {
            "model": budget_metrics(test, "score", budget),
            "rules_app": budget_metrics(test, "rule_app", budget),
        },
        "typology_recall_test": typology,
        "top_features_gain": {k: round(float(v), 3) for k, v in top.items()},
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
