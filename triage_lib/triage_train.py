"""Train and evaluate the XGBoost triage model. Reads the output of triage_features (local or Spark).

Two evaluation modes:
- single split (run_train): one train / validation / test split. Used for HI-Small, which has too few days
  for walk-forward folds.
- walk-forward (walk_forward): several rounds, each training on earlier days, validating on a block
  just before its test block, and testing on the next block. A final block is held out last. The
  grid is chosen on validation only; the final block is scored once.

Every cut is followed by a gap of HORIZON days, so no label window crosses a split. Negatives are
downsampled for training only; validation and test keep the natural laundering rate.

Gate metrics are at a fixed alert budget: the top BUDGET account-days per day are flagged. Ties are
broken by a seeded random key, so a rule score full of zeros does not depend on file order.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.dataset as ds
import xgboost as xgb
from sklearn.metrics import average_precision_score

from triage_lib.triage_features import FEATURES_FILE, MANIFEST_FILE

NEG_RATIO = 10
ROUNDS = 1000
EARLY_STOP = 50
SEED = 7
TIE_SEED = SEED
MIN_CONCLUSIVE_POSITIVES = 200  # a test block with fewer positives is reported as not conclusive
VAL_NEG_RATIO = 100  # validation keeps all positives and ~100 negatives per positive (early stopping only)
GRID = [{"max_depth": d, "eta": e, "min_child_weight": m} for d, e, m in product((4, 6, 8), (0.05, 0.1), (1, 5))]
GRID_KEYS = ["max_depth", "eta", "min_child_weight"]
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


def fold_plan(max_t: int, horizon: int, folds: int, test_days: int, val_days: int) -> list[dict[str, Any]]:
    """Walk-forward blocks, oldest first. The last entry is the final held-out block.

    For each block: test = [test_start, test_end]; validation ends horizon days before the test
    block and is val_days long; training ends horizon days before the validation block and starts
    on day 1. Raises if the span is too short for any block.
    """
    final_start = max_t - test_days + 1
    plans: list[dict[str, Any]] = []
    for i in range(folds + 1):
        test_start = final_start - (folds - i) * test_days
        test_end = test_start + test_days - 1
        val_end = test_start - horizon - 1
        val_start = val_end - val_days + 1
        train_end = val_start - horizon - 1
        if train_end < 1:
            raise ValueError(
                f"block {i + 1} does not fit: span is days 1..{max_t}, horizon {horizon}, "
                f"test {test_days} days, validation {val_days} days, {folds} folds. "
                "Use fewer folds or shorter blocks."
            )
        name = "final" if i == folds else f"fold{i + 1}"
        plans.append({"name": name, "train_end": train_end, "val_start": val_start, "val_end": val_end,
                      "test_start": test_start, "test_end": test_end})
    return plans


def downsample(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    pos = df[df["y"] == 1]
    neg = df[df["y"] == 0]
    if pos.empty:
        raise ValueError("no positive rows in the training split")
    keep = min(len(neg), NEG_RATIO * len(pos))
    return pd.concat([pos, neg.sample(n=keep, random_state=seed)], ignore_index=True)


def load_block(path: Path, lo: int, hi: int, neg_ratio: int | None = None, seed: int = SEED) -> pd.DataFrame:
    """Rows with lo <= t <= hi from a Parquet file or a folder (partitioned by t, or flat).

    neg_ratio keeps every positive and about neg_ratio negatives per positive. Sampling reads one day at a
    time, so a large block never has to fit in memory at once. Use it for training and validation blocks.
    """
    dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
    in_range = (ds.field("t") >= lo) & (ds.field("t") <= hi)
    if neg_ratio is None:
        return dataset.to_table(filter=in_range).to_pandas().reset_index(drop=True)
    n_pos = dataset.count_rows(filter=in_range & (ds.field("y") == 1))
    n_neg = dataset.count_rows(filter=in_range & (ds.field("y") == 0))
    if n_pos == 0:
        raise ValueError(f"no positive rows in days {lo}..{hi}")
    keep_frac = min(1.0, neg_ratio * n_pos / max(n_neg, 1))
    rng = np.random.default_rng(seed)
    parts = []
    for day in range(lo, hi + 1):
        block = dataset.to_table(filter=ds.field("t") == day).to_pandas()
        if block.empty:
            continue
        keep = (block["y"].to_numpy() == 1) | (rng.random(len(block)) < keep_frac)
        parts.append(block[keep])
    if not parts:
        raise ValueError(f"no rows in days {lo}..{hi}")
    return pd.concat(parts, ignore_index=True)


def flag_top(df: pd.DataFrame, score_col: str, budget: int, seed: int = TIE_SEED) -> pd.Series:
    """True for the top `budget` rows of each day t. Ties are broken by a seeded random key."""
    tie = np.random.default_rng(seed).random(len(df))
    order = df.assign(_tie=tie).sort_values(["t", score_col, "_tie"], ascending=[True, False, True],
                                            kind="mergesort")
    rank = order.groupby("t").cumcount().to_numpy()
    return pd.Series(rank < budget, index=order.index).reindex(df.index)


def budget_metrics(df: pd.DataFrame, score_col: str, budget: int, seed: int = TIE_SEED) -> dict[str, Any]:
    flagged = flag_top(df, score_col, budget, seed).to_numpy()
    y = df["y"].to_numpy()
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    positives = int(y.sum())
    fn = positives - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    base_rate = positives / len(y) if len(y) else 0.0
    return {
        "budget_per_day": budget,
        "days": int(df["t"].nunique()),
        "flagged": int(flagged.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "positives": positives,
        "precision_at_budget": round(precision, 5),
        "recall_at_budget": round(tp / positives, 5) if positives else 0.0,
        "base_rate": round(base_rate, 6),
        "lift": round(precision / base_rate, 2) if base_rate else None,
        "pr_auc": round(float(average_precision_score(y, df[score_col])), 5) if positives else None,
    }


def predict(booster: xgb.Booster, df: pd.DataFrame, features: list[str],
            iteration_range: tuple[int, int] | None = None) -> np.ndarray:
    """Scores. Default: up to the early-stopped round (booster.best_iteration)."""
    if iteration_range is None:
        iteration_range = (0, booster.best_iteration + 1)
    dm = xgb.DMatrix(df[features].to_numpy(np.float32), feature_names=features)
    return booster.predict(dm, iteration_range=iteration_range)


def predict_chunked(booster: xgb.Booster, df: pd.DataFrame, features: list[str],
                    iteration_range: tuple[int, int] | None = None, chunk: int = 2_000_000) -> np.ndarray:
    parts = [predict(booster, df.iloc[s:s + chunk], features, iteration_range) for s in range(0, len(df), chunk)]
    return np.concatenate(parts) if parts else np.array([], dtype=float)


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


def select_point(grids: list[list[dict]]) -> dict[str, Any]:
    """Grid point with the best mean validation PR-AUC across folds."""
    frame = pd.concat([pd.DataFrame(g) for g in grids], ignore_index=True)
    mean = frame.groupby(GRID_KEYS)["val_aucpr"].mean().sort_values(ascending=False)
    return {k: v.item() if hasattr(v, "item") else v for k, v in zip(GRID_KEYS, mean.index[0])}


def label_shuffle_check(df: pd.DataFrame, score_col: str, seed: int = SEED) -> dict[str, float]:
    """Labels permuted within each day. PR-AUC should fall to the base rate; if not, something leaks."""
    rng = np.random.default_rng(seed)
    shuffled = df.groupby("t")["y"].transform(lambda s: pd.Series(rng.permutation(s.to_numpy()), index=s.index))
    return {
        "pr_auc_shuffled": round(float(average_precision_score(shuffled, df[score_col])), 5),
        "base_rate": round(float(df["y"].mean()), 6),
    }


def save_artifact(booster: xgb.Booster, out_dir: Path, manifest_path: Path) -> dict[str, str]:
    """model.json, the feature manifest it was trained with, and the library versions. The scorer needs all three."""
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_dir / "model.json"))
    shutil.copy(manifest_path, out_dir / MANIFEST_FILE)
    features = json.loads(manifest_path.read_text(encoding="utf-8"))["features"]
    versions = {
        "xgboost": xgb.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "pyarrow": pyarrow.__version__,
        "features_sha256": hashlib.sha256(json.dumps(features).encode("utf-8")).hexdigest(),
    }
    (out_dir / "versions.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
    return versions


def score_days(path: Path, lo: int, hi: int, features: list[str], booster: xgb.Booster,
               iteration_range: tuple[int, int] | None = None) -> pd.DataFrame:
    """Scores days lo..hi one day at a time, so a long test block never sits in memory at once.
    Returns acct, t, y, score and rule_app (the baseline) for every row."""
    parts = []
    for day in range(lo, hi + 1):
        block = load_block(path, day, day)
        if block.empty:
            continue
        block["score"] = predict_chunked(booster, block, features, iteration_range)
        block["rule_app"] = block["rule_score_app"].to_numpy()
        parts.append(block[["acct", "t", "y", "score", "rule_app"]])
    if not parts:
        return pd.DataFrame(columns=["acct", "t", "y", "score", "rule_app"])
    return pd.concat(parts, ignore_index=True)


def _score_test(path: Path, lo: int, hi: int, features: list[str], booster: xgb.Booster, budget: int,
                iteration_range: tuple[int, int] | None = None) -> tuple[dict, dict, pd.DataFrame]:
    scored = score_days(path, lo, hi, features, booster, iteration_range)
    return budget_metrics(scored, "score", budget), budget_metrics(scored, "rule_app", budget), scored


def run_block(path: Path, features: list[str], horizon: int, plan: dict[str, Any], budget: int,
              neg_ratio: int | None = NEG_RATIO, val_neg_ratio: int = VAL_NEG_RATIO) -> dict[str, Any]:
    """One walk-forward round: grid on validation, then the test block scored day by day with the early-stopped model.

    Validation is negative-downsampled. Early stopping ranks scores, so the comparison between grid points holds,
    but the validation PR-AUC is not a precision estimate. Test scores are on the full natural rate.
    """
    train = load_block(path, 1, plan["train_end"], neg_ratio=neg_ratio)
    val = load_block(path, plan["val_start"], plan["val_end"], neg_ratio=val_neg_ratio)
    if int(val["y"].sum()) == 0:
        raise ValueError(f"{plan['name']}: no positives in the validation block")
    (best_val, _params, booster), runs = fit_best(train, val, features)
    model_m, rules_m, scored = _score_test(path, plan["test_start"], plan["test_end"], features, booster, budget)
    test_positives = int(scored["y"].sum())
    return {
        "name": plan["name"],
        "plan": plan,
        "rows": {"train": len(train), "val_downsampled": len(val), "test": len(scored)},
        "positives": {"train": int(train["y"].sum()), "val": int(val["y"].sum()), "test": test_positives},
        "grid": runs,
        "best_iteration": int(booster.best_iteration),
        "val_aucpr": round(best_val, 5),
        "model": model_m,
        "rules_app": rules_m,
        "conclusive": test_positives >= MIN_CONCLUSIVE_POSITIVES,
    }


def run_final(path: Path, features: list[str], plan: dict[str, Any], selected: dict[str, Any], budget: int,
              model_dir: Path, manifest_path: Path, horizon: int, day0: pd.Timestamp | None = None,
              patterns_path: Path | None = None, neg_ratio: int | None = NEG_RATIO,
              val_neg_ratio: int = VAL_NEG_RATIO) -> dict[str, Any]:
    """Final model: early-stop with the selected parameters on the final validation block, then retrain on all
    development days for the same number of rounds, and score the final block once (day by day)."""
    params = {**BASE_PARAMS, **selected}
    train = load_block(path, 1, plan["train_end"], neg_ratio=neg_ratio)
    val = load_block(path, plan["val_start"], plan["val_end"], neg_ratio=val_neg_ratio)
    dtr = xgb.DMatrix(train[features].to_numpy(np.float32), label=train["y"].to_numpy(), feature_names=features)
    dva = xgb.DMatrix(val[features].to_numpy(np.float32), label=val["y"].to_numpy(), feature_names=features)
    probe = xgb.train(params, dtr, num_boost_round=ROUNDS, evals=[(dva, "val")],
                      early_stopping_rounds=EARLY_STOP, verbose_eval=False)
    rounds = int(probe.best_iteration) + 1
    del dtr, dva, train, val

    dev = load_block(path, 1, plan["val_end"], neg_ratio=neg_ratio)
    ddev = xgb.DMatrix(dev[features].to_numpy(np.float32), label=dev["y"].to_numpy(), feature_names=features)
    final_model = xgb.train(params, ddev, num_boost_round=rounds)
    dev_rows, dev_pos = len(dev), int(dev["y"].sum())
    del ddev, dev

    model_m, rules_m, scored = _score_test(path, plan["test_start"], plan["test_end"], features, final_model,
                                          budget, iteration_range=(0, rounds))
    save_artifact(final_model, model_dir, manifest_path)

    typology: dict[str, Any] = {}
    if patterns_path is not None and day0 is not None:
        flagged = flag_top(scored, "score", budget)
        typology = typology_recall(scored, flagged, parse_patterns(patterns_path, day0), horizon)
    test_positives = int(scored["y"].sum())
    return {
        "name": plan["name"],
        "plan": plan,
        "selected_params": selected,
        "rounds": rounds,
        "val_aucpr_probe": round(float(probe.best_score), 5),
        "rows": {"dev": dev_rows, "test": len(scored)},
        "positives": {"dev": dev_pos, "test": test_positives},
        "model": model_m,
        "rules_app": rules_m,
        "typology_recall_test": typology,
        "conclusive": test_positives >= MIN_CONCLUSIVE_POSITIVES,
    }


def _spread(values: list[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {"mean": round(float(arr.mean()), 5), "std": round(float(arr.std()), 5),
            "min": round(float(arr.min()), 5), "max": round(float(arr.max()), 5)}


def summarize(folds: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("model", "rules_app"):
        out[key] = {m: _spread([f[key][m] for f in folds])
                    for m in ("precision_at_budget", "recall_at_budget", "pr_auc", "lift")}
    out["fold_test_positives"] = [f["positives"]["test"] for f in folds]
    out["folds_conclusive"] = sum(bool(f["conclusive"]) for f in folds)
    return out


def walk_forward(path: Path, features: list[str], horizon: int, max_t: int, folds: int, test_days: int,
                 val_days: int, budget: int, out_dir: Path, model_dir: Path, manifest_path: Path,
                 neg_ratio: int | None = NEG_RATIO, day0: pd.Timestamp | None = None,
                 patterns_path: Path | None = None, val_neg_ratio: int = VAL_NEG_RATIO) -> dict[str, Any]:
    """Runs the folds, picks the grid point by mean validation PR-AUC, then trains and scores the final block.

    Writes out_dir/walk_forward.json and saves the final model to model_dir (model.json + manifest + versions).
    """
    plans = fold_plan(max_t, horizon, folds, test_days, val_days)
    fold_results = [run_block(path, features, horizon, p, budget, neg_ratio, val_neg_ratio) for p in plans[:-1]]
    selected = select_point([f["grid"] for f in fold_results])
    final = run_final(path, features, plans[-1], selected, budget, model_dir, manifest_path, horizon,
                      day0=day0, patterns_path=patterns_path, neg_ratio=neg_ratio, val_neg_ratio=val_neg_ratio)
    result = {
        "config": {"horizon": horizon, "max_t": max_t, "folds": folds, "test_days": test_days,
                   "val_days": val_days, "budget_per_day": budget, "neg_ratio": neg_ratio,
                   "val_neg_ratio": val_neg_ratio, "min_conclusive_positives": MIN_CONCLUSIVE_POSITIVES},
        "plans": plans,
        "folds": fold_results,
        "selected_params": selected,
        "summary": summarize(fold_results),
        "final": final,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "walk_forward.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


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
    """Single split. For HI-Small (too few days for walk-forward folds)."""
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
    shutil.copy(features_dir / MANIFEST_FILE, run_dir / MANIFEST_FILE)

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
