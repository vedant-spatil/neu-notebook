"""Walk-forward evaluation, sanity checks and artifact saving (triage_lib.triage_train, triage_features)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from triage_lib.triage_features import (
    FEATURES_FILE,
    MANIFEST_FILE,
    build_features,
    future_perturbation_diff,
    write_features,
)
from triage_lib.triage_train import (
    fold_plan,
    flag_top,
    label_shuffle_check,
    select_point,
    walk_forward,
)

DAY0 = pd.Timestamp("2022-09-01")


def _random_frame(seed: int, n: int, days: int, accounts: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frm = rng.integers(0, accounts, n)
    to = (frm + rng.integers(1, accounts, n)) % accounts
    ts = DAY0 + pd.to_timedelta(rng.uniform(0, days * 24 - 1, n), unit="h")
    amt = rng.uniform(10, 9000, n).round(2)
    return pd.DataFrame({
        "ts": pd.Series(ts).dt.floor("min"),
        "from_account": [f"A{i}" for i in frm],
        "to_account": [f"A{i}" for i in to],
        "amount_received": amt,
        "amount_paid": amt,
        "receiving_currency": "US Dollar",
        "payment_currency": "US Dollar",
        "payment_format": rng.choice(["ACH", "Cash", "Wire"], n),
        "is_laundering": (rng.random(n) < 0.05).astype(int),
        "txn_id": np.arange(1, n + 1),
    })


def test_fold_plan_geometry_matches_hi_large_span():
    # HI-Large: 97 days, horizon 7 -> max_t 90. Four folds + final, 7-day test and validation blocks.
    plans = fold_plan(max_t=90, horizon=7, folds=4, test_days=7, val_days=7)
    assert [p["name"] for p in plans] == ["fold1", "fold2", "fold3", "fold4", "final"]
    assert plans[-1]["test_end"] == 90
    for p in plans:
        assert p["val_start"] == p["train_end"] + 7 + 1  # purge gap before validation
        assert p["test_start"] == p["val_end"] + 7 + 1  # purge gap before test
        assert p["val_end"] - p["val_start"] + 1 == 7
        assert p["test_end"] - p["test_start"] + 1 == 7
    tests = [(p["test_start"], p["test_end"]) for p in plans]
    assert all(b[0] > a[1] for a, b in zip(tests, tests[1:]))  # test blocks do not overlap


def test_fold_plan_rejects_span_that_is_too_short():
    with pytest.raises(ValueError, match="does not fit"):
        fold_plan(max_t=16, horizon=2, folds=4, test_days=2, val_days=2)


def test_select_point_uses_mean_validation_score():
    fold_a = [{"max_depth": 4, "eta": 0.05, "min_child_weight": 1, "val_aucpr": 0.30},
              {"max_depth": 6, "eta": 0.05, "min_child_weight": 1, "val_aucpr": 0.10}]
    fold_b = [{"max_depth": 4, "eta": 0.05, "min_child_weight": 1, "val_aucpr": 0.10},
              {"max_depth": 6, "eta": 0.05, "min_child_weight": 1, "val_aucpr": 0.40}]
    # Means: depth 4 -> 0.20, depth 6 -> 0.25, so depth 6 wins.
    assert select_point([fold_a, fold_b]) == {"max_depth": 6, "eta": 0.05, "min_child_weight": 1}


def test_flag_top_is_deterministic_and_respects_budget():
    df = pd.DataFrame({"t": [1] * 6, "score": [0.0] * 6, "y": [0, 1, 0, 0, 1, 0]})
    first = flag_top(df, "score", budget=2)
    second = flag_top(df, "score", budget=2)
    assert first.tolist() == second.tolist()
    assert int(first.sum()) == 2


def test_label_shuffle_check_falls_to_base_rate():
    rng = np.random.default_rng(1)
    n = 20_000
    df = pd.DataFrame({"t": rng.integers(1, 30, n), "y": (rng.random(n) < 0.02).astype(int),
                       "score": rng.random(n)})
    out = label_shuffle_check(df, "score")
    assert abs(out["pr_auc_shuffled"] - out["base_rate"]) < 0.01


def test_future_changes_do_not_change_past_features():
    df = _random_frame(seed=11, n=600, days=12, accounts=30)
    assert future_perturbation_diff(df, horizon=2, cutoff_day=6) == 0.0


def test_walk_forward_end_to_end_on_synthetic_span(tmp_path, monkeypatch):
    # Two grid points only: this checks the plumbing (folds, selection, final model, artifacts), not the grid.
    import triage_lib.triage_train as tt
    monkeypatch.setattr(tt, "GRID", tt.GRID[:2])
    df = _random_frame(seed=3, n=12_000, days=40, accounts=3_000)
    frame, manifest = build_features(df, horizon=2)
    features_dir = tmp_path / "features"
    write_features(frame, manifest, features_dir)
    max_t = int(frame["t"].max())
    out_dir = tmp_path / "eval"
    model_dir = tmp_path / "models"

    result = walk_forward(
        path=features_dir / FEATURES_FILE,
        features=manifest["features"],
        horizon=2,
        max_t=max_t,
        folds=1,
        test_days=3,
        val_days=3,
        budget=5,
        out_dir=out_dir,
        model_dir=model_dir,
        manifest_path=features_dir / MANIFEST_FILE,
        neg_ratio=None,
    )
    assert len(result["folds"]) == 1
    assert result["final"]["name"] == "final"
    assert set(result["summary"]["model"]) == {"precision_at_budget", "recall_at_budget", "pr_auc", "lift"}
    assert (out_dir / "walk_forward.json").exists()
    for name in ("model.json", MANIFEST_FILE, "versions.json"):
        assert (model_dir / name).exists()
    versions = json.loads((model_dir / "versions.json").read_text())
    assert len(versions["features_sha256"]) == 64
