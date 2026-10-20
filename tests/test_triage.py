"""XGBoost triage: features, labels, splits, budget metrics, typology parsing, scoring (plan.md section 24)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from triage_lib.triage_features import FEATURES_FILE, LOOKBACK_DAYS, build_features, run_features
from triage_lib.triage_train import (
    assign_splits,
    budget_metrics,
    default_cuts,
    downsample,
    parse_patterns,
    typology_recall,
    flag_top,
)
from triage_lib.explain import reason_codes

DAY0 = pd.Timestamp("2022-09-01")


def _frame(rows: list[tuple]) -> pd.DataFrame:
    """rows: (day, from, to, amount, is_laundering)."""
    return pd.DataFrame({
        "ts": [DAY0 + pd.Timedelta(days=d - 1, hours=12) for d, *_ in rows],
        "from_account": [f for _, f, *_ in rows],
        "to_account": [t for _, _, t, *_ in rows],
        "amount_received": [float(a) for *_, a, _ in rows],
        "amount_paid": [float(a) for *_, a, _ in rows],
        "receiving_currency": "US Dollar",
        "payment_currency": "US Dollar",
        "payment_format": "ACH",
        "is_laundering": [int(lab) for *_, lab in rows],
        "txn_id": range(1, len(rows) + 1),
    })


def _random_frame(seed: int, n: int = 400, days: int = 12, accounts: int = 30) -> pd.DataFrame:
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
        "is_laundering": (rng.random(n) < 0.03).astype(int),
        "txn_id": np.arange(1, n + 1),
    })


def test_features_ignore_transactions_after_day_t():
    horizon, t0 = 2, 6
    df = _random_frame(seed=1)
    base, manifest = build_features(df, horizon)

    cutoff = DAY0 + pd.Timedelta(days=t0)  # start of day t0+1
    later = df[df["ts"] >= cutoff].copy()
    changed = later.assign(amount_received=later["amount_received"] * 3, amount_paid=later["amount_paid"] * 3,
                           to_account="A0", from_account="A1")
    perturbed = pd.concat([df[df["ts"] < cutoff], changed,
                           _frame([(11, "A2", "A3", 500, 0)])], ignore_index=True)
    alt, _ = build_features(perturbed, horizon)

    cols = manifest["features"]
    a = base[base["t"] <= t0].set_index(["acct", "t"])[cols].sort_index()
    b = alt[alt["t"] <= t0].set_index(["acct", "t"])[cols].sort_index()
    assert a.index.equals(b.index)
    assert np.allclose(a.to_numpy(float), b.to_numpy(float))


def test_bucketed_run_matches_single_pass(tmp_path):
    df = _random_frame(seed=5)
    csv = tmp_path / "trans.csv"
    pd.DataFrame({
        "Timestamp": df["ts"].dt.strftime("%Y/%m/%d %H:%M"),
        "From Bank": "001", "Account": df["from_account"], "To Bank": "002", "Account.1": df["to_account"],
        "Amount Received": df["amount_received"], "Receiving Currency": "US Dollar",
        "Amount Paid": df["amount_paid"], "Payment Currency": "US Dollar",
        "Payment Format": df["payment_format"], "Is Laundering": df["is_laundering"],
    }).to_csv(csv, index=False)
    run_features(csv, tmp_path / "one", horizon=2, buckets=1)
    run_features(csv, tmp_path / "three", horizon=2, buckets=3)
    key = ["acct", "t"]
    one = pd.read_parquet(tmp_path / "one" / FEATURES_FILE).sort_values(key).reset_index(drop=True)
    three = pd.read_parquet(tmp_path / "three" / FEATURES_FILE).sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(one, three)


def test_resumed_run_reuses_cache_and_matches(tmp_path):
    df = _random_frame(seed=9)
    csv = tmp_path / "trans.csv"
    pd.DataFrame({
        "Timestamp": df["ts"].dt.strftime("%Y/%m/%d %H:%M"),
        "From Bank": "001", "Account": df["from_account"], "To Bank": "002", "Account.1": df["to_account"],
        "Amount Received": df["amount_received"], "Receiving Currency": "US Dollar",
        "Amount Paid": df["amount_paid"], "Payment Currency": "US Dollar",
        "Payment Format": df["payment_format"], "Is Laundering": df["is_laundering"],
    }).to_csv(csv, index=False)
    out = tmp_path / "features"
    first = run_features(csv, out, horizon=2, buckets=2)
    assert (out / "rules.pkl").exists() and (out / "bucket_0.parquet").exists()
    bucket_mtime = (out / "bucket_0.parquet").stat().st_mtime_ns
    second = run_features(csv, out, horizon=2, buckets=2)
    assert (out / "bucket_0.parquet").stat().st_mtime_ns == bucket_mtime  # not rebuilt
    assert first == second
    # A different horizon invalidates the cache and rebuilds the buckets.
    run_features(csv, out, horizon=3, buckets=2)
    assert (out / "bucket_0.parquet").stat().st_mtime_ns != bucket_mtime


def test_label_is_forward_looking_and_excludes_day_t():
    df = _frame([
        (1, "X", "Y", 100, 0),
        (5, "X", "Y", 100, 1),  # laundering on day 5
        (9, "Z", "W", 100, 0),
    ])
    frame, _ = build_features(df, horizon=2)
    x = frame[frame["acct"] == "X"]
    assert set(x.loc[x["y"] == 1, "t"]) == {3, 4}  # days 3 and 4 look ahead to day 5; day 5 itself does not


def test_window_covers_lookback_days():
    df = _frame([(1, "X", "Y", 100, 0), (9, "Z", "W", 100, 0)])
    frame, _ = build_features(df, horizon=1)
    x = frame[frame["acct"] == "X"].set_index("t")
    # A transaction on day 1 counts in windows t = 1 .. 1 + LOOKBACK_DAYS - 1 only.
    assert x.loc[1 + LOOKBACK_DAYS - 1, "out_n"] == 1
    assert x.loc[1, "out_n"] == 1
    assert 1 + LOOKBACK_DAYS not in x.index or x.loc[1 + LOOKBACK_DAYS, "out_n"] == 0


def test_split_gap_drops_label_overlap():
    t = np.arange(1, 13)
    s = assign_splits(t, horizon=2, train_end=4, val_end=8)
    assert list(s[:4]) == ["train"] * 4
    assert list(s[4:6]) == ["gap", "gap"]  # days 5 and 6: label windows would cross into val
    assert list(s[6:8]) == ["val", "val"]
    assert list(s[8:10]) == ["gap", "gap"]
    assert list(s[10:]) == ["test", "test"]


def test_default_cuts_rejects_short_span():
    with pytest.raises(ValueError):
        default_cuts(10, horizon=2)


def test_downsample_keeps_all_positives_at_ratio():
    df = pd.DataFrame({"y": [1] * 3 + [0] * 500, "x": range(503)})
    out = downsample(df)
    assert int(out["y"].sum()) == 3
    assert int((out["y"] == 0).sum()) == 30


def test_budget_metrics_flags_top_rows_per_day():
    df = pd.DataFrame({"t": [1, 1, 1, 2, 2, 2], "y": [1, 0, 0, 0, 1, 0],
                       "score": [0.9, 0.5, 0.1, 0.2, 0.8, 0.1]})
    m = budget_metrics(df, "score", budget=1)
    assert m["flagged"] == 2
    assert m["precision_at_budget"] == 1.0
    assert m["recall_at_budget"] == 1.0


def test_flag_top_respects_day_groups():
    df = pd.DataFrame({"t": [1, 1, 2, 2], "score": [0.3, 0.9, 0.1, 0.2]})
    assert flag_top(df, "score", budget=1).tolist() == [False, True, False, True]


def test_parse_patterns_reads_both_sides(tmp_path):
    path = tmp_path / "patterns.txt"
    path.write_text(
        "BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out\n"
        "2022/09/02 00:06,021174,800737690,012,80011F990,2848.96,Euro,2848.96,Euro,ACH,1\n"
        "END LAUNDERING ATTEMPT - FAN-OUT\n",
        encoding="utf-8",
    )
    got = parse_patterns(path, DAY0)
    assert set(map(tuple, got[["acct", "day", "typology"]].to_numpy())) == {
        ("800737690", 2, "FAN-OUT"), ("80011F990", 2, "FAN-OUT"),
    }


def test_typology_recall_counts_flagged_positives():
    patterns = pd.DataFrame({"acct": ["A"], "day": [5], "typology": ["FAN-OUT"]})
    test = pd.DataFrame({"acct": ["A", "A", "B"], "t": [3, 4, 3]})
    flagged = pd.Series([True, False, True])
    out = typology_recall(test, flagged, patterns, horizon=2)
    assert out["FAN-OUT"] == {"positives": 2, "flagged": 1, "recall_at_budget": 0.5}


def test_reason_codes_name_real_features(tmp_path):
    rng = np.random.default_rng(3)
    features = ["f0", "f1", "f2", "f3"]
    x = rng.normal(size=(600, 4)).astype(np.float32)
    y = (x[:, 0] > 0.8).astype(int)
    dtr = xgb.DMatrix(x[:400], label=y[:400], feature_names=features)
    dva = xgb.DMatrix(x[400:], label=y[400:], feature_names=features)
    booster = xgb.train({"objective": "binary:logistic", "eval_metric": "aucpr", "max_depth": 3},
                        dtr, num_boost_round=200, evals=[(dva, "val")], early_stopping_rounds=10,
                        verbose_eval=False)
    path = tmp_path / "model.json"
    booster.save_model(str(path))
    loaded = xgb.Booster()
    loaded.load_model(str(path))
    assert loaded.best_iteration == booster.best_iteration
    frame = pd.DataFrame(x[400:], columns=features)
    codes = reason_codes(loaded, frame, features)
    assert len(codes) == len(frame)
    assert all(set(c) <= set(features) and len(c) <= 3 for c in codes)
