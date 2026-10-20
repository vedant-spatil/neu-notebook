"""Notebook copy: this folder is the training source of truth for the SageMaker notebook.
It was copied from backend/app/ml/triage_features.py. Changes here: resumable run_features (cache files).

Account-day features and forward-looking labels for the XGBoost triage model (plan.md section 24).

One row per (account, t). Features use transactions on days t-6..t (7-day lookback), so they see
data only up to the end of day t. The label is 1 if the account is a sender or receiver in a
laundering transaction on days t+1..t+HORIZON. Pure pandas (HI-Small scale). HI-Large runs the
same logic as a Spark job (plan.md section 24, step B).

Leakage rules:
- Transaction features use days <= t only.
- A rule feature switches on once its first window has finished (window start + 72 h falls on or
  before day t).
- `rule_score_app` is the baseline column: the app's rule score, which uses whole-period window counts
  exactly as triage does today. It is never a model feature, and it is excluded from the leakage test.
- Known gap: the graph DEGREE_CAP in graph_rules.py uses degree over the whole file.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from triage_lib.alerts import MAX_TXN_IDS, Alert, build_alerts, load_csv

LOOKBACK_DAYS = 7
BUCKETS = 8  # account buckets per run_features pass; see run_features
RULE_WINDOW = pd.Timedelta(hours=72)  # longest rule window (graph rules R5 to R7)
RULES = (
    "R1_structuring", "R2_fan_out", "R3_fan_in", "R4_pass_through",
    "R5_cycle", "R6_scatter_gather", "R7_gather_scatter",
)
GRAPH_RULES = frozenset({"R5_cycle", "R6_scatter_gather", "R7_gather_scatter"})
NEVER = 10**9  # fire day for rules that never fired

FEATURES_FILE = "features.parquet"
MANIFEST_FILE = "features_manifest.json"

WINDOW_AGG = {"n": "sum", "amt_sum": "sum", "amt_max": "max", "sub": "sum",
              "rnd": "sum", "night": "sum", "nonusd": "sum"}
COUNT_FEATURES = ["out_n", "in_n", "out_amt_sum", "in_amt_sum", "out_amt_max", "in_amt_max",
                  "out_distinct", "in_distinct", "out_sub"]
DERIVED_FEATURES = ["out_mean", "in_mean", "out_fanout", "in_fanin", "pass_through",
                    "sub_share", "round_share", "night_share", "nonusd_share"]
# R1 to R4 are windowed, so a window is known to have fired once it has ended. R5 to R7 are built over
# the whole file (graph_rules.py), so the alert that reaches triage can depend on later transactions.
# They stay out of the model's features and are used only in the baseline column.
MODEL_RULES = RULES[:4]
RULE_FEATURES = [f"rule_{r}" for r in MODEL_RULES] + ["rules_fired"]


def _slug(name: str) -> str:
    return re.sub(r"\W+", "_", str(name)).strip("_").lower()


def _day_index(ts: pd.Series, day0: pd.Timestamp) -> np.ndarray:
    """Day 1 is the first calendar day in the file."""
    return ((ts.dt.normalize() - day0).dt.days + 1).to_numpy(dtype="int32")


def _legs(df: pd.DataFrame, day0: pd.Timestamp, accounts: pd.Index | None) -> tuple[pd.DataFrame, pd.Index]:
    """Two rows per transaction: one for the sender (dir 0) and one for the receiver (dir 1).

    accounts: a fixed account index (run_features), so codes match across passes. None: codes from df."""
    n = len(df)
    if accounts is None:
        codes, uniques = pd.factorize(pd.concat([df["from_account"], df["to_account"]], ignore_index=True))
    else:
        uniques = accounts
        codes = np.concatenate([accounts.get_indexer(df["from_account"]), accounts.get_indexer(df["to_account"])])
    usd = ((df["receiving_currency"] == "US Dollar") & (df["payment_currency"] == "US Dollar")).to_numpy()
    hour = df["ts"].dt.hour.to_numpy()
    base = pd.DataFrame({
        "day": _day_index(df["ts"], day0),
        "fmt": df["payment_format"].to_numpy(),
        "lab": df["is_laundering"].astype(bool).to_numpy(),
        "night": (hour < 6) | (hour >= 22),
        "nonusd": ~usd,
    })
    # R1 structuring is judged on the sender's Cash, USD, under $10,000 transactions.
    sub = ((df["payment_format"] == "Cash").to_numpy() & usd & (df["amount_received"] < 10_000).to_numpy())
    out = base.assign(acct=codes[:n], cp=codes[n:], dir=0,
                      amt=df["amount_paid"].to_numpy(dtype=float), sub=sub)
    inn = base.assign(acct=codes[n:], cp=codes[:n], dir=1,
                      amt=df["amount_received"].to_numpy(dtype=float), sub=False)
    legs = pd.concat([out, inn], ignore_index=True)
    legs["rnd"] = (legs["amt"] > 0) & (legs["amt"] % 100 == 0)
    return legs, uniques


def _window(daily: pd.DataFrame, keys: list[str], agg: dict[str, str]) -> pd.DataFrame:
    """Sum (or max) each day's row into the lookback windows it belongs to: window t covers days t-6..t,
    so a row for day d counts toward t = d .. d+6."""
    parts = [daily.assign(t=daily["day"] + k) for k in range(LOOKBACK_DAYS)]
    return pd.concat(parts, ignore_index=True).groupby(keys + ["t"]).agg(agg).reset_index()


def _distinct_window(cp_days: pd.DataFrame) -> pd.DataFrame:
    """Distinct counterparties per (account, dir, window). Needs dedupe, so not a plain sum."""
    parts = [cp_days.assign(t=cp_days["day"] + k) for k in range(LOOKBACK_DAYS)]
    exp = pd.concat(parts, ignore_index=True).drop_duplicates(["acct", "dir", "cp", "t"])
    return exp.groupby(["acct", "dir", "t"]).size().rename("distinct").reset_index()


def _rule_arrays(alerts: list[Alert], code_of: dict[str, int], day0: pd.Timestamp,
                 n_acc: int, ts_by_txn: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """fire[acct, rule] = first day the rule is known to have fired; score[acct, rule] = app rule score.

    Windowed rules (R1 to R4) are known once their first window ends. Graph groups (R5 to R7) are built
    over the whole file, so they are known only once their last transaction has happened. Groups whose
    transaction list was truncated (MAX_TXN_IDS) cannot be dated, so they stay off.
    """
    fire = np.full((n_acc, len(RULES)), NEVER, dtype=np.int64)
    score = np.zeros((n_acc, len(RULES)), dtype=float)
    rule_pos = {r: k for k, r in enumerate(RULES)}
    for a in alerts:
        code = code_of.get(a.account_id)
        if code is None:
            continue
        for rule in a.rules:
            k = rule_pos.get(rule)
            if k is None:
                continue
            if rule in GRAPH_RULES:
                if not a.txn_ids or len(a.txn_ids) >= MAX_TXN_IDS:
                    continue
                known = max(a.ts + RULE_WINDOW, pd.Timestamp(ts_by_txn.loc[a.txn_ids].max()))
            else:
                known = a.ts + RULE_WINDOW
            fire_day = int((known.normalize() - day0).days) + 1
            fire[code, k] = min(fire[code, k], fire_day)
            score[code, k] = max(score[code, k], a.risk_score)
    return fire, score


def build_features(df: pd.DataFrame, horizon: int, day0: pd.Timestamp | None = None,
                   max_day: int | None = None, accounts: pd.Index | None = None,
                   rule_arrays: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """df: rows from app.ingest.alerts.load_csv. Returns (frame, manifest).

    day0, max_day: the whole-file range (default: range of df). accounts and rule_arrays: shared with
    run_features, so one bucket pass sees the same account codes and the same rule alerts as a full pass."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1 day")
    if day0 is None:
        day0 = df["ts"].min().normalize()
    if max_day is None:
        max_day = int((df["ts"].dt.normalize() - day0).dt.days.max()) + 1

    legs, uniques = _legs(df, day0, accounts)
    n_acc = len(uniques)
    daily = legs.groupby(["acct", "dir", "day"]).agg(
        n=("amt", "size"), amt_sum=("amt", "sum"), amt_max=("amt", "max"),
        sub=("sub", "sum"), rnd=("rnd", "sum"), night=("night", "sum"), nonusd=("nonusd", "sum"),
    ).reset_index()
    cp_days = legs[["acct", "dir", "cp", "day"]].drop_duplicates()
    fmt_daily = legs.groupby(["acct", "day", "fmt"]).size().unstack("fmt", fill_value=0)
    fmt_daily.columns = [f"fmt_{_slug(c)}" for c in fmt_daily.columns]
    fmt_daily = fmt_daily.reset_index()
    fmt_cols = [c for c in fmt_daily.columns if c.startswith("fmt_")]

    win = _window(daily, ["acct", "dir"], WINDOW_AGG).set_index(["acct", "dir", "t"]).unstack("dir")
    win.columns = [f"{'out' if d == 0 else 'in'}_{m}" for m, d in win.columns]
    dist = _distinct_window(cp_days).set_index(["acct", "dir", "t"]).unstack("dir")
    dist.columns = [f"{'out' if d == 0 else 'in'}_distinct" for _, d in dist.columns]
    fmt_win = _window(fmt_daily, ["acct"], {c: "sum" for c in fmt_cols}).set_index(["acct", "t"])

    grid = win.join(dist, how="outer").join(fmt_win, how="outer").fillna(0).reset_index()
    for c in COUNT_FEATURES + ["out_rnd", "in_rnd", "out_night", "in_night", "out_nonusd", "in_nonusd"]:
        if c not in grid:
            grid[c] = 0.0
    grid = grid[grid["t"] <= max_day - horizon].reset_index(drop=True)  # labels must be observable

    def div(num: pd.Series, den: pd.Series) -> np.ndarray:
        return np.asarray(num, dtype=float) / np.maximum(np.asarray(den, dtype=float), 1.0)

    n_all = grid["out_n"] + grid["in_n"]
    lo = np.minimum(grid["out_amt_sum"], grid["in_amt_sum"]).to_numpy(float)
    hi = np.maximum(grid["out_amt_sum"], grid["in_amt_sum"]).to_numpy(float)
    grid["out_mean"] = div(grid["out_amt_sum"], grid["out_n"])
    grid["in_mean"] = div(grid["in_amt_sum"], grid["in_n"])
    grid["out_fanout"] = div(grid["out_distinct"], grid["out_n"])
    grid["in_fanin"] = div(grid["in_distinct"], grid["in_n"])
    grid["pass_through"] = np.where(hi > 0, lo / np.where(hi > 0, hi, 1.0), 0.0)
    grid["sub_share"] = div(grid["out_sub"], grid["out_n"])
    grid["round_share"] = div(grid["out_rnd"] + grid["in_rnd"], n_all)
    grid["night_share"] = div(grid["out_night"] + grid["in_night"], n_all)
    grid["nonusd_share"] = div(grid["out_nonusd"] + grid["in_nonusd"], n_all)

    if rule_arrays is None:
        alerts = build_alerts(df)
        ts_by_txn = pd.Series(df["ts"].to_numpy(), index=df["txn_id"].to_numpy())
        rule_arrays = _rule_arrays(alerts, {str(a): i for i, a in enumerate(uniques)}, day0, n_acc, ts_by_txn)
    fire, score = rule_arrays
    acct = grid["acct"].to_numpy(dtype=np.int64)
    t = grid["t"].to_numpy(dtype=np.int64)
    active = t[:, None] >= fire[acct]  # rule has fired by the end of day t
    for k, name in enumerate(MODEL_RULES):
        grid[f"rule_{name}"] = active[:, k].astype("int8")
    grid["rules_fired"] = active[:, : len(MODEL_RULES)].sum(axis=1).astype("int8")
    grid["rule_score_app"] = np.where(active, score[acct], 0.0).max(axis=1)

    # Labels: account is in a laundering transaction on days t+1 .. t+horizon.
    pos = legs.loc[legs["lab"], ["acct", "day"]].drop_duplicates()
    lab = pd.concat([pos.assign(t=pos["day"] - k) for k in range(1, horizon + 1)], ignore_index=True)
    lab = lab[["acct", "t"]].drop_duplicates().assign(y=1)
    grid = grid.merge(lab, on=["acct", "t"], how="left")
    grid["y"] = grid["y"].fillna(0).astype("int8")

    grid["acct"] = np.asarray(uniques)[grid["acct"].to_numpy(dtype=np.int64)]
    grid["t_date"] = day0 + pd.to_timedelta(grid["t"] - 1, unit="D")
    features = COUNT_FEATURES + DERIVED_FEATURES + fmt_cols + RULE_FEATURES
    frame = grid[["acct", "t", "t_date", *features, "rule_score_app", "y"]].reset_index(drop=True)

    manifest = {
        "day0": day0.date().isoformat(),
        "max_day": max_day,
        "horizon": horizon,
        "lookback_days": LOOKBACK_DAYS,
        "features": features,
        "formats": [c.removeprefix("fmt_") for c in fmt_cols],
        "rows": int(len(frame)),
        "positives": int(frame["y"].sum()),
        "label_rows_total": int(len(lab)),
        "label_rows_unmatched": int(len(lab) - frame["y"].sum()),
    }
    return frame, manifest


def write_features(frame: pd.DataFrame, manifest: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / FEATURES_FILE
    frame.to_parquet(path, index=False)
    (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def run_features(csv_path: Path, out_dir: Path, horizon: int, buckets: int = BUCKETS) -> dict[str, Any]:
    """Builds features for the whole file, one account bucket at a time, to bound memory.

    Rule alerts (build_alerts) run once on the full file, because graph rules R5 to R7 need every edge.
    Each bucket then gets every transaction that touches one of its accounts, so that account's
    transaction features, labels and R1 to R4 flags are complete. Rows are kept only for the bucket's
    own accounts, so each (account, t) appears exactly once.

    Resumable: the rule pass is saved to out_dir/rules.pkl and each bucket to out_dir/bucket_<i>.parquet
    (with bucket_<i>.json for its manifest) as soon as it finishes. A re-run reuses those files.
    cache_key.json ties them to the CSV size, horizon and bucket count; if any of those change, the
    pieces are deleted and rebuilt.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(csv_path)
    key = {"csv": csv_path.name, "csv_bytes": csv_path.stat().st_size, "horizon": horizon, "buckets": buckets}
    key_path = out_dir / "cache_key.json"
    if not key_path.exists() or json.loads(key_path.read_text(encoding="utf-8")) != key:
        for piece in list(out_dir.glob("bucket_*")) + [out_dir / "rules.pkl"]:
            piece.unlink(missing_ok=True)
        key_path.write_text(json.dumps(key, indent=2), encoding="utf-8")

    df = load_csv(str(csv_path))
    day0 = df["ts"].min().normalize()
    max_day = int((df["ts"].dt.normalize() - day0).dt.days.max()) + 1
    accounts = pd.Index(pd.unique(pd.concat([df["from_account"], df["to_account"]], ignore_index=True)))

    rules_path = out_dir / "rules.pkl"
    if rules_path.exists():
        with rules_path.open("rb") as fh:
            rule_arrays = pickle.load(fh)
        print("[triage] rule pass: cached", flush=True)
    else:
        alerts = build_alerts(df)
        ts_by_txn = pd.Series(df["ts"].to_numpy(), index=df["txn_id"].to_numpy())
        rule_arrays = _rule_arrays(alerts, {str(a): i for i, a in enumerate(accounts)}, day0, len(accounts), ts_by_txn)
        del alerts
        with rules_path.open("wb") as fh:
            pickle.dump(rule_arrays, fh)

    bucket_of = pd.Series(np.arange(len(accounts)) % buckets, index=accounts)
    from_b = df["from_account"].map(bucket_of).to_numpy()
    to_b = df["to_account"].map(bucket_of).to_numpy()
    parts = []
    manifest: dict[str, Any] = {}
    for b in range(buckets):
        part_path = out_dir / f"bucket_{b}.parquet"
        meta_path = out_dir / f"bucket_{b}.json"
        if part_path.exists() and meta_path.exists():
            part = pd.read_parquet(part_path)
            manifest = json.loads(meta_path.read_text(encoding="utf-8"))
            print(f"[triage] bucket {b + 1}/{buckets}: cached rows={len(part)}", flush=True)
        else:
            sub = df[(from_b == b) | (to_b == b)]
            frame, manifest = build_features(sub, horizon, day0=day0, max_day=max_day,
                                             accounts=accounts, rule_arrays=rule_arrays)
            part = frame[frame["acct"].map(bucket_of).to_numpy() == b]
            part.to_parquet(part_path, index=False)
            meta_path.write_text(json.dumps(manifest), encoding="utf-8")
            print(f"[triage] bucket {b + 1}/{buckets}: txns={len(sub)} rows={len(part)}", flush=True)
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    # Label-row counts from the last pass cover that bucket only, so they are not reported globally.
    manifest.pop("label_rows_total", None)
    manifest.pop("label_rows_unmatched", None)
    manifest["rows"] = int(len(frame))
    manifest["positives"] = int(frame["y"].sum())
    manifest["buckets"] = buckets
    write_features(frame, manifest, out_dir)
    return manifest
