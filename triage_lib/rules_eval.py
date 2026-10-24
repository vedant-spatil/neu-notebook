"""Rule-only accuracy check: lift of each rule (R1 to R7) and of the combined rulebase.

A rule alert counts on the day it opens (its fire day, from _rule_arrays). The alert is scored against
laundering in the next HORIZON days for the same account. Base rate: laundering rate over every
account active on a day (the population). Lift = precision / base rate.

Caveat: graph rules R5 to R7 are built over the whole file (graph_rules.py), so their fire days can depend
on later transactions. This backtests the rulebase as the app runs it today; it is not a point-in-time
test. The metrics file says so.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from triage_lib.triage_features import RULES, _rule_arrays

LIFT_BAR = 10.0       # a rule must flag account-days with at least this many times the base laundering rate
MIN_ALERTS = 20       # fewer alerts than this: not enough to judge the rule
COMBINED = {"ANY_R1_R7": RULES, "ANY_R1_R4": RULES[:4]}


def score_keys(alert_keys: np.ndarray, pos_keys: np.ndarray, pop_keys: np.ndarray, pop_y: np.ndarray,
               n_days: int) -> dict[str, Any]:
    """Metrics for one set of alerts. Keys are acct * (n_days + 2) + t."""
    alert_keys = np.unique(alert_keys)
    n = len(alert_keys)
    tp = int(np.isin(alert_keys, pos_keys).sum())
    precision = tp / n if n else 0.0
    base_rate = float(pop_y.mean()) if len(pop_y) else 0.0
    lift = precision / base_rate if base_rate else None
    recall = float(np.isin(pos_keys, alert_keys).mean()) if len(pos_keys) else 0.0
    return {
        "alerts": n,
        "tp": tp,
        "fp": n - tp,
        "precision": round(precision, 5),
        "base_rate": round(base_rate, 6),
        "lift": round(lift, 2) if lift is not None else None,
        "recall": round(recall, 5),
        "alerts_per_day": round(n / n_days, 2) if n_days else 0.0,
        "conclusive": n >= MIN_ALERTS,
        "passes_lift_bar": bool(n >= MIN_ALERTS and lift is not None and lift >= LIFT_BAR),
    }


def population_and_labels(df: pd.DataFrame, accounts: pd.Index, day0: pd.Timestamp, horizon: int,
                          last_t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (pop_keys, pop_y, pos_keys). Population: every account with a transaction on day t (t <= last_t).
    Positive: account in a laundering transaction on days t+1..t+horizon."""
    stride = last_t + 2
    day = ((df["ts"].dt.normalize() - day0).dt.days + 1).to_numpy()
    fa = accounts.get_indexer(df["from_account"])
    ta = accounts.get_indexer(df["to_account"])
    acct = np.concatenate([fa, ta]).astype(np.int64)
    days = np.concatenate([day, day]).astype(np.int64)
    keep = days <= last_t
    pop_keys = np.unique(acct[keep] * stride + days[keep])

    lab = df["is_laundering"].to_numpy(bool)
    la = np.concatenate([fa[lab], ta[lab]]).astype(np.int64)
    ld = np.concatenate([day[lab], day[lab]]).astype(np.int64)
    offsets = np.arange(1, horizon + 1, dtype=np.int64)
    t_pos = (ld[:, None] - offsets[None, :]).ravel()
    a_pos = np.repeat(la, horizon)
    ok = (t_pos >= 1) & (t_pos <= last_t)
    # Positives count only for accounts active on that day, so they are scoreable like the population.
    pos_keys = np.intersect1d(np.unique(a_pos[ok] * stride + t_pos[ok]), pop_keys)
    pop_y = np.isin(pop_keys, pos_keys).astype(np.int8)
    return pop_keys, pop_y, pos_keys


def evaluate_rules(df: pd.DataFrame, alerts: list, horizon: int) -> dict[str, Any]:
    """df: rows from load_csv. alerts: build_alerts(df) output (graph rules included). Returns per-rule and combined metrics."""
    day0 = df["ts"].min().normalize()
    max_day = int((df["ts"].dt.normalize() - day0).dt.days.max()) + 1
    last_t = max_day - horizon
    if last_t < 1:
        raise ValueError(f"span of {max_day} days is too short for horizon {horizon}")
    accounts = pd.Index(pd.unique(pd.concat([df["from_account"], df["to_account"]], ignore_index=True)))
    code_of = {str(a): i for i, a in enumerate(accounts)}
    ts_by_txn = pd.Series(df["ts"].to_numpy(), index=df["txn_id"].to_numpy())
    fire, _score = _rule_arrays(alerts, code_of, day0, len(accounts), ts_by_txn)

    stride = last_t + 2
    pop_keys, pop_y, pos_keys = population_and_labels(df, accounts, day0, horizon, last_t)
    per_rule_keys: dict[str, np.ndarray] = {}
    for k, name in enumerate(RULES):
        acct_idx = np.where(fire[:, k] <= last_t)[0].astype(np.int64)
        per_rule_keys[name] = acct_idx * stride + fire[acct_idx, k]

    rules = {name: score_keys(per_rule_keys[name], pos_keys, pop_keys, pop_y, last_t) for name in RULES}
    combined = {}
    for label, names in COMBINED.items():
        keys = np.concatenate([per_rule_keys[n] for n in names]) if names else np.array([], dtype=np.int64)
        combined[label] = score_keys(keys, pos_keys, pop_keys, pop_y, last_t)
    return {
        "horizon": horizon,
        "days_evaluated": last_t,
        "population_rows": int(len(pop_keys)),
        "positive_rows": int(len(pos_keys)),
        "base_rate": round(float(pop_y.mean()), 6) if len(pop_y) else 0.0,
        "lift_bar": LIFT_BAR,
        "min_alerts": MIN_ALERTS,
        "rules": rules,
        "combined": combined,
        "caveat": ("Graph rules R5 to R7 use the whole file, so their fire days can depend on later transactions. "
                   "This backtests the rulebase as the app runs it; it is not a point-in-time test."),
    }
