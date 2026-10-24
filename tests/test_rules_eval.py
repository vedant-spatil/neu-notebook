"""Rule-only accuracy check (triage_lib.rules_eval): hand-checked metrics and a smoke run on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from triage_lib.alerts import build_alerts
from triage_lib.rules_eval import LIFT_BAR, MIN_ALERTS, evaluate_rules, population_and_labels, score_keys

DAY0 = pd.Timestamp("2022-09-01")


def _frame(rows):
    """rows: (day, from, to, is_laundering)."""
    return pd.DataFrame({
        "ts": [DAY0 + pd.Timedelta(days=d - 1, hours=12) for d, *_ in rows],
        "from_account": [f for _, f, *_ in rows],
        "to_account": [t for _, _, t, _ in rows],
        "amount_received": 100.0,
        "amount_paid": 100.0,
        "receiving_currency": "US Dollar",
        "payment_currency": "US Dollar",
        "payment_format": "ACH",
        "is_laundering": [int(lab) for *_, lab in rows],
        "txn_id": range(1, len(rows) + 1),
    })


def test_population_and_labels_by_hand():
    # A and B trade on days 1, 2 and 3. Day 2 is laundering, so with horizon 1 the day-1 rows are positive.
    df = _frame([(1, "A", "B", 0), (2, "A", "B", 1), (3, "A", "B", 0)])
    accounts = pd.Index(["A", "B"])
    pop, pop_y, pos = population_and_labels(df, accounts, DAY0, horizon=1, last_t=2)
    stride = 4  # last_t + 2; keys are acct * stride + t
    assert pop.tolist() == [1, 2, 5, 6]  # A1, A2, B1, B2
    assert pop_y.tolist() == [1, 0, 1, 0]  # positives: A1 and B1 (laundering on day 2)
    assert pos.tolist() == [1, 5]
    assert pos.tolist() == [0 * stride + 1, 1 * stride + 1]


def test_score_keys_hand_checked_lift():
    pop = np.arange(1, 11)
    pos = np.array([2, 3, 9])
    pop_y = np.isin(pop, pos).astype(np.int8)  # base rate 0.3
    m = score_keys(np.array([1, 2, 3, 4]), pos, pop, pop_y, n_days=10)
    assert m["alerts"] == 4 and m["tp"] == 2 and m["fp"] == 2
    assert m["precision"] == 0.5
    assert m["base_rate"] == 0.3
    assert m["lift"] == 1.67
    assert m["recall"] == round(2 / 3, 5)
    assert m["conclusive"] is False and m["passes_lift_bar"] is False  # too few alerts to judge


def test_score_keys_passes_when_lift_clears_bar():
    pop = np.arange(1, 1001)
    pos = np.arange(1, 26)  # 25 positives in 1000 rows: base rate 0.025
    pop_y = np.isin(pop, pos).astype(np.int8)
    m = score_keys(pos.copy(), pos, pop, pop_y, n_days=10)  # 25 alerts, all real laundering
    assert m["precision"] == 1.0
    assert m["lift"] >= LIFT_BAR
    assert m["alerts"] >= MIN_ALERTS and m["passes_lift_bar"] is True


def test_evaluate_rules_runs_on_synthetic_frame():
    rng = np.random.default_rng(5)
    rows = []
    for i in range(600):
        d = int(rng.integers(1, 9))
        f, t = f"A{rng.integers(0, 40)}", f"A{rng.integers(0, 40)}"
        if f != t:
            rows.append((d, f, t, bool(rng.random() < 0.05)))
    df = _frame(rows)
    alerts = build_alerts(df)
    out = evaluate_rules(df, alerts, horizon=2)
    assert set(out["rules"]) == {"R1_structuring", "R2_fan_out", "R3_fan_in", "R4_pass_through",
                                 "R5_cycle", "R6_scatter_gather", "R7_gather_scatter"}
    assert set(out["combined"]) == {"ANY_R1_R7", "ANY_R1_R4"}
    assert 0.0 <= out["base_rate"] <= 1.0
    assert out["caveat"].startswith("Graph rules R5 to R7")
