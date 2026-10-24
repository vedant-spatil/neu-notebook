"""Parallel R1 to R4 (triage_lib.alerts) must give exactly the serial alerts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from triage_lib.alerts import build_alerts

DAY0 = pd.Timestamp("2022-09-01")


def _frame(seed: int, n: int = 1500, days: int = 3, accounts: int = 25) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frm = rng.integers(0, accounts, n)
    to = (frm + rng.integers(1, accounts, n)) % accounts
    # Coarse timestamps (to the hour) so many rows tie: tie order must match the serial code too.
    ts = DAY0 + pd.to_timedelta(rng.integers(0, days * 24, n), unit="h")
    amt = rng.choice([50.0, 500.0, 9000.0, 12000.0], n)
    return pd.DataFrame({
        "ts": ts,
        "from_account": [f"A{i}" for i in frm],
        "to_account": [f"A{i}" for i in to],
        "amount_received": amt,
        "amount_paid": amt,
        "receiving_currency": "US Dollar",
        "payment_currency": "US Dollar",
        "payment_format": rng.choice(["Cash", "ACH"], n),
        "is_laundering": (rng.random(n) < 0.05).astype(int),
        "txn_id": np.arange(1, n + 1),
    })


def _key(alerts):
    return sorted(
        (a.account_id, a.rules[0], a.ts, a.last_ts, a.amount, a.risk_score, tuple(a.txn_ids),
         bool(a.is_laundering), tuple(sorted(a.rule_counts.items())))
        for a in alerts
    )


def test_parallel_matches_serial_on_synthetic_frame():
    df = _frame(seed=7)
    serial = build_alerts(df, include_graph=False, workers=1)
    parallel = build_alerts(df, include_graph=False, workers=3)
    assert len(serial) > 0
    assert _key(serial) == _key(parallel)


def test_serial_output_order_is_stable():
    df = _frame(seed=8)
    a = build_alerts(df, include_graph=False, workers=1)
    b = build_alerts(df, include_graph=False, workers=1)
    assert [x.account_id for x in a] == [x.account_id for x in b]
