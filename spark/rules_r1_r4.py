"""R1 to R4 rule windows, one account at a time. Used by the Spark feature job (build_features_spark.py).

Copied from triage_lib/alerts.py (build_alerts, R1 to R4 blocks). The logic is the same, but instead of
building Alert objects it keeps only what the features and the baseline need, per (account, rule):
the start of the first window that fired, and the number of windows that fired.

Each account's R1 to R4 alerts depend only on that account's own transactions, so the Spark job can
run this on each account's group. Graph rules R5 to R7 are not here: they need the whole edge set.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter

import pandas as pd

H24 = pd.Timedelta(hours=24)
H48 = pd.Timedelta(hours=48)
RULE_WINDOW = pd.Timedelta(hours=72)
WEIGHTS = {"R1_structuring": 0.15, "R2_fan_out": 0.3, "R3_fan_in": 0.6, "R4_pass_through": 0.15}


def rule_windows(usd: pd.DataFrame) -> dict[tuple[str, str], list]:
    """usd: transactions in USD, columns from_account, to_account, ts, amount_received, amount_paid,
    payment_format. Returns {(account, rule): [first_window_start, windows_fired]}."""
    found: dict[tuple[str, str], list] = {}

    def add(acct: str, rule: str, ts: pd.Timestamp) -> None:
        key = (acct, rule)
        if key in found:
            found[key][0] = min(found[key][0], ts)
            found[key][1] += 1
        else:
            found[key] = [ts, 1]

    # R1 structuring: Cash, under $10,000 (received), at least 3 within 24 h, per sender.
    cash = usd[(usd["payment_format"] == "Cash") & (usd["amount_received"] < 10_000)]
    for acct, g in cash.sort_values("ts").groupby("from_account"):
        if len(g) < 3:
            continue
        times = g["ts"].tolist()
        for i in range(len(times)):
            j = bisect_right(times, times[i] + H24)
            if j - i >= 3:
                add(acct, "R1_structuring", times[i])

    # R2 fan-out (per sender) and R3 fan-in (per receiver): at least 10 distinct counterparties within 48 h.
    for key_col, other_col, rule in (("from_account", "to_account", "R2_fan_out"),
                                     ("to_account", "from_account", "R3_fan_in")):
        for acct, g in usd.sort_values("ts").groupby(key_col):
            if g[other_col].nunique() < 10:
                continue
            times = g["ts"].tolist()
            others = g[other_col].tolist()
            counts: Counter[str] = Counter()
            j = 0
            for i in range(len(times)):
                while j < len(times) and times[j] <= times[i] + H48:
                    counts[others[j]] += 1
                    j += 1
                if len(counts) >= 10:
                    add(acct, rule, times[i])
                counts[others[i]] -= 1
                if counts[others[i]] == 0:
                    del counts[others[i]]

    # R4 pass-through: money received, and at least 80% of it sent out within 48 h after receipt.
    inbound = usd.sort_values("ts")
    outbound_by_acct = {a: g.sort_values("ts") for a, g in usd.groupby("from_account")}
    for acct, g in inbound.groupby("to_account"):
        if acct not in outbound_by_acct:
            continue
        out_g = outbound_by_acct[acct]
        out_times = out_g["ts"].tolist()
        out_cum = out_g["amount_paid"].cumsum().tolist()
        for t0, received in zip(g["ts"], g["amount_received"]):
            lo = bisect_right(out_times, t0)
            hi = bisect_right(out_times, t0 + H48)
            if hi <= lo:
                continue
            sent = out_cum[hi - 1] - (out_cum[lo - 1] if lo > 0 else 0.0)
            if received > 0 and sent >= 0.8 * received:
                add(acct, "R4_pass_through", t0)
    return found


def score_windows(counts_by_rule: dict[str, int]) -> float:
    """score_alert from alerts.py, for R1 to R4 (all windowed, so frequency scales with the window count)."""
    total = 0.0
    for rule, n in counts_by_rule.items():
        total += WEIGHTS[rule] * (0.5 + 0.5 * min(1.0, n / 20))
    return round(min(1.0, total), 4)
