"""Alert generation from HI-Small (plan 16.1, 16.2). Pure pandas, no LLM.

Rules (US, USD only, per account, rolling windows):
  R1 structuring        >=3 Cash-format USD txns < $10,000 within 24 h
  R2 fan-out            sends to >=10 distinct accounts within 48 h
  R3 fan-in             receives from >=10 distinct accounts within 48 h
  R4 pass-through       receives and sends >=80% out within 48 h
  R5 cycle              directed cycle of length 3 or 4 within 72 h (graph_rules.py)
  R6 scatter-gather     >=3 intermediates between a source and a sink within 72 h (graph_rules.py)
  R7 gather-scatter     hub with >=3 sources and >=3 targets within 72 h (graph_rules.py)
Weights are in WEIGHTS below. risk_score = min(1.0, sum of contributions). A case opens at 0.6.
"""

from __future__ import annotations

import resource
import time
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from triage_lib.graph_rules import graph_hits

# R1 to R4 tuned on HI-Small days 1-7 (train) and confirmed on days 8-10 (held out).
# R3 fan-in is the only rule with useful precision (7.8% train, 11.7% held out), so it alone can open a case.
# R1 structuring fired 1 laundering alert in 2,012 (train) and 0 in 93 (held out): supporting weight only.
# R4 pass-through is 0.5% to 1.3% precision: supporting weight only.
# R5 to R7 are PROVISIONAL: 0.6 lets one group open a case. Set from measured precision before the demo.
WEIGHTS = {
    "R1_structuring": 0.15,
    "R2_fan_out": 0.3,
    "R3_fan_in": 0.6,
    "R4_pass_through": 0.15,
    "R5_cycle": 0.6,
    "R6_scatter_gather": 0.6,
    "R7_gather_scatter": 0.6,
}
# Graph rules fire once per group, not per sliding window, so one hit gives the full weight.
ONE_HIT_RULES = {"R5_cycle", "R6_scatter_gather", "R7_gather_scatter"}
H24 = pd.Timedelta(hours=24)
H48 = pd.Timedelta(hours=48)
MAX_TXN_IDS = 200


@dataclass
class Alert:
    """One alert per (account, rule). Windows that fired for the same pair are merged."""
    account_id: str
    ts: pd.Timestamp  # first window start
    amount: float  # peak single-window total
    rules: list[str]
    risk_score: float
    txn_ids: list[int] = field(default_factory=list)
    is_laundering: bool | None = None
    rule_counts: dict[str, int] = field(default_factory=dict)  # rule -> windows fired
    last_ts: pd.Timestamp | None = None  # last window start


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={"From Bank": str, "To Bank": str, "Account": str, "Account.1": str},
    )
    df = df.rename(
        columns={
            "Timestamp": "ts",
            "From Bank": "from_bank",
            "Account": "from_account",
            "To Bank": "to_bank",
            "Account.1": "to_account",
            "Amount Received": "amount_received",
            "Receiving Currency": "receiving_currency",
            "Amount Paid": "amount_paid",
            "Payment Currency": "payment_currency",
            "Payment Format": "payment_format",
            "Is Laundering": "is_laundering",
        }
    )
    df["ts"] = pd.to_datetime(df["ts"], format="%Y/%m/%d %H:%M")
    df["txn_id"] = df.index + 1  # 1-based row id; same as transactions.id and TRANSFER.txn_id
    return df


def _usd(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["receiving_currency"] == "US Dollar") & (df["payment_currency"] == "US Dollar")]


def _progress(stage: str, started: float) -> None:
    """Progress line for long runs: elapsed seconds and peak memory (Linux ru_maxrss is in KB)."""
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[alerts] {stage} elapsed={time.perf_counter() - started:.0f}s peak_rss={peak_mb:.0f}MB", flush=True)


def build_alerts(df: pd.DataFrame, include_graph: bool = True) -> list[Alert]:
    started = time.perf_counter()
    usd = _usd(df)
    print(f"[alerts] usd rows={len(usd)} of {len(df)}", flush=True)
    alerts: dict[tuple[str, str, pd.Timestamp], Alert] = {}

    def add(acct: str, rule: str, ts: pd.Timestamp, amount: float, txn_ids: list[int], lab: bool | None) -> None:
        key = (acct, rule)
        if key in alerts:
            a = alerts[key]
            a.rule_counts[rule] = a.rule_counts.get(rule, 0) + 1
            a.amount = max(a.amount, amount)
            a.last_ts = max(a.last_ts or ts, ts)
            if ts < a.ts:
                a.ts = ts
            seen = set(a.txn_ids)
            a.txn_ids += [i for i in txn_ids if i not in seen][: MAX_TXN_IDS - len(a.txn_ids)]
            a.is_laundering = bool(a.is_laundering) or bool(lab)
        else:
            alerts[key] = Alert(acct, ts, amount, [rule], 0.0, list(txn_ids[:MAX_TXN_IDS]), lab,
                                {rule: 1}, ts)

    # R1 structuring: Cash, USD, under $10k, >=3 within 24h, per sender.
    cash = usd[(usd["payment_format"] == "Cash") & (usd["amount_received"] < 10_000)]
    for acct, g in cash.sort_values("ts").groupby("from_account"):
        if len(g) < 3:
            continue
        times = g["ts"].tolist()
        ids = g["txn_id"].tolist()
        amts = g["amount_received"].tolist()
        labs = g["is_laundering"].tolist()
        for i in range(len(times)):
            j = bisect_right(times, times[i] + H24)
            if j - i >= 3:
                add(acct, "R1_structuring", times[i], sum(amts[i:j]), ids[i:j], any(labs[i:j]))

    # R2 fan-out and R3 fan-in: distinct counterparties within 48h.
    for side, key_col, other_col, rule in (
        ("out", "from_account", "to_account", "R2_fan_out"),
        ("in", "to_account", "from_account", "R3_fan_in"),
    ):
        for acct, g in usd.sort_values("ts").groupby(key_col):
            if g[other_col].nunique() < 10:
                continue
            times = g["ts"].tolist()
            others = g[other_col].tolist()
            ids = g["txn_id"].tolist()
            amts = g["amount_received"].tolist()
            labs = g["is_laundering"].tolist()
            counts: Counter[str] = Counter()
            j = 0
            for i in range(len(times)):
                while j < len(times) and times[j] <= times[i] + H48:
                    counts[others[j]] += 1
                    j += 1
                if len(counts) >= 10:
                    add(acct, rule, times[i], sum(amts[i:j]), ids[i:j], any(labs[i:j]))
                counts[others[i]] -= 1
                if counts[others[i]] == 0:
                    del counts[others[i]]
        _progress(f"{rule} done", started)

    # R4 pass-through: in-amount, >=80% sent out within 48h after receipt.
    inbound = usd.sort_values("ts")
    outbound_by_acct = {a: g.sort_values("ts") for a, g in usd.groupby("from_account")}
    for acct, g in inbound.groupby("to_account"):
        if acct not in outbound_by_acct:
            continue
        out_g = outbound_by_acct[acct]
        out_times = out_g["ts"].tolist()
        out_cum = out_g["amount_paid"].cumsum().tolist()
        for t0, received, txn_id, lab in zip(g["ts"], g["amount_received"], g["txn_id"], g["is_laundering"]):
            lo = bisect_right(out_times, t0)
            hi = bisect_right(out_times, t0 + H48)
            if hi <= lo:
                continue
            sent = out_cum[hi - 1] - (out_cum[lo - 1] if lo > 0 else 0.0)
            if received > 0 and sent >= 0.8 * received:
                add(acct, "R4_pass_through", t0, float(received), [int(txn_id)], bool(lab))

    _progress("R4 done", started)

    # R5 to R7: graph groups over the USD edges.
    graph_count = 0
    for hit in (graph_hits(usd) if include_graph else []):
        add(*hit)
        graph_count += 1
    _progress(f"R5-R7 done, {graph_count} group hits", started)

    out: list[Alert] = []
    for a in alerts.values():
        a.risk_score = score_alert(a.rule_counts)
        out.append(a)
    return out


def score_alert(rule_counts: dict[str, int]) -> float:
    """Each rule contributes its weight, scaled by how often it fired (saturates at 20 windows).
    One window gives half weight; 20+ windows give full weight. Sum is capped at 1.0."""
    total = 0.0
    for rule, n in rule_counts.items():
        frequency = 1.0 if rule in ONE_HIT_RULES else 0.5 + 0.5 * min(1.0, n / 20)
        total += WEIGHTS[rule] * frequency
    return round(min(1.0, total), 4)
