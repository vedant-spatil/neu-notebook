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

Parallel: R1 to R4 run one account at a time in worker processes (fork). Each account's windows depend only on its
own transactions, and the results are merged in the same order as the serial loop, so the alerts are identical
to a serial run. Set workers=1 (or TRIAGE_WORKERS=1) for the serial path. R5 to R7 are unchanged.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import resource
import time
from bisect import bisect_right
from dataclasses import dataclass, field

import numpy as np
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
H24_NS = int(H24.value)
H48_NS = int(H48.value)
MAX_TXN_IDS = 200

# Read by worker processes. Set in the parent before the pool forks, so the children inherit it.
_WORKER_DATA: dict = {}


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


def _default_workers() -> int:
    return max(1, int(os.environ.get("TRIAGE_WORKERS", os.cpu_count() or 1)))


def _groups(frame: pd.DataFrame, key_col: str) -> tuple[pd.DataFrame, list[tuple[str, int, int]]]:
    """Rows sorted by ts, then grouped by key. Same order as sort_values("ts").groupby(key) in the serial code:
    the second sort is stable, so ts order holds inside each key. Returns (sorted frame, [(key, lo, hi)])."""
    s = frame.sort_values("ts").sort_values(key_col, kind="mergesort")
    keys = s[key_col].to_numpy()
    if len(keys) == 0:
        return s, []
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    ends = np.r_[starts[1:], len(keys)]
    return s, [(str(keys[lo]), int(lo), int(hi)) for lo, hi in zip(starts, ends)]


def _run(fn, n: int, workers: int) -> list:
    """fn(i) for i in range(n), in order. Forked worker processes when workers > 1."""
    if workers <= 1 or n < 2 * workers:
        return [fn(i) for i in range(n)]
    with mp.get_context("fork").Pool(workers) as pool:
        return pool.map(fn, range(n), chunksize=max(1, n // (workers * 8)))


# ---- R1: cash structuring, per sender ---------------------------------------------------------------------------

def _r1_account(gi: int) -> tuple[str, list]:
    d = _WORKER_DATA["r1"]
    acct, lo, hi = d["groups"][gi]
    if hi - lo < 3:
        return acct, []
    times = d["ts"][lo:hi].tolist()
    ids = d["ids"][lo:hi].tolist()
    amts = d["amt"][lo:hi].tolist()
    labs = d["lab"][lo:hi].tolist()
    hits = []
    for i in range(len(times)):
        j = bisect_right(times, times[i] + H24_NS)
        if j - i >= 3:
            hits.append((times[i], sum(amts[i:j]), ids[i:j], any(labs[i:j])))
    return acct, hits


# ---- R2 fan-out / R3 fan-in: distinct counterparties within 48 h ----------------------------------------------

def _fan_worker(gi: int) -> tuple[str, list]:
    """Module-level so it can be sent to worker processes; the dataset (r2 or r3) is set in _WORKER_DATA["fan"]."""
    return _fan_account(_WORKER_DATA["fan"], gi)


def _fan_account(name: str, gi: int) -> tuple[str, list]:
    d = _WORKER_DATA[name]
    acct, lo, hi = d["groups"][gi]
    others_all = d["others"][lo:hi].tolist()
    if len(set(others_all)) < 10:
        return acct, []
    times = d["ts"][lo:hi].tolist()
    others = others_all
    ids = d["ids"][lo:hi].tolist()
    amts = d["amt"][lo:hi].tolist()
    labs = d["lab"][lo:hi].tolist()
    hits = []
    from collections import Counter
    counts: Counter = Counter()
    j = 0
    for i in range(len(times)):
        while j < len(times) and times[j] <= times[i] + H48_NS:
            counts[others[j]] += 1
            j += 1
        if len(counts) >= 10:
            hits.append((times[i], sum(amts[i:j]), ids[i:j], any(labs[i:j])))
        counts[others[i]] -= 1
        if counts[others[i]] == 0:
            del counts[others[i]]
    return acct, hits


# ---- R4 pass-through: receipts with >=80% sent out within 48 h -------------------------------------------------

def _r4_account(gi: int) -> tuple[str, list]:
    d = _WORKER_DATA["r4"]
    acct, lo, hi = d["in_groups"][gi]
    if acct not in d["out_index"]:
        return acct, []
    olo, ohi = d["out_index"][acct]
    out_times = d["out_ts"][olo:ohi].tolist()
    out_cum = d["out_cum"][olo:ohi].tolist()
    hits = []
    for t0, received, txn_id, lab in zip(d["in_ts"][lo:hi].tolist(), d["in_amt"][lo:hi].tolist(),
                                         d["in_ids"][lo:hi].tolist(), d["in_lab"][lo:hi].tolist()):
        a = bisect_right(out_times, t0)
        b = bisect_right(out_times, t0 + H48_NS)
        if b <= a:
            continue
        sent = out_cum[b - 1] - (out_cum[a - 1] if a > 0 else 0.0)
        if received > 0 and sent >= 0.8 * received:
            hits.append((t0, float(received), [int(txn_id)], bool(lab)))
    return acct, hits


def build_alerts(df: pd.DataFrame, include_graph: bool = True, workers: int | None = None) -> list[Alert]:
    started = time.perf_counter()
    workers = _default_workers() if workers is None else max(1, workers)
    usd = _usd(df)
    print(f"[alerts] usd rows={len(usd)} of {len(df)} workers={workers}", flush=True)
    alerts: dict[tuple[str, str], Alert] = {}

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

    def merge(acct_hits: list, rule: str) -> None:
        for acct, hits in acct_hits:
            for ts_ns, amount, ids, lab in hits:
                add(acct, rule, pd.Timestamp(ts_ns), amount, ids, lab)

    # R1 structuring: Cash, USD, under $10k, >=3 within 24h, per sender.
    cash = usd[(usd["payment_format"] == "Cash") & (usd["amount_received"] < 10_000)]
    s, groups = _groups(cash, "from_account")
    _WORKER_DATA["r1"] = {
        "groups": groups,
        "ts": s["ts"].astype("int64").to_numpy(),
        "ids": s["txn_id"].to_numpy(),
        "amt": s["amount_received"].to_numpy(float),
        "lab": s["is_laundering"].to_numpy(bool),
    }
    merge(_run(_r1_account, len(groups), workers), "R1_structuring")

    # R2 fan-out (per sender) and R3 fan-in (per receiver): distinct counterparties within 48h.
    for name, key_col, other_col, rule in (
        ("r2", "from_account", "to_account", "R2_fan_out"),
        ("r3", "to_account", "from_account", "R3_fan_in"),
    ):
        s, groups = _groups(usd, key_col)
        _WORKER_DATA[name] = {
            "groups": groups,
            "ts": s["ts"].astype("int64").to_numpy(),
            "others": s[other_col].to_numpy(object),
            "ids": s["txn_id"].to_numpy(),
            "amt": s["amount_received"].to_numpy(float),
            "lab": s["is_laundering"].to_numpy(bool),
        }
        _WORKER_DATA["fan"] = name
        merge(_run(_fan_worker, len(groups), workers), rule)
        _progress(f"{rule} done", started)

    # R4 pass-through: in-amount, >=80% sent out within 48h after receipt.
    s_in, in_groups = _groups(usd, "to_account")
    s_out, out_groups = _groups(usd, "from_account")
    out_amount = s_out["amount_paid"].to_numpy(float)
    out_cum = np.empty(len(out_amount))
    out_index: dict[str, tuple[int, int]] = {}
    for acct, lo, hi in out_groups:
        out_cum[lo:hi] = np.cumsum(out_amount[lo:hi])
        out_index[acct] = (lo, hi)
    _WORKER_DATA["r4"] = {
        "in_groups": in_groups,
        "in_ts": s_in["ts"].astype("int64").to_numpy(),
        "in_amt": s_in["amount_received"].to_numpy(float),
        "in_ids": s_in["txn_id"].to_numpy(),
        "in_lab": s_in["is_laundering"].to_numpy(bool),
        "out_ts": s_out["ts"].astype("int64").to_numpy(),
        "out_cum": out_cum,
        "out_index": out_index,
    }
    merge(_run(_r4_account, len(in_groups), workers), "R4_pass_through")
    _progress("R4 done", started)

    # R5 to R7: graph groups over the USD edges (unchanged; graph_rules.py).
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
