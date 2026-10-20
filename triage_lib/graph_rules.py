"""Graph rules R5 to R7 (cycle, scatter-gather, gather-scatter). Pure pandas, USD edges only.

  R5 cycle            directed cycle of length 3 or 4, every hop inside WINDOW
  R6 scatter-gather   source a sends to >=3 intermediates, >=3 of which send to one sink c, inside WINDOW
  R7 gather-scatter   hub b receives from >=3 sources and sends to >=3 targets, inside WINDOW

Edges are USD rows (same filter as R1 to R4). Two-hop paths a->b->c are built first; R5 to R7 reuse them.
Nodes with more than DEGREE_CAP in or out edges are skipped to bound the joins. Each group found is
yielded once as (account, rule, ts, amount, txn_ids, is_laundering) for build_alerts() to merge.
"""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

WINDOW = pd.Timedelta(hours=72)
DEGREE_CAP = 500
MIN_INTERMEDIATES = 3

Hit = tuple[str, str, pd.Timestamp, float, list[int], bool]


def _edges(usd: pd.DataFrame) -> pd.DataFrame:
    e = pd.DataFrame(
        {
            "src": usd["from_account"].values,
            "dst": usd["to_account"].values,
            "t": usd["ts"].values,
            "amt": usd["amount_received"].astype(float).values,
            "id": usd["txn_id"].astype(int).values,
            "lab": usd["is_laundering"].astype(bool).values,
        }
    )
    e = e[e["src"] != e["dst"]]
    out_deg = e.groupby("src")["id"].transform("size")
    in_deg = e.groupby("dst")["id"].transform("size")
    return e[(out_deg <= DEGREE_CAP) & (in_deg <= DEGREE_CAP)].reset_index(drop=True)


def _leg(e: pd.DataFrame, src: str, dst: str, k: str) -> pd.DataFrame:
    return e.rename(columns={"src": src, "dst": dst, "t": f"t{k}", "amt": f"amt{k}", "id": f"id{k}", "lab": f"lab{k}"})


def _two_hop(e: pd.DataFrame) -> pd.DataFrame:
    p = _leg(e, "a", "b", "1").merge(_leg(e, "b", "c", "2"), on="b")
    p = p[(p["t2"] >= p["t1"]) & (p["t2"] <= p["t1"] + WINDOW) & (p["a"] != p["c"])]
    return p.reset_index(drop=True)


def _cycles(e: pd.DataFrame, p: pd.DataFrame) -> Iterator[Hit]:
    # 3-cycles: a->b->c->a
    c3 = p.merge(_leg(e, "c", "a", "3"), on=["c", "a"])
    c3 = c3[(c3["t3"] >= c3["t2"]) & (c3["t3"] <= c3["t1"] + WINDOW)]
    seen: set[tuple] = set()
    for row in c3.itertuples(index=False):
        ids = (row.id1, row.id2, row.id3)
        key = tuple(sorted(ids))
        if key in seen:
            continue
        seen.add(key)
        yield from _cycle_hits([row.a, row.b, row.c], ids, row.t1, row.amt1, row.lab1 or row.lab2 or row.lab3)

    # 4-cycles: a->b->c->d->a = a two-hop path a->b->c plus a two-hop path c->d->a.
    # Pair the two-hop paths on their endpoints instead of extending each path by one edge.
    # The old join multiplied paths by out-degree and ran out of memory on HI-Small.
    q = p.rename(columns={"a": "x", "b": "d", "c": "y", "t1": "t3", "t2": "t4", "id1": "id3", "id2": "id4",
                          "amt1": "amt3", "lab1": "lab3", "lab2": "lab4"})
    m = p.merge(q, left_on=["a", "c"], right_on=["y", "x"])
    m = m[(m["d"] != m["a"]) & (m["d"] != m["b"]) & (m["t3"] >= m["t2"]) & (m["t4"] <= m["t1"] + WINDOW)]
    print(f"[graph] four-cycle candidates={len(m)}", flush=True)
    for row in m.itertuples(index=False):
        ids = (row.id1, row.id2, row.id3, row.id4)
        key = tuple(sorted(ids))
        if key in seen:
            continue
        seen.add(key)
        yield from _cycle_hits([row.a, row.b, row.c, row.d], ids, row.t1, row.amt1,
                               row.lab1 or row.lab2 or row.lab3 or row.lab4)


def _cycle_hits(members: list[str], ids: tuple, ts: pd.Timestamp, amount: float, lab: bool) -> Iterator[Hit]:
    # Every account in the cycle gets the alert (one per account and rule).
    for acct in members:
        yield (acct, "R5_cycle", ts, float(amount), [int(i) for i in ids], bool(lab))


def _scatter_gather(p: pd.DataFrame) -> Iterator[Hit]:
    # Source a and sink c with >=3 distinct intermediates on a->b->c paths.
    counts = p.groupby(["a", "c"])["b"].nunique().rename("n").reset_index()
    hot = p.merge(counts[counts["n"] >= MIN_INTERMEDIATES][["a", "c"]], on=["a", "c"])
    for (a, c), legs in hot.groupby(["a", "c"]):
        ids = list(dict.fromkeys(legs[["id1", "id2"]].values.ravel().tolist()))
        lab = bool(legs[["lab1", "lab2"]].values.any())
        ts = legs["t1"].min()
        amount = float(legs["amt1"].sum())
        for acct in (a, c):
            yield (acct, "R6_scatter_gather", ts, amount, [int(i) for i in ids], lab)


def _gather_scatter(p: pd.DataFrame) -> Iterator[Hit]:
    # Hub b with >=3 distinct sources (a) and >=3 distinct targets (c) on its paths.
    stats = p.groupby("b").agg(na=("a", "nunique"), nc=("c", "nunique")).reset_index()
    hubs = stats[(stats["na"] >= MIN_INTERMEDIATES) & (stats["nc"] >= MIN_INTERMEDIATES)][["b"]]
    hot = p.merge(hubs, on="b")
    for hub, legs in hot.groupby("b"):
        ids = list(dict.fromkeys(legs[["id1", "id2"]].values.ravel().tolist()))
        lab = bool(legs[["lab1", "lab2"]].values.any())
        yield (hub, "R7_gather_scatter", legs["t1"].min(), float(legs["amt1"].sum()), [int(i) for i in ids], lab)


def graph_hits(usd: pd.DataFrame) -> Iterator[Hit]:
    """Yield every R5 to R7 group hit from the USD rows."""
    e = _edges(usd)
    print(f"[graph] edges after degree cap={len(e)}", flush=True)
    if e.empty:
        return
    p = _two_hop(e)
    print(f"[graph] two-hop paths={len(p)}", flush=True)
    if p.empty:
        return
    yield from _cycles(e, p)
    print("[graph] cycles done", flush=True)
    yield from _scatter_gather(p)
    print("[graph] scatter-gather done", flush=True)
    yield from _gather_scatter(p)
    print("[graph] gather-scatter done", flush=True)
