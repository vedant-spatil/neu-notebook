import pandas as pd

from triage_lib.triage_train import best_rate, flag_rate, rate_metrics


def _frame():
    # day 1: 10 rows, the top 2 by score are laundering; day 2: 20 rows, the top 4 are laundering
    rows = []
    for t, n, pos in ((1, 10, 2), (2, 20, 4)):
        for i in range(n):
            rows.append({"t": t, "score": float(n - i), "y": 1 if i < pos else 0})
    return pd.DataFrame(rows)


def test_flag_rate_flags_a_share_of_each_day_with_at_least_one():
    df = _frame()
    assert int(flag_rate(df, "score", 0.1).sum()) == 1 + 2  # 10% of 10 rows, 10% of 20 rows
    assert int(flag_rate(df, "score", 0.0001).sum()) == 2   # never fewer than one row a day


def test_rate_metrics_perfect_ranking_gives_f1_one():
    m = rate_metrics(_frame(), "score", 0.2)  # 2 flags on day 1, 4 on day 2
    assert (m["tp"], m["fp"], m["fn"]) == (6, 0, 0)
    assert m["f1"] == 1.0


def test_best_rate_picks_highest_f1():
    rate, table = best_rate(_frame(), "score", rates=(0.1, 0.2, 0.5))
    assert rate == 0.2  # F1 is 1.0 at 0.2; at 0.5 precision drops
    assert {r["rate"] for r in table} == {0.1, 0.2, 0.5}
