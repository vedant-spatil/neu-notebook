"""Parity check: Spark feature job output vs the pandas build (triage_lib.triage_features.build_features).

    python parity_check.py --csv <csv> --spark-dir <spark output folder> --manifest <spark manifest json> --horizon 7

Compares, on the same CSV: row keys (acct, t), every model feature, the label y, and the baseline
rule_score_app against a pandas baseline that uses R1 to R4 only. Exits non-zero if anything differs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))         # SageMaker job folder: triage_lib is bundled next to this file
sys.path.insert(0, str(HERE.parent))  # local layout: notebooks/spark -> notebooks/triage_lib

from triage_lib.alerts import build_alerts, load_csv  # noqa: E402
from triage_lib.triage_features import (  # noqa: E402
    MODEL_RULES,
    RULES,
    _rule_arrays,
    build_features,
)

FLOAT_TOL = 1e-6


def pandas_reference(csv: Path, horizon: int) -> tuple[pd.DataFrame, dict]:
    """One pass: R1 to R4 alerts only (graph rules skipped, they are not used by the Spark job), then the
    feature build with those alerts. The baseline column is therefore the R1 to R4 baseline, like Spark's."""
    df = load_csv(str(csv))
    alerts = build_alerts(df, include_graph=False)
    accounts = pd.Index(pd.unique(pd.concat([df["from_account"], df["to_account"]], ignore_index=True)))
    day0 = pd.Timestamp(df["ts"].min().normalize())
    ts_by_txn = pd.Series(df["ts"].to_numpy(), index=df["txn_id"].to_numpy())
    rule_arrays = _rule_arrays(alerts, {str(a): i for i, a in enumerate(accounts)}, day0, len(accounts), ts_by_txn)
    frame, manifest = build_features(df, horizon, accounts=accounts, rule_arrays=rule_arrays)
    return frame.assign(rule_score_app_r14=frame["rule_score_app"]), manifest


def compare(pandas_frame: pd.DataFrame, spark: pd.DataFrame, features: list[str]) -> list[str]:
    problems: list[str] = []
    key = ["acct", "t"]
    p = pandas_frame.set_index(key).sort_index()
    s = spark.set_index(key).sort_index()
    if not p.index.equals(s.index):
        missing = len(p.index.difference(s.index))
        extra = len(s.index.difference(p.index))
        problems.append(f"row keys differ: pandas-only={missing} spark-only={extra}")
        common = p.index.intersection(s.index)
        p, s = p.loc[common], s.loc[common]
    for col in features:
        a = p[col].to_numpy(float)
        b = s[col].to_numpy(float)
        bad = ~np.isclose(a, b, rtol=FLOAT_TOL, atol=FLOAT_TOL)
        if bad.any():
            problems.append(f"{col}: {int(bad.sum())} rows differ (first: pandas={a[bad][0]} spark={b[bad][0]})")
    y_bad = int((p["y"].to_numpy() != s["y"].to_numpy()).sum())
    if y_bad:
        problems.append(f"y: {y_bad} rows differ")
    base_bad = ~np.isclose(p["rule_score_app_r14"].to_numpy(float), s["rule_score_app"].to_numpy(float),
                           rtol=FLOAT_TOL, atol=1e-4)
    if base_bad.any():
        problems.append(f"rule_score_app (R1-R4): {int(base_bad.sum())} rows differ "
                        f"(tie ordering can cause small count differences; see the Spark job docstring)")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--spark-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--verdict", type=Path, default=None, help="write PASS or FAIL here (read by spark_notebook)")
    args = ap.parse_args()

    spark_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    features = spark_manifest["features"]
    pandas_frame, pandas_manifest = pandas_reference(args.csv, args.horizon)
    if pandas_manifest["features"] != features:
        print("feature lists differ:\n pandas:", pandas_manifest["features"], "\n spark:", features)
        raise SystemExit(1)
    spark = pd.read_parquet(args.spark_dir)
    print(f"pandas rows={len(pandas_frame)} positives={int(pandas_frame['y'].sum())}; "
          f"spark rows={len(spark)} positives={int(spark['y'].sum())}")
    problems = compare(pandas_frame, spark, features)
    if args.verdict is not None:
        args.verdict.parent.mkdir(parents=True, exist_ok=True)
        status = "FAIL" if problems else "PASS"
        args.verdict.write_text(status + "\n" + "\n".join(problems) + "\n", encoding="utf-8")
    if problems:
        print("PARITY FAILED:")
        for line in problems:
            print(" -", line)
        raise SystemExit(1)
    print(f"PARITY OK: {len(features)} features, labels and keys match.")


if __name__ == "__main__":
    main()
