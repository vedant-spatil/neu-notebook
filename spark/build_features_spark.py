"""Spark feature job for the triage model, for HI-Large scale. Same features, labels and definitions as
triage_lib/triage_features.py (build_features), with one difference in the baseline column.

Run as a SageMaker Processing job (PySpark container), or locally for the parity check:

    spark-submit build_features_spark.py --input <csv or s3 path> --output <dir> \
        --manifest <json path, outside the output folder> --horizon 7

Output: Parquet part files in --output (no other files in that folder, so pyarrow can read it as a dataset)
and the manifest JSON at --manifest. Load it in the notebook with triage_lib.triage_train.load_block.

Differences from the pandas build, on purpose:
- rule_score_app is the baseline from R1 to R4 only. R5 to R7 need the whole edge set, which the
  Spark job does not compute. The notebook reports this in the metrics.
- Ties in the rule windows are ordered by Spark, not by file order, so window counts on heavily tied
  timestamps can differ slightly from pandas. The model features are not affected: they use no rule
  window counts.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame, SparkSession, functions as F, types as T

from rules_r1_r4 import rule_windows

LOOKBACK_DAYS = 7
NEVER = 10**9
RULES = ["R1_structuring", "R2_fan_out", "R3_fan_in", "R4_pass_through"]
COUNT_FEATURES = ["out_n", "in_n", "out_amt_sum", "in_amt_sum", "out_amt_max", "in_amt_max",
                  "out_distinct", "in_distinct", "out_sub"]
DERIVED_FEATURES = ["out_mean", "in_mean", "out_fanout", "in_fanin", "pass_through",
                    "sub_share", "round_share", "night_share", "nonusd_share"]
RULE_FEATURES = [f"rule_{r}" for r in RULES] + ["rules_fired"]
RULE_SCHEMA = T.StructType([
    T.StructField("acct", T.StringType()),
    T.StructField("rule", T.StringType()),
    T.StructField("first_ts", T.TimestampType()),
    T.StructField("windows", T.LongType()),
])


def _slug(name: str) -> str:
    return re.sub(r"\W+", "_", str(name)).strip("_").lower()


CSV_COLUMNS = ["Timestamp", "From Bank", "from_acct", "To Bank", "to_acct", "Amount Received",
               "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering"]


def local_uri(path: str) -> str:
    """On a cluster, Spark reads plain paths from HDFS. SageMaker's /opt/ml paths are local, so use file://."""
    return f"file://{path}" if path.startswith("/") else path


def read_transactions(spark: SparkSession, path: str) -> DataFrame:
    # The file has two columns named "Account", which Spark renames on read. Read by position instead.
    raw = spark.read.csv(local_uri(path), header=False).toDF(*CSV_COLUMNS)
    raw = raw.filter(F.col("Timestamp") != "Timestamp")  # drop the header row
    return raw.select(
        F.to_timestamp("Timestamp", "yyyy/MM/dd HH:mm").alias("ts"),
        F.col("from_acct").alias("from_account"),
        F.col("to_acct").alias("to_account"),
        F.col("Amount Received").cast("double").alias("amount_received"),
        F.col("Amount Paid").cast("double").alias("amount_paid"),
        F.col("Receiving Currency").alias("rcur"),
        F.col("Payment Currency").alias("pcur"),
        F.col("Payment Format").alias("fmt"),
        (F.col("Is Laundering").cast("int") == 1).alias("lab"),
    )


def expand(df: DataFrame) -> DataFrame:
    """One row per window end t in day..day+6, for each daily row (a day counts toward the 7 windows ending on it)."""
    return df.withColumn("t", F.explode(F.sequence(F.col("day"), F.col("day") + (LOOKBACK_DAYS - 1))))


def pivot_sides(df: DataFrame, metrics: list[str]) -> DataFrame:
    """One row per (acct, t) with out_<metric> and in_<metric> columns. Each (acct, dir, t) is one row,
    so max() just picks that value; a missing side becomes null, filled later."""
    aggs = [F.max(F.when(F.col("dir") == d, F.col(m))).alias(f"{p}_{m}")
            for d, p in ((0, "out"), (1, "in")) for m in metrics]
    return df.groupBy("acct", "t").agg(*aggs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--partitions", type=int, default=400)
    args = ap.parse_args()
    if args.horizon < 1:
        raise SystemExit("horizon must be at least 1 day")

    spark = (SparkSession.builder.appName("triage_features")
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.sql.shuffle.partitions", args.partitions)
             .getOrCreate())
    # rules_r1_r4.py is shipped by spark-submit --py-files (see the SageMaker notebooks), so no addPyFile here.

    txn = read_transactions(spark, args.input)
    day0 = txn.agg(F.min(F.to_date("ts"))).first()[0]
    txn = txn.withColumn("day", (F.datediff(F.to_date("ts"), F.lit(day0)) + 1).cast("int"))
    max_day = int(txn.agg(F.max("day")).first()[0])
    usd = (F.col("rcur") == "US Dollar") & (F.col("pcur") == "US Dollar")
    txn = (txn.withColumn("usd", usd)
              .withColumn("night", (F.hour("ts") < 6) | (F.hour("ts") >= 22))
              .withColumn("nonusd", ~F.col("usd")))

    # Two legs per transaction: the sender (dir 0) and the receiver (dir 1), as in triage_features._legs.
    sender = txn.select(
        F.col("from_account").alias("acct"), F.col("to_account").alias("cp"), F.lit(0).alias("dir"),
        F.col("amount_paid").alias("amt"), "day", "fmt", "lab", "night", "nonusd", "usd", "ts",
        "amount_received", "amount_paid",
        ((F.col("fmt") == "Cash") & F.col("usd") & (F.col("amount_received") < 10_000)).alias("sub"),
    )
    receiver = txn.select(
        F.col("to_account").alias("acct"), F.col("from_account").alias("cp"), F.lit(1).alias("dir"),
        F.col("amount_received").alias("amt"), "day", "fmt", "lab", "night", "nonusd", "usd", "ts",
        "amount_received", "amount_paid", F.lit(False).alias("sub"),
    )
    legs = sender.unionByName(receiver).withColumn("rnd", (F.col("amt") > 0) & (F.col("amt") % 100 == 0))
    legs = legs.cache()

    # Windowed transaction features.
    daily = legs.groupBy("acct", "dir", "day").agg(
        F.count("*").alias("n"), F.sum("amt").alias("amt_sum"), F.max("amt").alias("amt_max"),
        F.sum(F.col("sub").cast("int")).alias("sub"), F.sum(F.col("rnd").cast("int")).alias("rnd"),
        F.sum(F.col("night").cast("int")).alias("night"), F.sum(F.col("nonusd").cast("int")).alias("nonusd"),
    )
    win = expand(daily).groupBy("acct", "dir", "t").agg(
        F.sum("n").alias("n"), F.sum("amt_sum").alias("amt_sum"), F.max("amt_max").alias("amt_max"),
        F.sum("sub").alias("sub"), F.sum("rnd").alias("rnd"), F.sum("night").alias("night"),
        F.sum("nonusd").alias("nonusd"),
    )
    side = pivot_sides(win, ["n", "amt_sum", "amt_max", "sub", "rnd", "night", "nonusd"])

    cp_days = legs.select("acct", "dir", "cp", "day").distinct()
    dist = (expand(cp_days).groupBy("acct", "dir", "t")
            .agg(F.countDistinct("cp").alias("distinct_cp")))
    dist_side = pivot_sides(dist, ["distinct_cp"]).withColumnRenamed("out_distinct_cp", "out_distinct") \
        .withColumnRenamed("in_distinct_cp", "in_distinct")

    fmt_rows = legs.select("fmt").distinct().collect()
    formats = sorted(r[0] for r in fmt_rows)
    fmt_daily = legs.groupBy("acct", "day", "fmt").count()
    fmt_win = expand(fmt_daily).groupBy("acct", "t", "fmt").agg(F.sum("count").alias("c"))
    fmt_wide = fmt_win.groupBy("acct", "t").pivot("fmt", formats).agg(F.sum("c"))
    fmt_cols = [f"fmt_{_slug(f)}" for f in formats]
    fmt_wide = fmt_wide.select("acct", "t", *[F.col(f"`{f}`").alias(c) for f, c in zip(formats, fmt_cols)])

    grid = side.join(dist_side, ["acct", "t"], "full").join(fmt_wide, ["acct", "t"], "full")
    grid = grid.na.fill(0)
    grid = grid.filter(F.col("t") <= max_day - args.horizon)

    def div(num, den):
        return num / F.greatest(den, F.lit(1.0))

    lo = F.least(F.col("out_amt_sum"), F.col("in_amt_sum"))
    hi = F.greatest(F.col("out_amt_sum"), F.col("in_amt_sum"))
    n_all = F.col("out_n") + F.col("in_n")
    grid = (grid
            .withColumn("out_mean", div(F.col("out_amt_sum"), F.col("out_n")))
            .withColumn("in_mean", div(F.col("in_amt_sum"), F.col("in_n")))
            .withColumn("out_fanout", div(F.col("out_distinct"), F.col("out_n")))
            .withColumn("in_fanin", div(F.col("in_distinct"), F.col("in_n")))
            .withColumn("pass_through", F.when(hi > 0, lo / F.when(hi > 0, hi).otherwise(F.lit(1.0)))
                        .otherwise(F.lit(0.0)))
            .withColumn("sub_share", div(F.col("out_sub"), F.col("out_n")))
            .withColumn("round_share", div(F.col("out_rnd") + F.col("in_rnd"), n_all))
            .withColumn("night_share", div(F.col("out_night") + F.col("in_night"), n_all))
            .withColumn("nonusd_share", div(F.col("out_nonusd") + F.col("in_nonusd"), n_all)))

    # Rule windows for R1 to R4, one account at a time (rules_r1_r4.rule_windows).
    # Plain RDD grouping, not applyInPandas: the SageMaker Spark image has no pyarrow, which applyInPandas needs.
    rule_src = legs.select("acct", "cp", "dir", "ts", "amount_received", "amount_paid", "fmt", "usd")
    rule_rdd = (rule_src.rdd
                .map(lambda r: (r["acct"], (r["acct"], r["cp"], r["dir"], r["ts"], r["amount_received"],
                                            r["amount_paid"], r["fmt"], r["usd"])))
                .groupByKey()
                .flatMap(_rules_for_group))
    rules = spark.createDataFrame(rule_rdd, schema=RULE_SCHEMA)
    rules = (rules
             .withColumn("fire_day", (F.datediff(F.to_date(F.col("first_ts") + F.expr("INTERVAL 72 HOURS")),
                                                 F.lit(day0)) + 1).cast("int"))
             .withColumn("weight", F.when(F.col("rule") == "R1_structuring", 0.15)
                         .when(F.col("rule") == "R2_fan_out", 0.3)
                         .when(F.col("rule") == "R3_fan_in", 0.6)
                         .otherwise(0.15))
             .withColumn("score", F.round(F.least(F.lit(1.0),
                                                  F.col("weight") * (F.lit(0.5) + F.lit(0.5) *
                                                                     F.least(F.lit(1.0), F.col("windows") / 20.0))),
                                          4)))
    rules_wide = rules.groupBy("acct").pivot("rule", RULES).agg(F.max("fire_day").alias("fire"),
                                                               F.max("score").alias("score"))
    # Fire days stay null for rules that never fired (coalesced to NEVER below). Scores default to 0.
    score_cols = [c for c in rules_wide.columns if c.endswith("_score")]
    rules_wide = rules_wide.na.fill(0.0, score_cols)
    grid = grid.join(rules_wide, "acct", "left")

    for r in RULES:
        fire = F.coalesce(F.col(f"`{r}_fire`"), F.lit(NEVER))
        grid = grid.withColumn(f"rule_{r}", F.when(F.col("t") >= fire, 1).otherwise(0).cast("tinyint"))
    grid = grid.withColumn("rules_fired", sum(F.col(f"rule_{r}") for r in RULES).cast("tinyint"))
    grid = grid.withColumn("rule_score_app", F.greatest(*[
        F.when(F.col("t") >= F.coalesce(F.col(f"`{r}_fire`"), F.lit(NEVER)),
               F.coalesce(F.col(f"`{r}_score`"), F.lit(0.0))).otherwise(F.lit(0.0)) for r in RULES]))

    # Labels: account is in a laundering transaction on days t+1 .. t+horizon.
    pos = legs.filter(F.col("lab")).select("acct", "day").distinct()
    lab = (pos.withColumn("t", F.explode(F.sequence(F.col("day") - args.horizon, F.col("day") - 1)))
              .select("acct", "t").distinct().withColumn("lab_flag", F.lit(1)))
    grid = grid.join(lab, ["acct", "t"], "left")
    grid = grid.withColumn("y", F.coalesce(F.col("lab_flag"), F.lit(0)).cast("tinyint")).drop("lab_flag")
    grid = grid.withColumn("t_date", F.date_add(F.lit(day0), F.col("t") - 1))

    features = COUNT_FEATURES + DERIVED_FEATURES + fmt_cols + RULE_FEATURES
    out = grid.select("acct", "t", "t_date", *features, "rule_score_app", "y")
    # Partitioned by window day t (folders t=1, t=2, ...), so the notebook reads one day at a time.
    out.write.mode("overwrite").partitionBy("t").parquet(local_uri(args.output))

    totals = spark.read.parquet(local_uri(args.output)).agg(F.count("*").alias("rows"), F.sum("y").alias("positives")).first()
    manifest = {
        "day0": day0.isoformat(),
        "max_day": max_day,
        "horizon": args.horizon,
        "lookback_days": LOOKBACK_DAYS,
        "features": features,
        "formats": [_slug(f) for f in formats],
        "rows": int(totals["rows"]),
        "positives": int(totals["positives"]),
        "engine": "spark",
        "baseline_rules": RULES,
    }
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"features written: rows={manifest['rows']} positives={manifest['positives']} max_day={max_day}", flush=True)
    spark.stop()


def _rules_for_account(pdf: pd.DataFrame) -> pd.DataFrame:
    """Runs on one account's legs. Rebuilds that account's USD transactions (each once, self-transfers included)."""
    acct = pdf["acct"].iloc[0]
    usd_legs = pdf[pdf["usd"]]
    sent = usd_legs[usd_legs["dir"] == 0]
    received = usd_legs[(usd_legs["dir"] == 1) & (usd_legs["cp"] != acct)]
    txns = pd.concat([
        pd.DataFrame({"from_account": acct, "to_account": sent["cp"].to_numpy(), "ts": sent["ts"].to_numpy(),
                      "amount_received": sent["amount_received"].to_numpy(),
                      "amount_paid": sent["amount_paid"].to_numpy(), "payment_format": sent["fmt"].to_numpy()}),
        pd.DataFrame({"from_account": received["cp"].to_numpy(), "to_account": acct,
                      "ts": received["ts"].to_numpy(), "amount_received": received["amount_received"].to_numpy(),
                      "amount_paid": received["amount_paid"].to_numpy(),
                      "payment_format": received["fmt"].to_numpy()}),
    ], ignore_index=True)
    if txns.empty:
        return pd.DataFrame(columns=["acct", "rule", "first_ts", "windows"])
    found = rule_windows(txns)
    rows = [(a, rule, pd.Timestamp(ts).to_pydatetime(), int(n)) for (a, rule), (ts, n) in found.items() if a == acct]
    return pd.DataFrame(rows, columns=["acct", "rule", "first_ts", "windows"])


def _rules_for_group(kv):
    """RDD version of _rules_for_account: one account's rows in, plain tuples out."""
    acct, values = kv
    pdf = pd.DataFrame(list(values), columns=["acct", "cp", "dir", "ts", "amount_received", "amount_paid",
                                              "fmt", "usd"])
    return [(a, rule, pd.Timestamp(t).to_pydatetime(), int(n))
            for a, rule, t, n in _rules_for_account(pdf).itertuples(index=False, name=None)]


if __name__ == "__main__":
    main()
