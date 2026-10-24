"""Train the triage model on HI-Small with the single-split path (run_train).

Run from the notebooks/ folder, in a Studio notebook cell:  %run run_hi_small_train.py
or in a terminal:  python run_hi_small_train.py

Steps: download HI-Small from S3 (if missing), build features for the whole file, write one
features.parquet + manifest, train with run_train (default 60/20/20 day split), upload results.
"""

from pathlib import Path
import json
import os
import sys
import time

import boto3

NB_DIR = Path.cwd()  # run from the notebooks/ folder
sys.path.insert(0, str(NB_DIR))
# build_alerts defaults to one worker per CPU (16 on ml.r5.4xlarge). Each forked worker copies part of the
# full table, so parallel mode blows up memory. Serial is the safe setting for this run.
os.environ.setdefault("TRIAGE_WORKERS", "1")

from triage_lib.alerts import load_csv  # noqa: E402
from triage_lib.triage_features import build_features, write_features  # noqa: E402
from triage_lib.triage_train import ROUNDS, run_train  # noqa: E402

S3_BUCKET = "neu-datasets"
RUN_NAME = "triage_hi_small_single"   # new name, so the earlier HI-Small results are kept
HORIZON = 1                           # label horizon in days (HI-Small is about 10 days)
BUDGET = 50                           # placeholder: flagged account-days per day
UPLOAD_RESULTS = True

DATA_DIR = NB_DIR / "data"
CSV = DATA_DIR / "HI-Small_Trans.csv"
PATTERNS = DATA_DIR / "HI-Small_Patterns.txt"
FEATURES_DIR = NB_DIR / "features_small"
RUN_DIR = NB_DIR / "models" / RUN_NAME
EVAL_DIR = NB_DIR / "eval" / RUN_NAME
for folder in (DATA_DIR, FEATURES_DIR, RUN_DIR, EVAL_DIR):
    folder.mkdir(parents=True, exist_ok=True)

s3 = boto3.client("s3")
for path in (CSV, PATTERNS):
    if not path.exists():
        s3.download_file(S3_BUCKET, path.name, str(path))
        print(f"downloaded {path.name}")

# 1. Features for the whole file.
t0 = time.time()
df = load_csv(str(CSV))
print(f"rows: {len(df):,}  span: {df['ts'].min()} to {df['ts'].max()}  ({time.time() - t0:.0f}s)")
frame, manifest = build_features(df, HORIZON)
del df
write_features(frame, manifest, FEATURES_DIR)
print(f"features: {manifest['rows']:,} rows, {manifest['positives']:,} positives, "
      f"{len(manifest['features'])} features, days 1..{manifest['max_day']}  ({(time.time() - t0) / 60:.1f} min)")
del frame

# 2. Train on the single split (train days, validation days, test days, with a gap of HORIZON days).
t0 = time.time()
metrics = run_train(FEATURES_DIR, RUN_DIR, BUDGET, patterns_path=PATTERNS)
print(f"trained in {(time.time() - t0) / 60:.1f} min")
print("split:", {k: metrics[k] for k in ("train_end", "val_end", "rows", "positives")})
print("best params:", metrics["best_params"], " best iteration:", metrics["best_iteration"],
      " val PR-AUC:", metrics["val_aucpr"])
if metrics["best_iteration"] >= ROUNDS - 1:
    print(f"WARNING: best iteration hit the {ROUNDS}-round cap. The model may be underfitting; raise ROUNDS.")
print("test, model:", metrics["test"]["model"])
print("test, rules:", metrics["test"]["rules_app"])
print("typology recall:", metrics["typology_recall_test"])

# 3. Save the results next to the model, then upload.
(EVAL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
if UPLOAD_RESULTS:
    for name in ("model.json", "features_manifest.json", "metrics.json", "versions.json"):
        src = RUN_DIR / name if name != "metrics.json" else EVAL_DIR / name
        if src.exists():
            s3.upload_file(str(src), S3_BUCKET, f"triage/{RUN_NAME}/{name}")
            print(f"uploaded s3://{S3_BUCKET}/triage/{RUN_NAME}/{name}")
