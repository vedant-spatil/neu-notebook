"""Walk-forward on HI-Small, using the features already built by run_hi_small_train.py.

Five non-overlapping 2-day test blocks (days 8 to 17), so the test positives can be pooled. Each block gets its
own model, trained on the days before it. The grid point is chosen on validation, then one final model is scored.

Run from the notebooks/ folder, after run_hi_small_train.py has written features_small/:
    TRIAGE_WORKERS=1 python run_hi_small_walk.py 2>&1 | tee hi_small_walk.log
"""

from pathlib import Path
import json
import sys
import time

import boto3
import pandas as pd

NB_DIR = Path.cwd()  # run from the notebooks/ folder
sys.path.insert(0, str(NB_DIR))

from triage_lib.triage_features import FEATURES_FILE, MANIFEST_FILE  # noqa: E402
from triage_lib.triage_train import walk_forward  # noqa: E402

S3_BUCKET = "neu-datasets"
RUN_NAME = "triage_hi_small_walk"
HORIZON = 1
FOLDS = 4          # earlier folds, plus the final block: 5 test blocks in total
TEST_DAYS = 2
VAL_DAYS = 2
BUDGET = 50        # placeholder, same as the single-split run
UPLOAD_RESULTS = True

FEATURES_DIR = NB_DIR / "features_small"
PATTERNS = NB_DIR / "data" / "HI-Small_Patterns.txt"
RUN_DIR = NB_DIR / "models" / RUN_NAME
EVAL_DIR = NB_DIR / "eval" / RUN_NAME

manifest = json.loads((FEATURES_DIR / MANIFEST_FILE).read_text(encoding="utf-8"))
max_t = int(manifest["max_day"]) - HORIZON  # labels need HORIZON observable days after t
day0 = pd.Timestamp(manifest["day0"])
features = manifest["features"]

t0 = time.time()
result = walk_forward(
    path=FEATURES_DIR / FEATURES_FILE,   # one flat file: load_block filters on t
    features=features,
    horizon=HORIZON,
    max_t=max_t,
    folds=FOLDS,
    test_days=TEST_DAYS,
    val_days=VAL_DAYS,
    budget=BUDGET,
    out_dir=EVAL_DIR,
    model_dir=RUN_DIR,
    manifest_path=FEATURES_DIR / MANIFEST_FILE,
    day0=day0,
    patterns_path=PATTERNS if PATTERNS.exists() else None,
)
print(f"walk-forward finished in {(time.time() - t0) / 60:.1f} min")
print("selected params:", result["selected_params"])
print("plans (test blocks):", [(p["name"], p["test_start"], p["test_end"]) for p in result["plans"]])

blocks = result["folds"] + [result["final"]]
for block in blocks:
    m, r = block["model_at_rate"], block["rules_at_rate"]
    print(f"{block['name']}: rate={block['chosen_rate']} model tp={m['tp']} fp={m['fp']} of {m['positives']} "
          f"f1={m['f1']} | rules f1={r['f1']} | conclusive={block['conclusive']}")
print("rate grid on validation (final block's probe):", [(g["rate"], g["f1"]) for g in result["final"]["rate_grid_val"]])
for key in ("model_at_rate", "rules_at_rate"):
    tp = sum(b[key]["tp"] for b in blocks)
    fp = sum(b[key]["fp"] for b in blocks)
    fn = sum(b[key]["fn"] for b in blocks)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    print(f"POOLED {key}: tp={tp} fp={fp} fn={fn} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
print("summary across folds:", result["summary"])
print("final block:", result["final"]["name"], "conclusive:", result["final"]["conclusive"])

if UPLOAD_RESULTS:
    s3 = boto3.client("s3")
    for path in (EVAL_DIR / "walk_forward.json", RUN_DIR / "model.json", RUN_DIR / MANIFEST_FILE):
        if path.exists():
            s3.upload_file(str(path), S3_BUCKET, f"triage/{RUN_NAME}/{path.name}")
            print(f"uploaded s3://{S3_BUCKET}/triage/{RUN_NAME}/{path.name}")
