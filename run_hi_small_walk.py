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

pooled = {"model": {"tp": 0, "fp": 0, "positives": 0}, "rules": {"tp": 0, "fp": 0, "positives": 0}}
for block in result["folds"] + [result["final"]]:
    m, r = block["model"], block["rules_app"]
    pooled["model"]["tp"] += m["tp"]
    pooled["model"]["fp"] += m["fp"]
    pooled["model"]["positives"] += m["positives"]
    pooled["rules"]["tp"] += r["tp"]
    pooled["rules"]["fp"] += r["fp"]
    pooled["rules"]["positives"] += r["positives"]
    print(f"{block['name']}: model tp={m['tp']} of {m['positives']} lift={m['lift']} | "
          f"rules tp={r['tp']} of {r['positives']} lift={r['lift']} | conclusive={block['conclusive']}")
for key in ("model", "rules"):
    p = pooled[key]
    flagged = p["tp"] + p["fp"]
    print(f"POOLED {key}: tp={p['tp']} of {p['positives']} positives, flagged={flagged}, "
          f"precision={p['tp'] / max(flagged, 1):.4f}")
print("summary across folds:", result["summary"])
print("final block:", result["final"]["name"], "conclusive:", result["final"]["conclusive"])

if UPLOAD_RESULTS:
    s3 = boto3.client("s3")
    for path in (EVAL_DIR / "walk_forward.json", RUN_DIR / "model.json", RUN_DIR / MANIFEST_FILE):
        if path.exists():
            s3.upload_file(str(path), S3_BUCKET, f"triage/{RUN_NAME}/{path.name}")
            print(f"uploaded s3://{S3_BUCKET}/triage/{RUN_NAME}/{path.name}")
