# Triage model training (HI-Large)

`train_triage_xgboost.ipynb` trains and evaluates the XGBoost triage model on IBM HI-Large, with walk-forward folds. It reads features written by the Spark job (`spark/build_features_spark.py`). The notebook never builds features itself at this scale.

## Contents

| Path | What it is | In git? |
|---|---|---|
| `train_triage_xgboost.ipynb` | Training and evaluation notebook (source of truth) | yes |
| `triage_lib/` | Feature, rule, training and explanation code (copies of `backend/app/ml`, changed for HI-Large) | yes |
| `spark/` | Spark feature job, R1 to R4 rule module, parity check | yes |
| `tests/` | Tests for `triage_lib` (leakage, equality, split, walk-forward) | yes |
| `eval/<run>/walk_forward.json` | Results of a run | yes |
| `data/`, `features_large/`, `features_manifest/`, `models/` | Local copies of S3 data and models | no |

## Data in S3

Bucket `neu-datasets`:

- `HI-Large_Trans.csv`: raw transactions (17 GB). Upload once with `aws s3 cp <file> s3://neu-datasets/ --profile <profile>`.
- `HI-Large_Patterns.txt`: optional, for per-typology recall.
- `features/hi_large/features/`: Spark output, partitioned by day (`t=1`, `t=2`, ...).
- `features/hi_large/manifest/features_manifest.json`: feature list and dataset facts.
- `triage/<run>/`: uploaded model, manifest, versions and results.

## Step 1: Spark feature job (SageMaker Processing)

The job runs PySpark on the raw CSV and writes the features. It needs about 300 GB of volume and several cores.

- Input: the raw CSV, mapped from `s3://neu-datasets/HI-Large_Trans.csv` to a local path in the container.
- Arguments: `--input <local csv> --output <local features dir> --manifest <local manifest path> --horizon 7 --partitions <N>`.
- Outputs: the features folder to `s3://neu-datasets/features/hi_large/features/`, and the manifest to `s3://neu-datasets/features/hi_large/manifest/`.

Before the full run, check the output on the HI-Small sample: `spark/parity_check.py` must report `PARITY OK`. The notebook's leakage check (section 6) also has to pass before training.

## Step 2: Notebook

1. Open a SageMaker notebook instance (`ml.m5.2xlarge` or larger, 100 GB volume). It needs an execution role with read and write on `neu-datasets`.
2. Clone this repo and open `notebooks/`.
3. Check the configuration (section 2): `RUN_NAME`, `HORIZON = 7`, `TEST_DAYS = 7`, `VAL_DAYS = 7`, `FOLDS = 4`, `BUDGET = 50` (placeholder).
4. Run all cells. Stop the instance when done.

Runtime is dominated by the walk-forward grid (12 points for each of 4 folds, plus the final retrain). Plan for hours, not minutes.

## Reading the results

- Each block reports the test positives. A block with fewer than 200 positives is not conclusive.
- The baseline is the app rule score restricted to R1 to R4. R5 to R7 need the whole edge set, which the Spark job does not compute.
- The validation PR-AUC uses sampled negatives. It is used to compare grid points, not as a precision estimate.
- The final row (`final`) is the result. The folds show whether it is stable.

## Tests

From `notebooks/`, with the backend venv:

```bash
/path/to/backend/mainvenv/bin/python -m pytest -q tests
```

## Limits

- HI-Large is synthetic, from the same generator as HI-Small. Real results will be lower.
- The budget (50 flagged account-days per day) is a placeholder until investigator capacity is known.
