# Triage model training notebook

`train_triage_xgboost.ipynb` trains the XGBoost triage model on IBM HI-Small. It builds account-day features, splits by time (75/25 development/test, with validation as the last 20% of development), trains with early stopping, and evaluates once on the test block.

## Contents

| Path | What it is | In git? |
|---|---|---|
| `train_triage_xgboost.ipynb` | The training notebook (source of truth) | yes |
| `triage_lib/` | Copied feature, rule, training, and explanation code | yes |
| `tests/` | Leakage, split, resume, and equality tests for `triage_lib` | yes |
| `eval/<run>/metrics.json` | Metrics of each run | yes |
| `data/` | `HI-Small_Trans.csv` (about 454 MiB) | **no**: stays in S3 |
| `features/` | Feature cache (Parquet, rule pass) | no |
| `models/<run>/model.json` | Trained model | no: copied to S3 |

## Data in S3

The bucket is `neu-datasets`. Both datasets are already uploaded:

- `s3://neu-datasets/HI-Small_Trans.csv`
- `s3://neu-datasets/SAML-D.csv` (outside test set, not used by this notebook yet)

To upload a new file: `aws s3 cp <file> s3://neu-datasets/`.

## Run on SageMaker

1. **Create a notebook instance.** Use `ml.m5.2xlarge` with a 100 GB volume. Keep internet access on, so `pip` and `git` work.
2. **Attach an execution role** that can read and write `neu-datasets`. It needs `s3:GetObject`, `s3:PutObject`, and `s3:ListBucket` on that bucket.
3. **Open a terminal** on the instance and clone this repo. The repo root is this `notebooks/` folder:
   ```bash
   git clone <your notebooks repo url> triage-notebooks
   cd triage-notebooks
   ```
4. **Open `train_triage_xgboost.ipynb`** and check the kernel is Python 3.12, the same version the backend venv uses. Pandas 3 needs a recent Python.
5. **Run all cells.** The full HI-Small run takes roughly 20 to 40 minutes for the feature build, plus training. The feature cache means an interrupted run resumes.
6. **Stop the instance** when finished. Notebook instances bill while running.

Outputs: `models/triage_hi_small/model.json` and `eval/triage_hi_small/metrics.json`. If `UPLOAD_RESULTS` is on, both are also copied to `s3://neu-datasets/triage/triage_hi_small/`.

## Settings to check

Set in the configuration cell:

- `HORIZON = 2`: label horizon in days. HI-Small is short, so 7 would leave no test days.
- `BUDGET = 50`: flagged account-days per day. **Placeholder** until the alert budget is confirmed.
- `RUN_NAME`: change it for each run so outputs don't overwrite each other.
- `S3_KEY_HI_SMALL`: change it to `HI-Large_Trans.csv` for the main model.

## Scope of the HI-Small results

HI-Small is a pipeline check. Its activity falls mostly on days 1 to 10, so the test block is about four days. Don't use these numbers as the model's result. The real evaluation runs on HI-Large.

## Tests

From this folder, with the backend venv:

```bash
/path/to/backend/mainvenv/bin/python -m pytest -q tests
```

## Keeping the copy in sync

`triage_lib/` is a copy of `backend/app/ml/triage_features.py` and `triage_train.py`, plus the `alerts.py` and `graph_rules.py` they depend on. This notebook is the source of truth for training. Changes to the feature logic should be made here first.
