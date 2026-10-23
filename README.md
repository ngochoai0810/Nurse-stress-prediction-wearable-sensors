# Nurse-stress-prediction-wearable-sensors

This repository contains code and a fast Random Forest baseline for predicting
stress levels in nurses from wearable-sensor data. This is a research baseline,
not a medical diagnostic tool.

## Project description

The project investigates physiological signals collected by wearable devices
from nurses working in a hospital. The signals include heart rate, skin
temperature, electrodermal activity (EDA), and accelerometer measurements.

The goal is to evaluate whether these signals can be used to predict the
participant-reported stress level. The implementation deliberately remains a
small baseline: it does not include deep learning, dashboards, real-time
streaming, large-scale hyperparameter tuning, or Dryad raw-data processing.

## Data sources and schema

The original project data is available from the following Google Drive folder:

<https://drive.google.com/drive/folders/10s_mez9cQ0J0J7hUnkNCVE3BVTNT8jU?usp=sharing>

For the fast baseline, the merged and labeled CSV was downloaded from the
official Kaggle dataset using the Kaggle CLI:

<https://www.kaggle.com/datasets/priyankraval/nurse-stress-prediction-wearable-sensors>

The downloaded file is `merged_data.csv`. It has the following schema:

```text
X, Y, Z, EDA, HR, TEMP, id, datetime, label
```

The local verification run found:

- 11,509,051 total rows;
- 15 unique subjects;
- 2,162,246 rows with label `0.0`;
- 806,222 rows with label `1.0`;
- 8,540,583 rows with label `2.0`;
- no missing values in the nine expected columns.

The large dataset is intentionally not committed to Git. Place it at
`data\merged_data.csv` when reproducing the run.

## Repository contents

- `main.ipynb`: original preprocessing and analysis notebook, including the
  updated fast-baseline training and evaluation cell.
- `Pre-Processing Python Files\`: original scripts for combining, merging, and
  labeling raw sensor data.
- `stress_pipeline.py`: chunked inference CLI using the saved model.
- `requirements.txt`: Python dependencies.
- `models\`: generated model artifacts, ignored by Git.
- `verification\`: generated metrics, plots, predictions, and summaries, ignored
  by Git.

## Requirements

Python 3.7 or higher. Install the dependencies with:

```text
python -m pip install -r requirements.txt
```

## Fast baseline implementation

The Kaggle merged/labeled CSV is loaded into `merged_df` in `main.ipynb`.
The Random Forest cell at the end of the notebook:

- uses only `X`, `Y`, `Z`, `TEMP`, `HR`, and `EDA`;
- excludes `id`, `datetime`, and `label` from model features;
- replaces positive and negative infinity with missing values;
- removes rows without a target or without any usable sensor feature;
- uses a subject-group split on `id` to prevent subject leakage;
- uses `random_state=42`;
- fits median imputation on training data through the saved pipeline;
- trains `RandomForestClassifier` with 100 trees, maximum depth 12,
  `class_weight="balanced"`, and `n_jobs=-1`;
- writes `models\stress_model.joblib`,
  `verification\kaggle\metrics.json`, and
  `verification\kaggle\confusion_matrix.png`.

For the verified run, the full dataset was split by subject first using
`GroupShuffleSplit(test_size=0.2, random_state=42)`. Twelve subjects were used
for training and three subjects were held out for testing. To keep the baseline
fast, a stratified maximum of 200,000 rows was used for training and 200,000
rows for testing.

Run the notebook after pointing the existing data-loading cells at
`data\merged_data.csv`:

```text
jupyter notebook main.ipynb
```

## Inference

Inference runs without retraining:

```text
python stress_pipeline.py data\merged_data.csv --model models\stress_model.joblib --output-dir verification\inference --chunksize 250000
```

The CLI reads input in chunks, validates the saved feature schema, preserves
the input row number, and writes predicted label, confidence, and true label
when the input contains `label`.

## Measured results

The following results were measured on the real Kaggle `merged_data.csv`, not
generated or copied from the original exploratory notebook:

| Metric | Measured value |
|---|---:|
| Training rows | 200,000 |
| Test rows | 200,000 |
| Training time | 18.97 seconds |
| Test inference time | 1.22 seconds |
| Macro F1 | 0.22146262134591366 |
| Balanced Accuracy | 0.2837993557976954 |

Full-data inference was also completed:

- Input rows: 11,509,051
- Predictions: 11,509,051
- Processing time: 208.20 seconds
- Throughput: approximately 55,280 rows/second

The test subjects contained no examples of class `1.0`. Consequently, the
class-`1.0` recall and F1 are zero in this split, and the macro metrics are not
fully representative of all classes. This limitation is recorded in
`verification\kaggle\metrics.json`.

Generated artifacts:

- `models\stress_model.joblib`
- `verification\kaggle\metrics.json`
- `verification\kaggle\confusion_matrix.png`
- `verification\inference\predictions.csv`
- `verification\inference\summary.json`

The generated model and verification outputs are ignored by Git because they
are large local artifacts. They remain available in the working directory after
running the baseline.

## Limitations and recommendations

- The test split is subject-disjoint, but the held-out subjects do not cover
  all three classes.
- The label distribution is highly imbalanced, especially for class `2.0`.
- Macro metrics should be interpreted together with per-class precision, recall,
  support, and the confusion matrix.
- Further work could evaluate repeated subject splits, class-aware sampling,
  signal-specific models, and additional physiological features.
- The model must not be used as a medical diagnostic system.

## License

This project is licensed under the MIT License; see [LICENSE](LICENSE) for
details.
