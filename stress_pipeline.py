"""Run chunked inference with the model saved by main.ipynb."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def run_inference(input_path: Path, model_path: Path, output_dir: Path, chunksize: int) -> dict:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    started = time.perf_counter()
    input_rows = 0
    prediction_rows = 0
    first_chunk = True

    for chunk in pd.read_csv(input_path, chunksize=chunksize, dtype={"id": "string"}, low_memory=False):
        missing = [feature for feature in feature_names if feature not in chunk.columns]
        if missing:
            raise ValueError(f"Input is missing model features: {missing}")
        values = chunk[feature_names].replace([np.inf, -np.inf], np.nan)
        predicted = model.predict(values)
        start_row = input_rows
        input_rows += len(chunk)
        result = pd.DataFrame(
            {"input_row": np.arange(start_row, input_rows), "predicted_stress": predicted}
        )
        if hasattr(model, "predict_proba"):
            result["prediction_confidence"] = model.predict_proba(values).max(axis=1)
        if "label" in chunk.columns:
            result["true_label"] = chunk["label"].to_numpy()
        result.to_csv(
            predictions_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )
        first_chunk = False
        prediction_rows += len(result)

    elapsed = time.perf_counter() - started
    summary = {
        "status": "success",
        "input_rows": input_rows,
        "predictions": prediction_rows,
        "processing_seconds": elapsed,
        "throughput_rows_per_second": prediction_rows / elapsed if elapsed else None,
        "model": str(model_path),
        "warnings": [],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunksize", type=int, default=100_000)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(
        json.dumps(
            run_inference(
                Path(arguments.input),
                Path(arguments.model),
                Path(arguments.output_dir),
                arguments.chunksize,
            ),
            indent=2,
        )
    )
