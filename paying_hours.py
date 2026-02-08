"""
Paying Hours – Linear Regression billing model

Train a model from (employee, project, hours) -> billing_amount,
then predict how much to bill clients for new work.

Usage:
  python paying_hours.py train <data.csv> [--model model.joblib]
  python paying_hours.py predict <data.csv> --model model.joblib [--out predictions.csv]
  python paying_hours.py prepare <data.csv> [--out prepared.csv]   # just validate & normalize columns
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Expected column names (adjust if your CSV uses different names)
COL_EMPLOYEE = "employee"
COL_PROJECT = "project"
COL_HOURS = "hours"
COL_BILLING = "billing_amount"

REQUIRED_FEATURE_COLS = [COL_EMPLOYEE, COL_PROJECT, COL_HOURS]
TARGET_COL = COL_BILLING


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to strip whitespace and match expected names."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def load_and_validate(
    csv_path: str,
    *,
    require_billing: bool = False,
) -> pd.DataFrame:
    """Load CSV and ensure required columns exist."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    df = _normalize_columns(df)

    missing = [c for c in REQUIRED_FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Expected columns: {REQUIRED_FEATURE_COLS} and optionally '{TARGET_COL}'."
        )

    if COL_HOURS in df.columns:
        df[COL_HOURS] = pd.to_numeric(df[COL_HOURS], errors="coerce")
    if df[COL_HOURS].isna().any():
        raise ValueError("All 'hours' must be numeric.")

    if require_billing and TARGET_COL not in df.columns:
        raise ValueError(
            f"For training, CSV must include a '{TARGET_COL}' column with known billing amounts."
        )
    if TARGET_COL in df.columns:
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    return df


def build_pipeline():
    """Build sklearn pipeline: encode categoricals + linear regression."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                [COL_EMPLOYEE, COL_PROJECT],
            ),
            ("num", "passthrough", [COL_HOURS]),
        ],
        remainder="drop",
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )
    return pipeline


def train(data_path: str, model_path: str) -> None:
    """Train model from CSV (must include billing_amount) and save pipeline."""
    df = load_and_validate(data_path, require_billing=True)
    # Drop rows where billing is missing (e.g. placeholder)
    train_df = df.dropna(subset=[TARGET_COL])
    if train_df.empty:
        raise ValueError(
            f"No rows with valid '{TARGET_COL}'. Add billing amounts for training."
        )

    X = train_df[REQUIRED_FEATURE_COLS]
    y = train_df[TARGET_COL]

    pipeline = build_pipeline()
    pipeline.fit(X, y)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Model trained on {len(train_df)} rows and saved to: {model_path}")

    # Optional: show simple score
    score = pipeline.score(X, y)
    print(f"R² on training data: {score:.4f}")


def predict(data_path: str, model_path: str, out_path: str | None) -> pd.DataFrame:
    """Load model, predict billing for CSV rows, optionally write results."""
    df = load_and_validate(data_path, require_billing=False)
    pipeline = joblib.load(model_path)

    X = df[REQUIRED_FEATURE_COLS]
    pred = pipeline.predict(X)
    pred = np.maximum(pred, 0)  # no negative billing

    result = df.copy()
    result["predicted_billing"] = np.round(pred, 2)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
        print(f"Predictions written to: {out_path}")

    return result


def prepare(data_path: str, out_path: str | None) -> pd.DataFrame:
    """Validate CSV and normalize columns; optionally write prepared CSV."""
    df = load_and_validate(data_path, require_billing=False)
    if out_path:
        df.to_csv(out_path, index=False)
        print(f"Prepared data written to: {out_path}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Train or run a Linear Regression billing model from paying hours data."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Train model from CSV with billing_amount")
    p_train.add_argument("data", help="Path to CSV (employee, project, hours, billing_amount)")
    p_train.add_argument("--model", default="billing_model.joblib", help="Output model path")

    # predict
    p_predict = sub.add_parser("predict", help="Predict billing for rows in CSV")
    p_predict.add_argument("data", help="Path to CSV (employee, project, hours)")
    p_predict.add_argument("--model", default="billing_model.joblib", required=True, help="Saved model path")
    p_predict.add_argument("--out", help="Output CSV path for predictions")

    # prepare
    p_prepare = sub.add_parser("prepare", help="Validate and normalize CSV columns")
    p_prepare.add_argument("data", help="Path to CSV")
    p_prepare.add_argument("--out", help="Output path for normalized CSV")

    args = parser.parse_args()

    try:
        if args.command == "train":
            train(args.data, args.model)
        elif args.command == "predict":
            predict(args.data, args.model, getattr(args, "out", None))
        elif args.command == "prepare":
            prepare(args.data, getattr(args, "out", None))
    except (FileNotFoundError, ValueError) as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
