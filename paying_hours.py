"""
Paying Hours -- Project Billing Estimator

Train a Linear Regression model from employee time-tracking data (Excel)
and employee hourly rates, then predict billing or export rate data
as JSON for the client-facing cost-estimator website.

Usage:
  python paying_hours.py train  <excel>  [--rates rates.json] [--model billing_model.joblib]
  python paying_hours.py predict <excel> --model billing_model.joblib [--rates rates.json] [--out predictions.csv]
  python paying_hours.py export  <excel>  [--rates rates.json] [--out model_data.json]
  python paying_hours.py prepare <excel>  [--out prepared.csv]
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ---------- column names (after renaming) ----------
COL_ID = "id"
COL_DATE = "Date"
COL_EMPLOYEE = "Employee"
COL_CLIENT = "Client"
COL_PROJECT = "Project"
COL_START = "Start Time"
COL_END = "End Time"
COL_HOURS = "Total Hours"
COL_BILLING = "billing_amount"

EXPECTED_COLUMNS = [COL_ID, COL_DATE, COL_EMPLOYEE, COL_CLIENT,
                    COL_PROJECT, COL_START, COL_END, COL_HOURS]
FEATURE_COLS = [COL_EMPLOYEE, COL_CLIENT, COL_PROJECT, COL_HOURS]
CAT_COLS = [COL_EMPLOYEE, COL_CLIENT, COL_PROJECT]
NUM_COLS = [COL_HOURS]


# =====================================================================
# Loading helpers
# =====================================================================

def load_rates(rates_path: str) -> tuple[dict, float, str]:
    """Return (rate_map, default_rate, currency) from a JSON config."""
    path = Path(rates_path)
    if not path.exists():
        raise FileNotFoundError(f"Rates file not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    rates = cfg.get("rates", {})
    default = cfg.get("default_rate", 0)
    currency = cfg.get("currency", "₪")
    return rates, default, currency


def _normalize_date(val):
    """Convert mixed date values to datetime."""
    if isinstance(val, datetime):
        return val
    if pd.isna(val):
        return pd.NaT
    s = str(val).strip()
    for fmt in ("%d.%m.%Y, %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d.%m.%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return pd.NaT


def load_excel(excel_path: str) -> pd.DataFrame:
    """Read the time-tracking Excel file and return a cleaned DataFrame."""
    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_excel(path, engine="openpyxl")

    # Rename columns by position (first column header may be a number)
    if len(df.columns) < 8:
        raise ValueError(
            f"Expected at least 8 columns, got {len(df.columns)}. "
            f"Columns found: {list(df.columns)}"
        )
    df.columns = EXPECTED_COLUMNS + list(df.columns[8:])

    # Normalize dates
    df[COL_DATE] = df[COL_DATE].apply(_normalize_date)

    # Ensure hours are numeric
    df[COL_HOURS] = pd.to_numeric(df[COL_HOURS], errors="coerce")
    if df[COL_HOURS].isna().any():
        raise ValueError("All 'Total Hours' values must be numeric.")

    # Strip whitespace from string columns
    for col in CAT_COLS:
        df[col] = df[col].astype(str).str.strip()

    return df


# =====================================================================
# Billing computation
# =====================================================================

def compute_billing(df: pd.DataFrame, rates: dict, default_rate: float) -> pd.DataFrame:
    """Add a billing_amount column = employee rate * hours."""
    df = df.copy()
    df[COL_BILLING] = df[COL_EMPLOYEE].map(rates).fillna(default_rate) * df[COL_HOURS]
    df[COL_BILLING] = df[COL_BILLING].round(2)
    return df


# =====================================================================
# Model
# =====================================================================

def build_pipeline() -> Pipeline:
    """Build sklearn pipeline: encode categoricals + linear regression."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
            ("num", "passthrough", NUM_COLS),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])


# =====================================================================
# Commands
# =====================================================================

def train(data_path: str, rates_path: str, model_path: str) -> None:
    """Train model from Excel data + rates and save pipeline."""
    df = load_excel(data_path)
    rates, default_rate, _ = load_rates(rates_path)
    df = compute_billing(df, rates, default_rate)

    train_df = df.dropna(subset=[COL_BILLING])
    if train_df.empty:
        raise ValueError("No rows with valid billing amounts for training.")

    X = train_df[FEATURE_COLS]
    y = train_df[COL_BILLING]

    pipeline = build_pipeline()
    pipeline.fit(X, y)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Model trained on {len(train_df)} rows and saved to: {model_path}")

    score = pipeline.score(X, y)
    print(f"R² on training data: {score:.4f}")


def predict(data_path: str, rates_path: str, model_path: str, out_path: str | None) -> pd.DataFrame:
    """Load model, predict billing for Excel rows."""
    df = load_excel(data_path)
    rates, default_rate, _ = load_rates(rates_path)
    df = compute_billing(df, rates, default_rate)

    pipeline = joblib.load(model_path)
    X = df[FEATURE_COLS]
    pred = pipeline.predict(X)
    pred = np.maximum(pred, 0)

    result = df.copy()
    result["predicted_billing"] = np.round(pred, 2)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
        print(f"Predictions written to: {out_path}")

    return result


def export_json(data_path: str, rates_path: str, out_path: str) -> None:
    """Aggregate rate data by project and export as JSON for the website."""
    df = load_excel(data_path)
    rates, default_rate, currency = load_rates(rates_path)
    df = compute_billing(df, rates, default_rate)

    # Per-hour rate for each row
    df["rate_per_hour"] = df[COL_BILLING] / df[COL_HOURS].replace(0, np.nan)

    projects = {}
    for project, group in df.groupby(COL_PROJECT):
        projects[project] = {
            "avg_rate_per_hour": round(group["rate_per_hour"].mean(), 2),
            "total_hours_logged": round(group[COL_HOURS].sum(), 4),
            "entry_count": int(len(group)),
            "clients": sorted(group[COL_CLIENT].unique().tolist()),
        }

    total_billing = df[COL_BILLING].sum()
    total_hours = df[COL_HOURS].sum()
    overall_avg = round(total_billing / total_hours, 2) if total_hours > 0 else 0

    data = {
        "generated_at": datetime.now().isoformat(),
        "currency": currency,
        "projects": projects,
        "project_list": sorted(projects.keys()),
        "overall_avg_rate": overall_avg,
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Model data exported to: {out_path}")


def prepare(data_path: str, out_path: str | None) -> pd.DataFrame:
    """Validate Excel and optionally write cleaned CSV."""
    df = load_excel(data_path)
    print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
    if out_path:
        df.to_csv(out_path, index=False)
        print(f"Prepared data written to: {out_path}")
    return df


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train or run a billing model from employee time-tracking data."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Train model from Excel + rates")
    p_train.add_argument("data", help="Path to Excel file (.xlsx)")
    p_train.add_argument("--rates", default="rates.json", help="Path to rates JSON config")
    p_train.add_argument("--model", default="billing_model.joblib", help="Output model path")

    # predict
    p_predict = sub.add_parser("predict", help="Predict billing for rows")
    p_predict.add_argument("data", help="Path to Excel file (.xlsx)")
    p_predict.add_argument("--model", default="billing_model.joblib", required=True, help="Saved model path")
    p_predict.add_argument("--rates", default="rates.json", help="Path to rates JSON config")
    p_predict.add_argument("--out", help="Output CSV path for predictions")

    # export
    p_export = sub.add_parser("export", help="Export rate data as JSON for the website")
    p_export.add_argument("data", help="Path to Excel file (.xlsx)")
    p_export.add_argument("--rates", default="rates.json", help="Path to rates JSON config")
    p_export.add_argument("--out", default="model_data.json", help="Output JSON path")

    # prepare
    p_prepare = sub.add_parser("prepare", help="Validate and normalize Excel data")
    p_prepare.add_argument("data", help="Path to Excel file (.xlsx)")
    p_prepare.add_argument("--out", help="Output path for normalized CSV")

    args = parser.parse_args()

    try:
        if args.command == "train":
            train(args.data, args.rates, args.model)
        elif args.command == "predict":
            predict(args.data, args.rates, args.model, getattr(args, "out", None))
        elif args.command == "export":
            export_json(args.data, args.rates, args.out)
        elif args.command == "prepare":
            prepare(args.data, getattr(args, "out", None))
    except (FileNotFoundError, ValueError) as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
