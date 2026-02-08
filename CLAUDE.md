# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Billing estimation tool that trains a Linear Regression model from employee time-tracking Excel data and hourly rates, then predicts project costs. Includes a static Hebrew RTL website for client-facing cost estimation.

## Commands

```bash
# Setup
pip install -r requirements.txt

# Validate and normalize Excel data
python paying_hours.py prepare "SAMPLE PROJECT.xlsx" --out prepared.csv

# Train the ML model
python paying_hours.py train "SAMPLE PROJECT.xlsx" --rates rates.json --model billing_model.joblib

# Predict billing amounts
python paying_hours.py predict "SAMPLE PROJECT.xlsx" --model billing_model.joblib --rates rates.json --out predictions.csv

# Export aggregated rate data as JSON for the website
python paying_hours.py export "SAMPLE PROJECT.xlsx" --rates rates.json --out model_data.json

# Serve the website (required — fetch() won't work over file://)
python -m http.server 8000
```

No test suite exists. Verify changes by running the four subcommands above against `SAMPLE PROJECT.xlsx`.

## Architecture

**Single-file backend** (`paying_hours.py`) with four CLI subcommands dispatched via argparse:

1. `load_excel()` reads `.xlsx` via openpyxl, renames columns by position (column A header is a number, not `#`), normalizes mixed date formats, coerces hours to numeric
2. `load_rates()` reads `rates.json` — maps employee names (Hebrew) to hourly rates
3. `compute_billing()` multiplies `employee_rate × Total Hours` to produce `billing_amount`
4. `build_pipeline()` creates an sklearn Pipeline: `ColumnTransformer(OneHotEncoder for [Employee, Client, Project] + passthrough for [Total Hours]) → LinearRegression`
5. `export_json()` aggregates by project, computes `avg_rate_per_hour`, and writes `model_data.json`

**Data flow:** Excel input → `load_excel` → `compute_billing` (with rates) → train/predict/export

**Website** (`index.html` + `style.css`): vanilla JS fetches `model_data.json`, populates a project dropdown, calculates `avg_rate × hours` on form submit. All user-facing text is Hebrew with `dir="rtl"`.

## Key Details

- Excel data has inconsistent date formats: native datetime objects and strings like `'4.2.2026, 13:32:36'`. The `_normalize_date()` function handles both.
- Column renaming is positional (not by header name) because the first column header in the Excel file is the number `4.0`.
- Hebrew content requires `ensure_ascii=False` when writing JSON and `encoding='utf-8'` for file I/O.
- Generated artifacts (`billing_model.joblib`, `predictions.csv`, `model_data.json`, `prepared.csv`) are outputs — regenerate with the CLI commands.
- `rates.json` has a `default_rate` fallback for employees not listed in the rates map.
