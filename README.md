# Paying Hours -- Project Billing Estimator

Train a **Linear Regression** model from employee time-tracking data (Excel) and hourly rates, then predict billing or export rate data as JSON for a client-facing cost-estimator website.

## Data format

Your Excel file (`.xlsx`) should have these columns in order:

| Column | Description |
|--------|-------------|
| `#` | Row ID (number) |
| `Date` | Date and time of entry |
| `Employee` | Employee name |
| `Client` | Client company name |
| `Project` | Project or task name |
| `Start Time` | Work start time |
| `End Time` | Work end time |
| `Total Hours` | Decimal hours worked |

## Configuration

### Employee rates (`rates.json`)

Create a `rates.json` file with hourly rates per employee:

```json
{
  "rates": {
    "חן": 150,
    "אושרת": 120
  },
  "currency": "₪",
  "default_rate": 130
}
```

- `rates` -- hourly rate per employee name
- `currency` -- currency symbol for display
- `default_rate` -- fallback rate for unlisted employees

## Setup

```bash
pip install -r requirements.txt
```

## Commands

### 1. Prepare (validate data)

```bash
python paying_hours.py prepare "SAMPLE PROJECT.xlsx" --out prepared.csv
```

### 2. Train the model

```bash
python paying_hours.py train "SAMPLE PROJECT.xlsx" --rates rates.json --model billing_model.joblib
```

### 3. Predict billing

```bash
python paying_hours.py predict "SAMPLE PROJECT.xlsx" --model billing_model.joblib --rates rates.json --out predictions.csv
```

### 4. Export data for website

```bash
python paying_hours.py export "SAMPLE PROJECT.xlsx" --rates rates.json --out model_data.json
```

## Client website

After running `export`, open the website:

```bash
python -m http.server 8000
```

Then visit `http://localhost:8000` in a browser. The website lets clients select a project type, enter estimated hours, and see an approximate cost.

> **Note:** The website uses `fetch()` to load `model_data.json`, so it must be served over HTTP -- opening `index.html` directly via `file://` will not work.

## How it works

- **Billing** is computed as `employee_hourly_rate * total_hours` using rates from `rates.json`
- **Model features**: `Employee`, `Client`, `Project` (one-hot encoded) + `Total Hours` (numeric)
- **Model**: sklearn `LinearRegression` pipeline, saved with `joblib`
- **Website**: Static HTML/CSS/JS that reads `model_data.json` and computes `avg_rate * hours` per project type
