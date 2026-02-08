# Paying Hours – Billing with Linear Regression

Use a **Linear Regression** model to turn your employee paying-hours data (per project) into **client billing amounts**. The model learns from your historical data how hours and project type relate to what you bill, then predicts billing for new work.

## Data format

Your CSV should have these columns (names must match exactly):

| Column          | Description                    |
|-----------------|--------------------------------|
| `employee`      | Employee name or ID            |
| `project`       | Project/client name            |
| `hours`         | Hours worked (number)          |
| `billing_amount`| **For training only**: amount billed (number). Omit for rows you want to predict. |

- **Training**: Include `billing_amount` for past work so the model can learn.
- **Prediction**: Use the same columns but leave `billing_amount` empty or omit it for new work.

Example: `sample_hours_data.csv` in this folder.

## Setup

```bash
pip install -r requirements.txt
```

## Commands

### 1. Prepare your data (optional)

Check that your CSV has the right columns and normalized headers:

```bash
python paying_hours.py prepare your_data.csv --out prepared.csv
```

### 2. Train the model

Train on a CSV that includes `billing_amount`:

```bash
python paying_hours.py train your_data.csv --model billing_model.joblib
```

Using the sample file:

```bash
python paying_hours.py train sample_hours_data.csv --model billing_model.joblib
```

### 3. Predict billing for new work

Use the saved model to predict billing for another CSV (same columns, no need for `billing_amount`):

```bash
python paying_hours.py predict new_work.csv --model billing_model.joblib --out predictions.csv
```

The output CSV will contain your original columns plus `predicted_billing`.

## How the model works

- **Features**: `employee`, `project`, and `hours` (project and employee are one-hot encoded so the model can learn different effective rates per project and per person).
- **Target**: `billing_amount`.
- **Model**: `sklearn` Linear Regression; the pipeline is saved with `joblib` so you can reload it without retraining.

You can replace `your_data.csv` with your real paying-hours table and add a `billing_amount` column (from past invoices) to train a model that fits your actual billing.
