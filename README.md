# House Price Regression Model

[![CI](https://github.com/debrej2021/testing-2-python/actions/workflows/ci.yml/badge.svg)](https://github.com/debrej2021/testing-2-python/actions/workflows/ci.yml)

A command-line tool that trains a linear regression model to predict house prices from housing features using scikit-learn.

## Features

- Loads and validates CSV data with automatic cleaning (drops non-numeric, negative, and missing values)
- Trains a `StandardScaler` + `LinearRegression` pipeline
- Reports MAE, RMSE, and R-squared metrics on a held-out test set
- Saves the trained model as a `.joblib` file for later use

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/debrej2021/testing-2-python.git
cd testing-2-python

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic

```bash
python m2_2.py --input house_prices.csv
```

### With custom options

```bash
python m2_2.py --input house_prices.csv --test-size 0.3 --random-state 99 --model-out models/house.joblib --verbose
```

### Options

| Flag             | Default        | Description                            |
| ---------------- | -------------- | -------------------------------------- |
| `--input`        | *(required)*   | Path to input CSV file                 |
| `--test-size`    | `0.25`         | Train/test split ratio (0.1 -- 0.5)   |
| `--random-state` | `42`           | Random seed for reproducibility        |
| `--model-out`    | `model.joblib` | Output path for the trained model file |
| `--verbose`      | off            | Enable debug-level logging             |

### Input CSV format

The CSV must contain these columns:

| Column     | Type  | Description              |
| ---------- | ----- | ------------------------ |
| `SizeSqFt` | float | Property size in sq. ft. |
| `Bedrooms` | int   | Number of bedrooms       |
| `AgeYears` | float | Age of property in years |
| `Price`    | float | Sale price (target)      |

### Example output

```
2026-02-14 10:00:00 INFO house_price_model - Training on 6 samples, evaluating on 2 samples
2026-02-14 10:00:00 INFO house_price_model - Metrics:
2026-02-14 10:00:00 INFO house_price_model -   MAE  = 3346.97
2026-02-14 10:00:00 INFO house_price_model -   RMSE = 3800.10
2026-02-14 10:00:00 INFO house_price_model -   R2   = 0.9882
2026-02-14 10:00:00 INFO house_price_model - Saved model to model.joblib
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage (requires pytest-cov)
pytest tests/ -v --cov=m2_2
```

## Project Structure

```
testing-2-python/
├── .github/workflows/ci.yml   # GitHub Actions CI pipeline
├── m2_2.py                    # Main application
├── house_prices.csv           # Sample training data
├── requirements.txt           # Python dependencies
├── tests/
│   ├── __init__.py
│   ├── test_validation.py     # Data loading & argument parsing tests
│   └── test_training.py       # Training, evaluation & integration tests
└── README.md
```

## License

This project is provided for educational purposes.
