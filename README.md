# House Price Regression Model

Trains a linear regression model on housing data using scikit-learn.

**Features:** SizeSqFt, Bedrooms, AgeYears
**Target:** Price

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

## Usage

```bash
python m2_2.py --input house_prices.csv
```

### Options

| Flag              | Default          | Description                      |
|-------------------|------------------|----------------------------------|
| `--input`         | *(required)*     | Path to input CSV                |
| `--test-size`     | `0.25`           | Test split ratio (0.1 – 0.5)    |
| `--random-state`  | `42`             | Random seed for reproducibility  |
| `--model-out`     | `model.joblib`   | Output path for trained model    |
| `--verbose`       | off              | Enable debug logging             |

## Running Tests

```bash
pip install pytest
pytest tests/
```
