from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LOG = logging.getLogger("house_price_model")

FEATURES = ["SizeSqFt", "Bedrooms", "AgeYears"]
TARGET = "Price"


@dataclass(frozen=True)
class Config:
    input_path: Path
    test_size: float
    random_state: int
    model_out: Path
    verbose: bool


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def parse_args(argv: list[str]) -> Config:
    p = argparse.ArgumentParser(
        description="Train a simple regression model on house_prices.csv"
    )
    p.add_argument("--input", required=True, help="Path to input CSV")
    p.add_argument("--test-size", type=float, default=0.25, help="Test split ratio")
    p.add_argument("--random-state", type=int, default=42, help="Random seed")
    p.add_argument("--model-out", default="model.joblib", help="Output model file")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = p.parse_args(argv)
    configure_logging(args.verbose)

    if not (0.1 <= args.test_size <= 0.5):
        raise ValueError("--test-size must be between 0.1 and 0.5")

    return Config(
        input_path=Path(args.input),
        test_size=args.test_size,
        random_state=args.random_state,
        model_out=Path(args.model_out),
        verbose=args.verbose,
    )


def load_and_validate(cfg: Config) -> pd.DataFrame:
    if not cfg.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {cfg.input_path}")

    df = pd.read_csv(cfg.input_path)

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Convert to numeric and drop invalid rows
    for c in FEATURES + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    before = len(df)
    df = df.dropna()
    after = len(df)

    if after != before:
        LOG.warning("Dropped %d invalid rows (non-numeric or missing)", before - after)

    # Reject rows with negative feature values or non-positive prices
    valid_mask = (df[FEATURES] >= 0).all(axis=1) & (df[TARGET] > 0)
    n_invalid = (~valid_mask).sum()
    if n_invalid:
        LOG.warning("Dropped %d rows with negative features or non-positive price", n_invalid)
        df = df[valid_mask].reset_index(drop=True)

    if len(df) < 4:
        raise ValueError("Not enough valid rows to train a model.")

    return df


def train_and_evaluate(df: pd.DataFrame, cfg: Config) -> int:
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        shuffle=True,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )

    LOG.info("Training model...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    LOG.info("Metrics:")
    LOG.info("  MAE  = %.2f", mae)
    LOG.info("  RMSE = %.2f", rmse)
    LOG.info("  R2   = %.4f", r2)

    cfg.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, cfg.model_out)
    LOG.info("Saved model to %s", cfg.model_out)

    return 0


def main(argv: list[str]) -> int:
    try:
        cfg = parse_args(argv)
        df = load_and_validate(cfg)
        return train_and_evaluate(df, cfg)
    except Exception as ex:
        LOG.error("Failed: %s", ex, exc_info=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
