from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from m2_2 import Config, load_and_validate, parse_args


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(tmp_path: Path, csv_text: str) -> Config:
    """Write *csv_text* to a temp file and return a Config pointing at it."""
    p = tmp_path / "data.csv"
    p.write_text(textwrap.dedent(csv_text))
    return Config(
        input_path=p,
        test_size=0.25,
        random_state=42,
        model_out=tmp_path / "model.joblib",
        verbose=False,
    )


# ---------------------------------------------------------------------------
# load_and_validate tests
# ---------------------------------------------------------------------------

class TestLoadAndValidate:
    def test_valid_csv(self, tmp_path: Path) -> None:
        csv = """\
            SizeSqFt,Bedrooms,AgeYears,Price
            800,1,20,50000
            900,2,15,60000
            1100,2,10,75000
            1300,3,8,90000
        """
        cfg = _make_cfg(tmp_path, csv)
        df = load_and_validate(cfg)
        assert len(df) == 4

    def test_missing_column_raises(self, tmp_path: Path) -> None:
        csv = """\
            SizeSqFt,Bedrooms,Price
            800,1,50000
            900,2,60000
            1100,2,75000
            1300,3,90000
        """
        cfg = _make_cfg(tmp_path, csv)
        with pytest.raises(ValueError, match="Missing required columns"):
            load_and_validate(cfg)

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        cfg = Config(
            input_path=tmp_path / "no_such_file.csv",
            test_size=0.25,
            random_state=42,
            model_out=tmp_path / "model.joblib",
            verbose=False,
        )
        with pytest.raises(FileNotFoundError):
            load_and_validate(cfg)

    def test_non_numeric_rows_dropped(self, tmp_path: Path) -> None:
        csv = """\
            SizeSqFt,Bedrooms,AgeYears,Price
            800,1,20,50000
            900,two,15,60000
            1100,2,10,75000
            1300,3,8,90000
            1500,3,5,110000
        """
        cfg = _make_cfg(tmp_path, csv)
        df = load_and_validate(cfg)
        assert len(df) == 4  # one row dropped

    def test_negative_features_dropped(self, tmp_path: Path) -> None:
        csv = """\
            SizeSqFt,Bedrooms,AgeYears,Price
            800,1,20,50000
            -100,2,15,60000
            1100,2,10,75000
            1300,3,8,90000
            1500,3,5,110000
        """
        cfg = _make_cfg(tmp_path, csv)
        df = load_and_validate(cfg)
        assert len(df) == 4  # negative-SizeSqFt row dropped

    def test_non_positive_price_dropped(self, tmp_path: Path) -> None:
        csv = """\
            SizeSqFt,Bedrooms,AgeYears,Price
            800,1,20,50000
            900,2,15,0
            1100,2,10,75000
            1300,3,8,90000
            1500,3,5,110000
        """
        cfg = _make_cfg(tmp_path, csv)
        df = load_and_validate(cfg)
        assert len(df) == 4  # zero-price row dropped

    def test_too_few_rows_raises(self, tmp_path: Path) -> None:
        csv = """\
            SizeSqFt,Bedrooms,AgeYears,Price
            800,1,20,50000
            900,2,15,60000
        """
        cfg = _make_cfg(tmp_path, csv)
        with pytest.raises(ValueError, match="Not enough valid rows"):
            load_and_validate(cfg)


# ---------------------------------------------------------------------------
# parse_args tests
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_defaults(self, tmp_path: Path) -> None:
        cfg = parse_args(["--input", str(tmp_path / "data.csv")])
        assert cfg.test_size == 0.25
        assert cfg.random_state == 42

    def test_test_size_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="--test-size must be between"):
            parse_args(["--input", "x.csv", "--test-size", "0.9"])
