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


VALID_CSV = """\
    SizeSqFt,Bedrooms,AgeYears,Price
    800,1,20,50000
    900,2,15,60000
    1100,2,10,75000
    1300,3,8,90000
    1500,3,5,110000
    1800,4,3,140000
    2000,4,2,160000
    2200,5,1,180000
"""


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

    def test_negative_price_dropped(self, tmp_path: Path) -> None:
        csv = """\
            SizeSqFt,Bedrooms,AgeYears,Price
            800,1,20,50000
            900,2,15,-5000
            1100,2,10,75000
            1300,3,8,90000
            1500,3,5,110000
        """
        cfg = _make_cfg(tmp_path, csv)
        df = load_and_validate(cfg)
        assert len(df) == 4

    def test_too_few_rows_raises(self, tmp_path: Path) -> None:
        csv = """\
            SizeSqFt,Bedrooms,AgeYears,Price
            800,1,20,50000
            900,2,15,60000
        """
        cfg = _make_cfg(tmp_path, csv)
        with pytest.raises(ValueError, match="Not enough valid rows"):
            load_and_validate(cfg)

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.csv"
        p.write_text("")
        cfg = Config(
            input_path=p,
            test_size=0.25,
            random_state=42,
            model_out=tmp_path / "model.joblib",
            verbose=False,
        )
        with pytest.raises(ValueError, match="Input file is empty"):
            load_and_validate(cfg)

    def test_headers_only_raises(self, tmp_path: Path) -> None:
        csv = """\
            SizeSqFt,Bedrooms,AgeYears,Price
        """
        cfg = _make_cfg(tmp_path, csv)
        with pytest.raises(ValueError, match="CSV file contains headers but no data rows"):
            load_and_validate(cfg)

    def test_all_rows_invalid_raises(self, tmp_path: Path) -> None:
        csv = """\
            SizeSqFt,Bedrooms,AgeYears,Price
            abc,def,ghi,jkl
            xyz,uvw,rst,opq
        """
        cfg = _make_cfg(tmp_path, csv)
        with pytest.raises(ValueError, match="Not enough valid rows"):
            load_and_validate(cfg)

    def test_returns_correct_columns(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path, VALID_CSV)
        df = load_and_validate(cfg)
        assert "SizeSqFt" in df.columns
        assert "Bedrooms" in df.columns
        assert "AgeYears" in df.columns
        assert "Price" in df.columns

    def test_extra_columns_preserved(self, tmp_path: Path) -> None:
        csv = """\
            SizeSqFt,Bedrooms,AgeYears,Price,Neighborhood
            800,1,20,50000,Downtown
            900,2,15,60000,Suburbs
            1100,2,10,75000,Downtown
            1300,3,8,90000,Suburbs
        """
        cfg = _make_cfg(tmp_path, csv)
        df = load_and_validate(cfg)
        assert "Neighborhood" in df.columns


# ---------------------------------------------------------------------------
# parse_args tests
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_defaults(self, tmp_path: Path) -> None:
        cfg = parse_args(["--input", str(tmp_path / "data.csv")])
        assert cfg.test_size == 0.25
        assert cfg.random_state == 42
        assert cfg.model_out == Path("model.joblib")
        assert cfg.verbose is False

    def test_test_size_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="--test-size must be between"):
            parse_args(["--input", "x.csv", "--test-size", "0.9"])

    def test_test_size_too_low(self) -> None:
        with pytest.raises(ValueError, match="--test-size must be between"):
            parse_args(["--input", "x.csv", "--test-size", "0.05"])

    def test_custom_args(self, tmp_path: Path) -> None:
        cfg = parse_args([
            "--input", str(tmp_path / "data.csv"),
            "--test-size", "0.3",
            "--random-state", "99",
            "--model-out", "out/my_model.joblib",
            "--verbose",
        ])
        assert cfg.test_size == 0.3
        assert cfg.random_state == 99
        assert cfg.model_out == Path("out/my_model.joblib")
        assert cfg.verbose is True

    def test_negative_random_state_raises(self) -> None:
        with pytest.raises(ValueError, match="--random-state must be a non-negative"):
            parse_args(["--input", "x.csv", "--random-state", "-1"])

    def test_missing_input_raises(self) -> None:
        with pytest.raises(SystemExit):
            parse_args([])

    def test_boundary_test_size_low(self, tmp_path: Path) -> None:
        cfg = parse_args(["--input", str(tmp_path / "d.csv"), "--test-size", "0.1"])
        assert cfg.test_size == 0.1

    def test_boundary_test_size_high(self, tmp_path: Path) -> None:
        cfg = parse_args(["--input", str(tmp_path / "d.csv"), "--test-size", "0.5"])
        assert cfg.test_size == 0.5
