from __future__ import annotations

import textwrap
from pathlib import Path

import joblib
import pytest

from m2_2 import Config, main, train_and_evaluate, load_and_validate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _write_csv(tmp_path: Path, csv_text: str = VALID_CSV) -> Path:
    p = tmp_path / "data.csv"
    p.write_text(textwrap.dedent(csv_text))
    return p


def _make_cfg(tmp_path: Path, csv_text: str = VALID_CSV) -> Config:
    p = _write_csv(tmp_path, csv_text)
    return Config(
        input_path=p,
        test_size=0.25,
        random_state=42,
        model_out=tmp_path / "model.joblib",
        verbose=False,
    )


# ---------------------------------------------------------------------------
# train_and_evaluate tests
# ---------------------------------------------------------------------------

class TestTrainAndEvaluate:
    def test_returns_zero_on_success(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        df = load_and_validate(cfg)
        result = train_and_evaluate(df, cfg)
        assert result == 0

    def test_creates_model_file(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        df = load_and_validate(cfg)
        train_and_evaluate(df, cfg)
        assert cfg.model_out.exists()

    def test_model_file_is_loadable(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        df = load_and_validate(cfg)
        train_and_evaluate(df, cfg)
        pipeline = joblib.load(cfg.model_out)
        assert hasattr(pipeline, "predict")

    def test_model_can_predict(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        df = load_and_validate(cfg)
        train_and_evaluate(df, cfg)
        pipeline = joblib.load(cfg.model_out)
        import pandas as pd
        sample = pd.DataFrame({"SizeSqFt": [1000], "Bedrooms": [2], "AgeYears": [10]})
        predictions = pipeline.predict(sample)
        assert len(predictions) == 1
        assert predictions[0] > 0

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        cfg = Config(
            input_path=_write_csv(tmp_path),
            test_size=0.25,
            random_state=42,
            model_out=tmp_path / "subdir" / "model.joblib",
            verbose=False,
        )
        df = load_and_validate(cfg)
        train_and_evaluate(df, cfg)
        assert cfg.model_out.exists()

    def test_different_random_states_produce_different_splits(self, tmp_path: Path) -> None:
        csv_path = _write_csv(tmp_path)
        cfg1 = Config(
            input_path=csv_path,
            test_size=0.25,
            random_state=42,
            model_out=tmp_path / "m1.joblib",
            verbose=False,
        )
        cfg2 = Config(
            input_path=csv_path,
            test_size=0.25,
            random_state=99,
            model_out=tmp_path / "m2.joblib",
            verbose=False,
        )
        df = load_and_validate(cfg1)
        train_and_evaluate(df, cfg1)
        train_and_evaluate(df, cfg2)
        # Both should succeed and produce model files
        assert cfg1.model_out.exists()
        assert cfg2.model_out.exists()


# ---------------------------------------------------------------------------
# main() integration tests
# ---------------------------------------------------------------------------

class TestMain:
    def test_success(self, tmp_path: Path) -> None:
        csv_path = _write_csv(tmp_path)
        model_out = tmp_path / "model.joblib"
        exit_code = main([
            "--input", str(csv_path),
            "--model-out", str(model_out),
        ])
        assert exit_code == 0
        assert model_out.exists()

    def test_missing_file_returns_error_code(self, tmp_path: Path) -> None:
        exit_code = main([
            "--input", str(tmp_path / "nonexistent.csv"),
            "--model-out", str(tmp_path / "model.joblib"),
        ])
        assert exit_code == 2

    def test_invalid_csv_returns_error_code(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.csv"
        p.write_text("col1,col2\n1,2\n")
        exit_code = main([
            "--input", str(p),
            "--model-out", str(tmp_path / "model.joblib"),
        ])
        assert exit_code == 2

    def test_verbose_mode(self, tmp_path: Path) -> None:
        csv_path = _write_csv(tmp_path)
        exit_code = main([
            "--input", str(csv_path),
            "--model-out", str(tmp_path / "model.joblib"),
            "--verbose",
        ])
        assert exit_code == 0

    def test_custom_test_size(self, tmp_path: Path) -> None:
        csv_path = _write_csv(tmp_path)
        exit_code = main([
            "--input", str(csv_path),
            "--model-out", str(tmp_path / "model.joblib"),
            "--test-size", "0.4",
        ])
        assert exit_code == 0
