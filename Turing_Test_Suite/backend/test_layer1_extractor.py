"""
test_layer1_extractor.py

Tests for `python-engine/services/layer1/extractor.py`.

Covers:
- `extract_numeric_features`, `extract_summary_statistics` (Mayank's path)
- `UniversalExtractor.extract_data`, `validate_data`, `_extract_tables_from_pdf`
- `AmbiguityDetector.analyze_graph`

Each test's docstring names the exact SWARM-3 bug ID it catches.
"""

from __future__ import annotations

import io
import os
import inspect
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from services.layer1.extractor import (
    extract_numeric_features,
    extract_summary_statistics,
    UniversalExtractor,
    AmbiguityDetector,
    _PENALTY_LOW_EDGE_CONF,
    _PENALTY_HIDDEN_NODE,
    _PENALTY_CONTRADICTED_EDGE,
    _PENALTY_BAD_FIT,
    _PENALTY_LOW_DEGREE,
)


# ===========================================================================
# Random jitter seed (BUG #5) — reproducibility
# ===========================================================================
class TestValidateDataSeed:
    """Locks in SWARM-3 BUG #5: validate_data adds random jitter without
    seeding — the entire pipeline is irreproducible. The fix must use a
    deterministic RNG (seeded from an env var or run_id)."""

    def test_validate_data_does_not_add_unseeded_jitter(self):
        """Catches SWARM-3 BUG #5: `numeric_df + np.random.normal(...)` uses
        the global numpy RNG without seeding. Two consecutive calls on
        the SAME input DataFrame must produce IDENTICAL output. The buggy
        code produces different output every call (irreproducible)."""
        df = pd.DataFrame({
            "a": np.arange(50.0),
            "b": np.arange(50.0) * 2 + 1.0,
        })
        out1, _ = UniversalExtractor.validate_data(df.copy())
        out2, _ = UniversalExtractor.validate_data(df.copy())
        # Both DataFrames must be EXACTLY equal (jitter is deterministic).
        assert np.array_equal(out1["a"].values, out2["a"].values), (
            "BUG #5: validate_data uses unseeded np.random.normal — two "
            "consecutive runs produce different jitter. The pipeline is "
            "irreproducible."
        )
        assert np.array_equal(out1["b"].values, out2["b"].values)

    def test_validate_data_uses_default_rng_or_seed(self):
        """The fix must use `np.random.default_rng(seed)` (a seeded RNG) or
        `np.random.seed(seed)` before the jitter call. This test inspects
        the source code to confirm the fix."""
        src = inspect.getsource(UniversalExtractor.validate_data)
        # The fix should NOT use bare np.random.normal() — it should use a
        # seeded RNG (default_rng with a seed, or np.random.seed before).
        uses_unseeded_global = (
            "np.random.normal(" in src and
            "default_rng" not in src and
            "np.random.seed" not in src
        )
        assert not uses_unseeded_global, (
            "BUG #5: validate_data uses bare np.random.normal() without "
            "seeding — the fix must use np.random.default_rng(seed) or "
            "np.random.seed(seed) before the jitter call."
        )

    def test_validate_data_reproducible(self):
        """Spec-level reproducibility test: same input → same output."""
        df = pd.DataFrame({
            "a": np.linspace(0, 1, 50),
            "b": np.linspace(1, 2, 50),
        })
        out1, _ = UniversalExtractor.validate_data(df.copy())
        out2, _ = UniversalExtractor.validate_data(df.copy())
        # The numerical values must be bit-identical.
        np.testing.assert_array_equal(out1["a"].values, out2["a"].values)
        np.testing.assert_array_equal(out1["b"].values, out2["b"].values)


# ===========================================================================
# NaN column handling
# ===========================================================================
class TestValidateDataNanColumns:
    """Per SWARM-3 BUG #35: `df.dropna()` can destroy the entire dataset for
    sparse data. The current implementation calls `df.dropna()` (no subset)
    which drops any row with ANY NaN. For a column that's all-NaN, this
    drops ALL rows."""

    def test_validate_data_handles_all_nan_column(self):
        """A column with all NaN must NOT crash validate_data. The current
        behavior: `df.dropna()` keeps rows that have at least one non-NaN
        value (the all-NaN column is dropped later by `select_dtypes`)."""
        df = pd.DataFrame({
            "a": np.arange(50.0),
            "all_nan": [np.nan] * 50,
        })
        # Must not crash.
        out, warnings = UniversalExtractor.validate_data(df)
        # The all-NaN column should be dropped (not present in output).
        assert "all_nan" not in out.columns
        assert "a" in out.columns

    def test_validate_data_handles_some_nan_rows(self):
        """A DataFrame with some NaN rows must drop those rows (current
        behavior). Verify the row count is reduced correctly."""
        df = pd.DataFrame({
            "a": [1.0, 2.0, np.nan, 4.0, 5.0] * 10,
            "b": [10.0, 20.0, 30.0, np.nan, 50.0] * 10,
        })
        out, warnings = UniversalExtractor.validate_data(df)
        # Rows containing any NaN must be dropped.
        # Original 50 rows; rows 3, 4 (5%5=0,1,2,3,4 — index 2 and 3) have NaN
        # in one column each, so they're dropped. That's 20 rows total.
        # Plus any NaN-NaN rows.
        assert len(out) < 50, "Rows with NaN must be dropped."
        assert len(out) > 0, "Result must be non-empty."
        assert out.isna().sum().sum() == 0, "No NaN should remain in output."


# ===========================================================================
# Mixed types
# ===========================================================================
class TestValidateDataMixedTypes:
    def test_validate_data_handles_mixed_types(self):
        """A DataFrame with mixed int/float/str columns must not crash.
        Numeric columns are kept; string columns are coerced or dropped."""
        df = pd.DataFrame({
            "int_col": np.arange(50, dtype=np.int64),
            "float_col": np.arange(50.0),
            "str_col": [f"item_{i}" for i in range(50)],
        })
        out, _ = UniversalExtractor.validate_data(df)
        # Int and float columns must be kept (numeric).
        assert "int_col" in out.columns
        assert "float_col" in out.columns
        # String column should be dropped (no numeric values).
        assert "str_col" not in out.columns

    def test_validate_data_coerces_numeric_strings(self):
        """A column of numeric strings (e.g., '1.0', '2.0') must be coerced
        to numeric, not dropped."""
        df = pd.DataFrame({
            "num_str": [f"{i}.0" for i in range(50)],
        })
        out, _ = UniversalExtractor.validate_data(df)
        # The column must be coerced to numeric and kept.
        assert "num_str" in out.columns
        assert out["num_str"].dtype.kind in ("i", "f"), (
            f"Numeric string column must be coerced to numeric; got {out['num_str'].dtype}."
        )


# ===========================================================================
# Datetime columns
# ===========================================================================
class TestValidateDataDatetime:
    def test_validate_data_handles_datetime_columns(self):
        """Datetime columns must NOT crash validate_data. They are
        non-numeric and should be dropped (datetime is not a number for
        causal discovery)."""
        df = pd.DataFrame({
            "ts": pd.date_range("2023-01-01", periods=50, freq="D"),
            "val": np.arange(50.0),
        })
        # Must not crash.
        out, _ = UniversalExtractor.validate_data(df)
        # The datetime column should be dropped (not numeric).
        assert "ts" not in out.columns
        assert "val" in out.columns

    def test_validate_data_datetime_does_not_propagate(self):
        """Datetime columns should not produce unexpected numeric coercion."""
        df = pd.DataFrame({
            "ts": pd.date_range("2023-01-01", periods=50, freq="D"),
        })
        # With only a datetime column (no numeric), validate_data must
        # raise ValueError "No numeric columns found".
        with pytest.raises(ValueError, match="No numeric columns"):
            UniversalExtractor.validate_data(df)


# ===========================================================================
# Empty DataFrame
# ===========================================================================
class TestValidateDataEmpty:
    def test_validate_data_empty_dataframe_raises(self):
        """An empty DataFrame must raise a clear ValueError, not crash with
        a cryptic numpy error."""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            UniversalExtractor.validate_data(df)


# ===========================================================================
# extract_numeric_features
# ===========================================================================
class TestExtractNumericFeatures:
    def test_extract_numeric_features_basic(self):
        """A simple tabular content dict must produce numeric columns and
        a 2D data matrix."""
        content = {
            "columns": ["a", "b", "c"],
            "rows": [
                [1.0, "x", 3.0],
                [2.0, "y", 4.0],
                [3.0, "z", 5.0],
            ],
            "row_count": 3,
        }
        cols, matrix = extract_numeric_features(content)
        # Only 'a' and 'c' are numeric; 'b' (strings) must be dropped.
        assert cols == ["a", "c"]
        assert len(matrix) == 3
        assert matrix[0] == [1.0, 3.0]

    def test_extract_numeric_features_handles_empty(self):
        """Empty content must return empty lists."""
        cols, matrix = extract_numeric_features({"columns": [], "rows": []})
        assert cols == []
        assert matrix == []

    def test_extract_numeric_features_fills_missing_with_mean(self):
        """Missing numeric values must be filled with the column mean (mean
        of all NON-missing values, per the implementation)."""
        content = {
            "columns": ["a"],
            "rows": [
                [1.0],
                [2.0],
                [None],  # missing → filled with mean of [1, 2, 3] = 2.0
                [3.0],
            ],
            "row_count": 4,
        }
        cols, matrix = extract_numeric_features(content)
        assert cols == ["a"]
        # Row 2 (index 2) must be the mean of [1.0, 2.0, 3.0] = 2.0.
        assert matrix[2] == [2.0], (
            f"Missing value must be filled with mean of non-None values "
            f"([1, 2, 3] = 2.0); got {matrix[2]}."
        )


# ===========================================================================
# extract_summary_statistics
# ===========================================================================
class TestExtractSummaryStatistics:
    def test_summary_statistics_basic(self):
        """Summary statistics for a single column must produce mean, std,
        min, max."""
        content = {
            "columns": ["a"],
            "rows": [[float(i)] for i in range(10)],
            "row_count": 10,
        }
        stats = extract_summary_statistics(content)
        assert "a" in stats
        assert "mean" in stats["a"]
        assert "std" in stats["a"]
        assert "min" in stats["a"]
        assert "max" in stats["a"]
        assert stats["a"]["min"] == 0.0
        assert stats["a"]["max"] == 9.0
        # Mean of [0..9] = 4.5
        assert abs(stats["a"]["mean"] - 4.5) < 0.001

    def test_summary_statistics_empty(self):
        """Empty content must return empty dict."""
        stats = extract_summary_statistics({"columns": [], "rows": []})
        assert stats == {}


# ===========================================================================
# AmbiguityDetector — penalty weights
# ===========================================================================
class TestAmbiguityDetectorPenalties:
    """Spec compliance: penalty weights must match the documented values."""

    def test_penalty_values_match_spec(self):
        """Verify the exact penalty values from the spec (lines 292-296)."""
        assert _PENALTY_LOW_EDGE_CONF == 20
        assert _PENALTY_HIDDEN_NODE == 30
        assert _PENALTY_CONTRADICTED_EDGE == 25
        assert _PENALTY_BAD_FIT == 15
        assert _PENALTY_LOW_DEGREE == 10

    def test_hidden_node_penalty(self):
        """A hidden node must receive the _PENALTY_HIDDEN_NODE penalty."""
        graph_data = {
            "nodes": [
                {"id": "A", "label": "A", "confidence": 100.0,
                 "is_hidden": True},
                {"id": "B", "label": "B", "confidence": 100.0},
            ],
            "edges": [
                {"source": "A", "target": "B", "relation": "CAUSES",
                 "confidence": 95.0},
                {"source": "B", "target": "A", "relation": "CAUSES",
                 "confidence": 95.0},  # Make degree > 1 for A and B
            ],
        }
        result = AmbiguityDetector.analyze_graph(graph_data)
        hidden_node = next(n for n in result["nodes"] if n["id"] == "A")
        assert hidden_node["ambiguity_score"] == _PENALTY_HIDDEN_NODE, (
            f"Hidden node A expected ambiguity={_PENALTY_HIDDEN_NODE}, "
            f"got {hidden_node['ambiguity_score']}."
        )

    def test_low_degree_penalty(self):
        """A node with degree=1 must receive the _PENALTY_LOW_DEGREE penalty."""
        graph_data = {
            "nodes": [
                {"id": "A", "label": "A", "confidence": 100.0},
                {"id": "B", "label": "B", "confidence": 100.0},
                {"id": "C", "label": "C", "confidence": 100.0},
            ],
            "edges": [
                # A has degree 2 (indeg 1, outdeg 1).
                {"source": "A", "target": "B", "relation": "CAUSES",
                 "confidence": 95.0},
                # B has degree 2.
                {"source": "B", "target": "C", "relation": "CAUSES",
                 "confidence": 95.0},
                # C has degree 1 (indeg 1, outdeg 0) → +10 penalty.
            ],
        }
        result = AmbiguityDetector.analyze_graph(graph_data)
        node_c = next(n for n in result["nodes"] if n["id"] == "C")
        assert node_c["ambiguity_score"] == _PENALTY_LOW_DEGREE, (
            f"Node C (degree=1) expected ambiguity={_PENALTY_LOW_DEGREE}, "
            f"got {node_c['ambiguity_score']}."
        )

    def test_urgent_nodes_sorted_desc(self):
        """urgent_nodes must be sorted DESC by urgency_score (per the spec
        comment in analyze_graph). This locks in the contract that
        ConfidenceChecker depends on."""
        graph_data = {
            "nodes": [
                {"id": "low", "label": "L", "confidence": 100.0,
                 "is_hidden": False},   # urgency 0
                {"id": "mid", "label": "M", "confidence": 100.0,
                 "is_hidden": True},     # urgency 30
                {"id": "high", "label": "H", "confidence": 100.0,
                 "is_hidden": True},     # urgency 30 too
            ],
            # Make all nodes have degree > 1 so low-degree penalty doesn't apply.
            "edges": [
                {"source": "low", "target": "mid", "confidence": 95.0},
                {"source": "low", "target": "high", "confidence": 95.0},
                {"source": "mid", "target": "high", "confidence": 95.0},
            ],
        }
        result = AmbiguityDetector.analyze_graph(graph_data)
        # urgent_nodes sorted DESC by urgency_score.
        urgencies = [u["urgency_score"] for u in result["urgent_nodes"]]
        assert urgencies == sorted(urgencies, reverse=True), (
            f"urgent_nodes must be sorted DESC by urgency_score; got {urgencies}"
        )


# ===========================================================================
# extract_data file-extension routing (smoke test)
# ===========================================================================
class TestExtractDataRouting:
    def test_extract_data_unsupported_extension_raises(self):
        """Unsupported file extension must raise a clear ValueError."""
        with pytest.raises(ValueError, match="Unsupported"):
            UniversalExtractor.extract_data(b"contents", "foo.unsupported")

    def test_extract_data_csv(self):
        """CSV extraction must produce a validated DataFrame."""
        csv_bytes = b"a,b,c\n" + b"\n".join(
            f"{i},{i*2},{i*3}".encode() for i in range(50)
        )
        df, warnings = UniversalExtractor.extract_data(csv_bytes, "test.csv")
        assert "a" in df.columns
        assert "b" in df.columns
        assert "c" in df.columns
        assert len(df) > 0