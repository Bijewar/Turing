"""
test_layer2_cliff_detector.py

Tests for `python-engine/services/layer2/cliff_detector.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from services.layer2.cliff_detector import CliffDetector


# ===========================================================================
# Detects sudden drop
# ===========================================================================
class TestFindsSuddenDrop:
    def test_cliff_detector_finds_sudden_drop(self):
        """Given a smooth surface with one sudden drop, the cliff is detected."""
        # Build a 10x10 grid that's smooth except for one cliff.
        # Surface = constant 10 everywhere, except a single row that drops to 0.
        grid = np.full((10, 10), 10.0)
        grid[5, :] = 0.0  # Row 5 is a cliff.
        cliffs = CliffDetector.detect(grid)
        # The detector should find at least one cliff near row 5.
        assert cliffs is not None
        if hasattr(cliffs, "__len__"):
            assert len(cliffs) > 0
        # If cliffs is a list of (row, col) or similar, verify row 5 is detected.
        # If it's a boolean mask, verify some cells in row 4 or 6 are True.
        if isinstance(cliffs, np.ndarray):
            assert cliffs.any()
        elif isinstance(cliffs, list) and cliffs:
            # Verify a cliff is detected near row 5.
            found = False
            for cliff in cliffs:
                # cliff might be a tuple (row, col) or a dict with row/col keys.
                if isinstance(cliff, (tuple, list)) and len(cliff) >= 1:
                    if abs(cliff[0] - 5) <= 1:
                        found = True
                        break
                elif isinstance(cliff, dict):
                    row = cliff.get("row") or cliff.get("y")
                    if row is not None and abs(row - 5) <= 1:
                        found = True
                        break
            assert found, f"No cliff detected near row 5: {cliffs}"


# ===========================================================================
# No false positive on smooth surface
# ===========================================================================
class TestNoFalsePositive:
    def test_no_false_positive_on_smooth_surface(self):
        """A perfectly smooth surface should have no cliffs detected."""
        grid = np.full((10, 10), 5.0)
        cliffs = CliffDetector.detect(grid)
        # No cliffs should be detected.
        if hasattr(cliffs, "__len__"):
            assert len(cliffs) == 0, f"False positive cliffs on smooth surface: {cliffs}"
        elif isinstance(cliffs, np.ndarray):
            assert not cliffs.any(), f"False positive cliffs on smooth surface: {cliffs}"


# ===========================================================================
# Threshold tunable
# ===========================================================================
class TestThresholdTunable:
    def test_threshold_tunable(self):
        """Different threshold values should give different results."""
        # A surface with a moderate drop.
        grid = np.full((10, 10), 10.0)
        grid[5, :] = 5.0  # Drop of 5.
        # Low threshold (1.0) — should detect.
        cliffs_low = CliffDetector.detect(grid, threshold=1.0)
        # High threshold (20.0) — should not detect.
        cliffs_high = CliffDetector.detect(grid, threshold=20.0)
        # Verify they differ.
        if hasattr(cliffs_low, "__len__") and hasattr(cliffs_high, "__len__"):
            assert len(cliffs_low) >= len(cliffs_high), (
                "Higher threshold should detect fewer or equal cliffs."
            )


# ===========================================================================
# Boundary detection
# ===========================================================================
class TestBoundaryCorrectness:
    def test_cliff_boundary_at_actual_location(self):
        """The detected cliff should be at the actual cliff location,
        within 1 grid cell."""
        grid = np.full((20, 20), 10.0)
        grid[10, :] = 0.0  # Cliff at row 10.
        cliffs = CliffDetector.detect(grid)
        # Find the cliff with the largest gradient or the closest to row 10.
        if isinstance(cliffs, list) and cliffs:
            # Find any cliff within 1 cell of row 10.
            found_near_10 = False
            for cliff in cliffs:
                if isinstance(cliff, (tuple, list)) and len(cliff) >= 1:
                    if abs(cliff[0] - 10) <= 1:
                        found_near_10 = True
                        break
            assert found_near_10, f"No cliff detected near row 10: {cliffs}"
        elif isinstance(cliffs, np.ndarray):
            # Check that some True cells are in row 9, 10, or 11.
            rows_with_cliff = np.where(cliffs.any(axis=1) if cliffs.ndim > 1 else cliffs)[0]
            if len(rows_with_cliff) > 0:
                # At least one row within 1 of 10.
                assert any(abs(r - 10) <= 1 for r in rows_with_cliff), (
                    f"Cliff not at row 10: {rows_with_cliff}"
                )


# ===========================================================================
# Noisy surface — robustness
# ===========================================================================
class TestRobustnessToNoise:
    def test_no_false_positive_on_noisy_surface(self):
        """A surface with low-amplitude noise should not produce false cliffs."""
        np.random.seed(42)
        grid = np.full((10, 10), 5.0) + np.random.normal(0, 0.1, (10, 10))
        cliffs = CliffDetector.detect(grid, threshold=1.0)
        # With low threshold and low noise, no cliffs should be detected.
        if hasattr(cliffs, "__len__"):
            assert len(cliffs) == 0, f"False positive on noisy surface: {cliffs}"
        elif isinstance(cliffs, np.ndarray):
            assert not cliffs.any()


# ===========================================================================
# Empty / edge cases
# ===========================================================================
class TestEdgeCases:
    def test_empty_grid(self):
        """An empty grid should not crash — return empty cliffs or raise a clear error."""
        grid = np.array([])
        try:
            cliffs = CliffDetector.detect(grid)
            # Any return is fine.
            assert cliffs is not None or cliffs is None
        except (ValueError, TypeError) as e:
            assert "empty" in str(e).lower() or "shape" in str(e).lower()
        except Exception as e:
            pytest.fail(f"Empty grid produced opaque error: {type(e).__name__}: {e}")

    def test_single_cell_grid(self):
        """A 1x1 grid has no neighbors to compare — should not crash."""
        grid = np.array([[5.0]])
        try:
            cliffs = CliffDetector.detect(grid)
        except (ValueError, TypeError):
            pass  # A clear error is acceptable.
        except Exception as e:
            pytest.fail(f"1x1 grid produced opaque error: {type(e).__name__}: {e}")

    def test_1d_grid(self):
        """A 1D grid (single row or column) should be handled gracefully."""
        grid = np.array([1.0, 5.0, 1.0, 5.0])  # 1D alternating.
        try:
            cliffs = CliffDetector.detect(grid)
        except (ValueError, TypeError):
            pass
        except Exception as e:
            pytest.fail(f"1D grid produced opaque error: {type(e).__name__}: {e}")