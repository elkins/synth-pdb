"""
TDD tests for scripts/train_quality_filter.py and synth_pdb/quality/features.py.

These tests were written BEFORE the fixes, so they are expected to FAIL against
the original code. They document the required behavior after the fixes are applied.
"""
import ast
import textwrap
import unittest
from pathlib import Path
import numpy as np

SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "train_quality_filter.py"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_script_ast() -> ast.Module:
    src = SCRIPT_PATH.read_text()
    return ast.parse(src, filename=str(SCRIPT_PATH))


def _bare_except_count(tree: ast.Module) -> int:
    """Count bare `except:` clauses (no exception type specified)."""
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            count += 1
    return count


def _get_rf_keyword(tree: ast.Module, keyword: str):
    """Return the value node for a keyword arg in a RandomForestClassifier() call."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "RandomForestClassifier"
        ):
            for kw in node.keywords:
                if kw.arg == keyword:
                    return kw.value
    return None  # keyword not present


# ---------------------------------------------------------------------------
# Test 1: No bare except clauses
# ---------------------------------------------------------------------------

class TestNoBareExcept(unittest.TestCase):
    """Fails before fix: the script has bare `except: pass` blocks."""

    def test_no_bare_except_in_generate_dataset(self):
        tree = _load_script_ast()
        count = _bare_except_count(tree)
        self.assertEqual(
            count,
            0,
            msg=(
                f"Found {count} bare `except:` clause(s) in {SCRIPT_PATH}. "
                "All exceptions must be caught as `except Exception as e:` so that "
                "failures are logged rather than silently discarded."
            ),
        )


# ---------------------------------------------------------------------------
# Test 2: RandomForest uses max_features="sqrt"
# ---------------------------------------------------------------------------

class TestRFHyperparameters(unittest.TestCase):
    """Fails before fix: max_features=None is used today."""

    def test_max_features_is_sqrt(self):
        tree = _load_script_ast()
        value_node = _get_rf_keyword(tree, "max_features")

        if value_node is None:
            # Keyword absent → sklearn uses its default which is "sqrt" for classifiers.
            # This is acceptable, but we prefer explicit.
            return

        # If present, it must be the string "sqrt", not None.
        self.assertIsInstance(
            value_node,
            ast.Constant,
            msg="max_features must be a constant (string), not a complex expression.",
        )
        self.assertEqual(
            value_node.value,
            "sqrt",
            msg=(
                f'max_features is set to {value_node.value!r}. '
                "It should be \"sqrt\" to maintain the randomization that makes "
                "Random Forests robust and reduces overfitting."
            ),
        )


# ---------------------------------------------------------------------------
# Test 3: generate_dataset returns close to expected sample count
# ---------------------------------------------------------------------------

class TestGenerateDatasetBalance(unittest.TestCase):
    """
    Fails in spirit before the fix: the fix makes failures visible and
    enforces a minimum yield. We test that the function at least returns
    a non-trivially-sized dataset, and that both classes are present.
    """

    def test_dataset_minimum_yield(self):
        # Import dynamically so that import errors surface clearly
        import importlib.util, sys

        spec = importlib.util.spec_from_file_location(
            "train_quality_filter", str(SCRIPT_PATH)
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        n_samples = 20
        X, y = mod.generate_dataset(n_samples=n_samples)

        self.assertIsInstance(X, np.ndarray, "X must be a numpy array")
        self.assertIsInstance(y, np.ndarray, "y must be a numpy array")

        min_expected = int(n_samples * 0.5)
        self.assertGreaterEqual(
            len(X),
            min_expected,
            msg=(
                f"generate_dataset returned only {len(X)} samples from a requested "
                f"{n_samples}. At least {min_expected} are required. "
                "Check for silently swallowed exceptions."
            ),
        )

        # Both class labels must be present
        unique_labels = set(y.tolist())
        self.assertIn(1, unique_labels, "No 'Good' (label=1) samples were generated")
        self.assertIn(0, unique_labels, "No 'Bad' (label=0) samples were generated")

        # Feature matrix must be 2D with the correct number of features (8)
        self.assertEqual(X.ndim, 2, "Feature matrix X must be 2D")
        self.assertEqual(
            X.shape[1],
            8,
            msg=f"Expected 8 features per sample, got {X.shape[1]}",
        )


# ---------------------------------------------------------------------------
# Test 4: "Good" vs "random" conformations are feature-separable
# ---------------------------------------------------------------------------

class TestFeatureSeparability(unittest.TestCase):
    """
    Validates the medium-severity concern: 'random' conformation must produce
    measurably worse Ramachandran statistics than 'alpha' (Good).
    Fails if the two classes are indistinguishable on the primary feature.
    """

    def test_good_vs_random_ramachandran_separation(self):
        from synth_pdb.generator import generate_pdb_content
        from synth_pdb.quality.features import extract_quality_features

        n_each = 5  # Small — this is a unit test, not a training run
        good_rama = []
        bad_rama = []

        for _ in range(n_each):
            pdb = generate_pdb_content(length=20, conformation="alpha", minimize_energy=False)
            feat = extract_quality_features(pdb)
            good_rama.append(feat[0])  # ramachandran_favored_pct

        for _ in range(n_each):
            pdb = generate_pdb_content(length=20, conformation="random", minimize_energy=False)
            feat = extract_quality_features(pdb)
            bad_rama.append(feat[0])

        mean_good = np.mean(good_rama)
        mean_bad = np.mean(bad_rama)

        self.assertGreater(
            mean_good,
            mean_bad + 5.0,
            msg=(
                f"Ramachandran favored % is not clearly separated: "
                f"Good mean={mean_good:.1f}%, Random mean={mean_bad:.1f}%. "
                "The 'random' conformation does not produce meaningfully worse "
                "Ramachandran statistics than 'alpha'. Consider a different or "
                "additional 'bad' conformation for training data diversity."
            ),
        )


if __name__ == "__main__":
    unittest.main()
