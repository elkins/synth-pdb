
import os
import argparse
import logging
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from synth_pdb.generator import generate_pdb_content
from synth_pdb.quality.features import extract_quality_features, get_feature_names

logger = logging.getLogger(__name__)


def generate_dataset(n_samples=200, random_state=42):
    """Generates a balanced dataset of Good and Bad structures.

    Args:
        n_samples: Total number of samples to generate.
        random_state: Seed for reproducible distortion in the Bad (Distorted) class.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Feature matrix X (shape: [n, 8]) and label
        vector y (1 = Good, 0 = Bad).

    Raises:
        RuntimeError: If fewer than 50% of requested samples were successfully generated,
            indicating a systemic generation or feature-extraction failure.
    """
    logger.info("Generating training data (%d samples)...", n_samples)

    import biotite.structure.io.pdb as pdb
    import io

    rng = np.random.default_rng(random_state)

    # Target: 40% Good, 20% Bad (Random), 20% Bad (Distorted), 20% Bad (Clashing)
    n_good = int(n_samples * 0.4)
    n_bad_random = int(n_samples * 0.2)
    n_bad_distorted = int(n_samples * 0.2)
    n_bad_clash = n_samples - n_good - n_bad_random - n_bad_distorted

    logger.info(
        "Dataset split: %d Good, %d Random, %d Distorted, %d Clashing",
        n_good, n_bad_random, n_bad_distorted, n_bad_clash,
    )

    X = []
    y = []
    failure_counts = {"good": 0, "random": 0, "distorted": 0, "clash": 0}

    # ------------------------------------------------------------------
    # 1. Good (Alpha Helix)
    # ------------------------------------------------------------------
    for i in range(n_good):
        if i % 10 == 0:
            logger.info("  Good %d/%d", i, n_good)
        try:
            pdb_content = generate_pdb_content(length=20, conformation='alpha', minimize_energy=False)
            X.append(extract_quality_features(pdb_content))
            y.append(1)
        except Exception as e:
            failure_counts["good"] += 1
            logger.warning("Good sample %d failed: %s", i, e, exc_info=True)

    # ------------------------------------------------------------------
    # 2. Bad (Random)
    # ------------------------------------------------------------------
    for i in range(n_bad_random):
        if i % 10 == 0:
            logger.info("  Random %d/%d", i, n_bad_random)
        try:
            pdb_content = generate_pdb_content(length=20, conformation='random', minimize_energy=False)
            X.append(extract_quality_features(pdb_content))
            y.append(0)
        except Exception as e:
            failure_counts["random"] += 1
            logger.warning("Random sample %d failed: %s", i, e, exc_info=True)

    # ------------------------------------------------------------------
    # 3. Bad (Distorted)
    # ------------------------------------------------------------------
    for i in range(n_bad_distorted):
        if i % 10 == 0:
            logger.info("  Distorted %d/%d", i, n_bad_distorted)
        try:
            clean = generate_pdb_content(length=20, conformation='alpha', minimize_energy=False)
            f = io.StringIO(clean)
            struc_obj = pdb.PDBFile.read(f).get_structure(model=1)
            struc_obj.coord += rng.normal(0, 0.5, struc_obj.coord.shape)
            f_out = io.StringIO()
            pdb_file = pdb.PDBFile()
            pdb_file.set_structure(struc_obj)
            pdb_file.write(f_out)
            X.append(extract_quality_features(f_out.getvalue()))
            y.append(0)
        except Exception as e:
            failure_counts["distorted"] += 1
            logger.warning("Distorted sample %d failed: %s", i, e, exc_info=True)

    # ------------------------------------------------------------------
    # 4. Bad (Single Clash)
    # ------------------------------------------------------------------
    for i in range(n_bad_clash):
        if i % 10 == 0:
            logger.info("  Clashing %d/%d", i, n_bad_clash)
        try:
            clean = generate_pdb_content(length=20, conformation='alpha', minimize_energy=False)
            f = io.StringIO(clean)
            struc_obj = pdb.PDBFile.read(f).get_structure(model=1)

            # Move a backbone atom (CA) to overlap another to ensure detection
            ca_indices = [j for j, a in enumerate(struc_obj) if a.atom_name == "CA"]
            if len(ca_indices) >= 5:
                # Move CA of residue 2 to CA of residue 5 (clash!)
                struc_obj.coord[ca_indices[1]] = struc_obj.coord[ca_indices[4]]

            f_out = io.StringIO()
            pdb_file = pdb.PDBFile()
            pdb_file.set_structure(struc_obj)
            pdb_file.write(f_out)
            X.append(extract_quality_features(f_out.getvalue()))
            y.append(0)
        except Exception as e:
            failure_counts["clash"] += 1
            logger.warning("Clash sample %d failed: %s", i, e, exc_info=True)

    # ------------------------------------------------------------------
    # Failure summary
    # ------------------------------------------------------------------
    total_failures = sum(failure_counts.values())
    if total_failures > 0:
        logger.warning(
            "Generation failures: Good=%d, Random=%d, Distorted=%d, Clash=%d (total=%d)",
            failure_counts["good"], failure_counts["random"],
            failure_counts["distorted"], failure_counts["clash"],
            total_failures,
        )
    else:
        logger.info("All samples generated successfully (0 failures).")

    if not X:
        raise RuntimeError("No samples were generated. Check generator and feature extractor.")

    X = np.array(X)
    y = np.array(y)

    # Enforce minimum yield: abort if more than half the requested samples failed.
    min_required = int(n_samples * 0.5)
    if len(X) < min_required:
        raise RuntimeError(
            f"Only {len(X)} of {n_samples} samples generated successfully "
            f"(minimum required: {min_required}). Failures: {failure_counts}. "
            "Inspect WARNING logs above for root causes."
        )

    # ------------------------------------------------------------------
    # Feature statistics (diagnostic)
    # ------------------------------------------------------------------
    logger.info("\nFeature Statistics (Mean):")
    feature_names = get_feature_names()
    header = f"{'Feature':<30} | {'Good':<10} | {'Bad':<10}"
    logger.info(header)
    logger.info("-" * 56)

    good_means = []
    bad_means = []
    for i, name in enumerate(feature_names):
        good_mean = np.mean(X[y == 1][:, i]) if np.any(y == 1) else 0.0
        bad_mean  = np.mean(X[y == 0][:, i]) if np.any(y == 0) else 0.0
        good_means.append(good_mean)
        bad_means.append(bad_mean)
        logger.info("%-30s | %-10.4f | %-10.4f", name, good_mean, bad_mean)

    # Warn if the primary discriminating feature shows insufficient separation.
    rama_idx = feature_names.index("ramachandran_favored_pct")
    if good_means[rama_idx] <= bad_means[rama_idx] + 5.0:
        logger.warning(
            "Low Ramachandran separation: Good=%.1f%% vs Bad=%.1f%%. "
            "The 'random' conformation may not be sufficiently distinct. "
            "Consider reviewing the Bad class composition.",
            good_means[rama_idx], bad_means[rama_idx],
        )
    else:
        logger.info(
            "Ramachandran separation OK: Good=%.1f%% vs Bad=%.1f%%",
            good_means[rama_idx], bad_means[rama_idx],
        )

    return X, y


def train_model(output_path, n_samples=200):
    X, y = generate_dataset(n_samples=n_samples)

    if X.ndim != 2 or X.shape[0] == 0:
        raise RuntimeError(
            f"Feature matrix has unexpected shape {X.shape}. "
            "Expected a non-empty 2D array."
        )

    logger.info("\nTraining RandomForest Classifier...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        max_features="sqrt",   # sklearn default for classifiers; explicit for clarity
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    y_pred = clf.predict(X_test)
    logger.info("\nModel Evaluation:")
    try:
        acc = accuracy_score(y_test, y_pred)
        logger.info("Accuracy: %.2f", acc)
        report = classification_report(
            y_test, y_pred, target_names=['Bad', 'Good'], labels=[0, 1]
        )
        logger.info("\n%s", report)
    except Exception as e:
        logger.warning("Evaluation warning: %s", e)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------
    feature_names = get_feature_names()
    importances = clf.feature_importances_
    logger.info("\nFeature Importance:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        logger.info("  %-35s %.4f", name, imp)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    logger.info("\nSaving model to %s...", output_path)
    joblib.dump(clf, output_path)
    logger.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Train Structure Quality Filter Model (Random Forest classifier)"
    )
    parser.add_argument(
        "--output",
        default="synth_pdb/quality/models/quality_filter_v1.joblib",
        help="Output path for model",
    )
    parser.add_argument(
        "--n-samples", type=int, default=200, help="Number of samples to generate"
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Random seed for reproducible distortion in Bad (Distorted) class",
    )
    args = parser.parse_args()

    # Ensure directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    train_model(args.output, n_samples=args.n_samples)
