"""
Benchmark: GNN Quality Scorer vs. Random Forest Quality Classifier.

Both models are trained on identical data from the same RNG seed and evaluated
on the same held-out test set, giving a fair apples-to-apples comparison.

Metrics reported:
  - Accuracy, Precision, Recall, F1 (per class + macro avg)
  - ROC-AUC
  - Mean inference time per sample (ms)
  - Confidence calibration (mean predicted probability for correct vs wrong predictions)

Usage:
    python scripts/benchmark_classifiers.py
    python scripts/benchmark_classifiers.py --n-samples 200 --gnn-epochs 50
"""

import argparse
import io
import logging
import os
import time
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared data generation
# ---------------------------------------------------------------------------

def generate_shared_dataset(n_samples: int = 100, random_state: int = 42):
    """
    Generate PDB strings + labels for both classifiers.
    Returns (pdb_list, y) — the GNN builds graph features directly,
    the RF builds tabular features via extract_quality_features.
    """
    import biotite.structure.io.pdb as pdb_io
    from synth_pdb.generator import generate_pdb_content

    rng = np.random.default_rng(random_state)

    n_good         = int(n_samples * 0.4)
    n_bad_random   = int(n_samples * 0.2)
    n_bad_distort  = int(n_samples * 0.2)
    n_bad_clash    = n_samples - n_good - n_bad_random - n_bad_distort

    logger.info(
        "Generating %d shared samples: %d Good, %d Random, %d Distorted, %d Clashing",
        n_samples, n_good, n_bad_random, n_bad_distort, n_bad_clash,
    )

    pdbs, labels = [], []
    failures = 0

    def _add(pdb_content, label):
        pdbs.append(pdb_content)
        labels.append(label)

    for i in range(n_good):
        try:
            _add(generate_pdb_content(length=20, conformation="alpha", minimize_energy=False), 1)
        except Exception as e:
            failures += 1
            logger.warning("Good %d: %s", i, e)

    for i in range(n_bad_random):
        try:
            _add(generate_pdb_content(length=20, conformation="random", minimize_energy=False), 0)
        except Exception as e:
            failures += 1
            logger.warning("Random %d: %s", i, e)

    for i in range(n_bad_distort):
        try:
            clean = generate_pdb_content(length=20, conformation="alpha", minimize_energy=False)
            so = pdb_io.PDBFile.read(io.StringIO(clean)).get_structure(model=1)
            so.coord += rng.normal(0, 0.5, so.coord.shape)
            f = io.StringIO()
            pf = pdb_io.PDBFile(); pf.set_structure(so); pf.write(f)
            _add(f.getvalue(), 0)
        except Exception as e:
            failures += 1
            logger.warning("Distorted %d: %s", i, e)

    for i in range(n_bad_clash):
        try:
            clean = generate_pdb_content(length=20, conformation="alpha", minimize_energy=False)
            so = pdb_io.PDBFile.read(io.StringIO(clean)).get_structure(model=1)
            ca = [j for j, a in enumerate(so) if a.atom_name == "CA"]
            if len(ca) >= 5:
                so.coord[ca[1]] = so.coord[ca[4]]
            f = io.StringIO()
            pf = pdb_io.PDBFile(); pf.set_structure(so); pf.write(f)
            _add(f.getvalue(), 0)
        except Exception as e:
            failures += 1
            logger.warning("Clash %d: %s", i, e)

    if failures:
        logger.warning("Total generation failures: %d", failures)

    return pdbs, np.array(labels, dtype=np.int64)


# ---------------------------------------------------------------------------
# RF training + evaluation
# ---------------------------------------------------------------------------

def benchmark_rf(pdbs_train, y_train, pdbs_test, y_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
    from synth_pdb.quality.features import extract_quality_features

    logger.info("--- Random Forest ---")

    # Build tabular features
    def _build_feats(pdbs, split_name):
        X, valid_y, skipped = [], [], 0
        for i, pdb in enumerate(pdbs):
            try:
                X.append(extract_quality_features(pdb))
                valid_y.append(y_train[i] if split_name == "train" else y_test[i])
            except Exception as e:
                skipped += 1
                logger.warning("RF feature extraction skipped sample %d (%s): %s", i, split_name, e)
        if skipped:
            logger.warning("RF: skipped %d / %d %s samples.", skipped, len(pdbs), split_name)
        return np.array(X), np.array(valid_y)

    X_train, y_tr = _build_feats(pdbs_train, "train")
    X_test,  y_te = _build_feats(pdbs_test,  "test")

    clf = RandomForestClassifier(n_estimators=100, max_features="sqrt", random_state=42)

    t0 = time.perf_counter()
    clf.fit(X_train, y_tr)
    train_time = time.perf_counter() - t0
    logger.info("RF training time: %.2f s", train_time)

    # Inference timing
    t0 = time.perf_counter()
    y_prob = clf.predict_proba(X_test)[:, 1]
    inf_time_ms = (time.perf_counter() - t0) / len(X_test) * 1000

    y_pred = (y_prob >= 0.5).astype(int)
    acc    = accuracy_score(y_te, y_pred)
    auc    = roc_auc_score(y_te, y_prob) if len(np.unique(y_te)) > 1 else float("nan")
    report = classification_report(y_te, y_pred, target_names=["Bad", "Good"], labels=[0,1])

    correct_probs = y_prob[y_pred == y_te]
    wrong_probs   = y_prob[y_pred != y_te] if (y_pred != y_te).any() else np.array([float("nan")])

    return {
        "name": "Random Forest",
        "accuracy": acc,
        "auc": auc,
        "report": report,
        "inf_ms": inf_time_ms,
        "train_s": train_time,
        "mean_prob_correct": float(np.mean(np.abs(correct_probs - (1 - y_te[y_pred == y_te])))),
        "mean_conf_correct": float(np.mean(np.maximum(y_prob[y_pred == y_te], 1 - y_prob[y_pred == y_te]))),
        "mean_conf_wrong":   float(np.mean(np.maximum(wrong_probs, 1 - wrong_probs))),
        "n_test": len(y_te),
    }


# ---------------------------------------------------------------------------
# GNN training + evaluation
# ---------------------------------------------------------------------------

def benchmark_gnn(pdbs_train, y_train, pdbs_test, y_test, epochs: int = 50, hidden_dim: int = 64):
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
    from synth_pdb.quality.gnn.graph import build_protein_graph
    from synth_pdb.quality.gnn.model import ProteinGNN

    logger.info("--- GNN (GATConv, hidden_dim=%d, epochs=%d) ---", hidden_dim, epochs)

    def _build_graphs(pdbs, ys, split_name):
        graphs, skipped = [], 0
        for i, (pdb, label) in enumerate(zip(pdbs, ys)):
            try:
                g = build_protein_graph(pdb)
                g.y = torch.tensor([int(label)], dtype=torch.long)
                graphs.append(g)
            except Exception as e:
                skipped += 1
                logger.warning("GNN graph build skipped sample %d (%s): %s", i, split_name, e)
        if skipped:
            logger.warning("GNN: skipped %d / %d %s samples.", skipped, len(pdbs), split_name)
        return graphs

    train_graphs = _build_graphs(pdbs_train, y_train, "train")
    test_graphs  = _build_graphs(pdbs_test,  y_test,  "test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProteinGNN(node_features=8, edge_features=2, hidden_dim=hidden_dim, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)

    t0 = time.perf_counter()
    model.train()
    for epoch in range(1, epochs + 1):
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            log_probs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = F.nll_loss(log_probs, batch.y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch % max(1, epochs // 5) == 0:
            logger.info("  GNN epoch %3d/%d  loss=%.4f", epoch, epochs, loss.item())
    train_time = time.perf_counter() - t0
    logger.info("GNN training time: %.2f s", train_time)

    # Inference
    from torch_geometric.data import Batch
    model.eval()
    test_batch = Batch.from_data_list(test_graphs).to(device)
    y_te = np.array([g.y.item() for g in test_graphs])

    t0 = time.perf_counter()
    with torch.no_grad():
        log_probs = model(test_batch.x, test_batch.edge_index, test_batch.edge_attr, test_batch.batch)
    inf_time_ms = (time.perf_counter() - t0) / len(test_graphs) * 1000

    y_prob = log_probs.exp()[:, 1].cpu().numpy()
    y_pred = (y_prob >= 0.5).astype(int)
    acc  = accuracy_score(y_te, y_pred)
    auc  = roc_auc_score(y_te, y_prob) if len(np.unique(y_te)) > 1 else float("nan")
    report = classification_report(y_te, y_pred, target_names=["Bad", "Good"], labels=[0,1])

    correct_mask = y_pred == y_te
    wrong_mask   = ~correct_mask

    return {
        "name": f"GNN (GATConv, h={hidden_dim}, ep={epochs})",
        "accuracy": acc,
        "auc": auc,
        "report": report,
        "inf_ms": inf_time_ms,
        "train_s": train_time,
        "mean_conf_correct": float(np.mean(np.maximum(y_prob[correct_mask], 1-y_prob[correct_mask]))) if correct_mask.any() else float("nan"),
        "mean_conf_wrong":   float(np.mean(np.maximum(y_prob[wrong_mask],   1-y_prob[wrong_mask])))   if wrong_mask.any()  else float("nan"),
        "n_test": len(y_te),
    }


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(results):
    print("\n" + "=" * 70)
    print("  CLASSIFIER BENCHMARK SUMMARY")
    print("=" * 70)

    header = f"{'Metric':<30} {'Random Forest':>18} {'GNN':>18}"
    print(header)
    print("-" * 70)

    rf, gnn = results[0], results[1]

    rows = [
        ("Accuracy",                f"{rf['accuracy']:.4f}",          f"{gnn['accuracy']:.4f}"),
        ("ROC-AUC",                 f"{rf['auc']:.4f}",               f"{gnn['auc']:.4f}"),
        ("Train time (s)",          f"{rf['train_s']:.2f}",           f"{gnn['train_s']:.2f}"),
        ("Inference / sample (ms)", f"{rf['inf_ms']:.3f}",            f"{gnn['inf_ms']:.3f}"),
        ("Mean conf (correct)",     f"{rf['mean_conf_correct']:.4f}", f"{gnn['mean_conf_correct']:.4f}"),
        ("Mean conf (wrong)",       f"{rf['mean_conf_wrong']:.4f}",   f"{gnn['mean_conf_wrong']:.4f}"),
        ("Test samples",            str(rf['n_test']),                 str(gnn['n_test'])),
    ]

    for label, rv, gv in rows:
        print(f"  {label:<28} {rv:>18} {gv:>18}")

    print("\n--- Random Forest classification report ---")
    print(rf["report"])
    print("--- GNN classification report ---")
    print(gnn["report"])
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark GNN vs RF quality classifier")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Total samples to generate (shared across both models)")
    parser.add_argument("--gnn-epochs", type=int, default=50,
                        help="Training epochs for GNN")
    parser.add_argument("--gnn-hidden-dim", type=int, default=64,
                        help="GNN hidden dimension")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data for held-out test set")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    from sklearn.model_selection import train_test_split

    # 1. Generate shared data
    pdbs, y = generate_shared_dataset(n_samples=args.n_samples, random_state=args.random_state)
    idx = np.arange(len(pdbs))
    idx_train, idx_test = train_test_split(
        idx, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    pdbs_train = [pdbs[i] for i in idx_train]
    pdbs_test  = [pdbs[i] for i in idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    logger.info("Train: %d  Test: %d", len(pdbs_train), len(pdbs_test))

    # 2. Benchmark both
    results = []
    results.append(benchmark_rf(pdbs_train, y_train, pdbs_test, y_test))
    results.append(benchmark_gnn(
        pdbs_train, y_train, pdbs_test, y_test,
        epochs=args.gnn_epochs,
        hidden_dim=args.gnn_hidden_dim,
    ))

    # 3. Print summary
    _print_summary(results)


if __name__ == "__main__":
    main()
