"""
Train the GNN-based protein quality classifier.

Usage
-----
    python scripts/train_gnn_quality_filter.py
    python scripts/train_gnn_quality_filter.py --n-samples 400 --epochs 100
    python scripts/train_gnn_quality_filter.py --n-samples 40 --epochs 5 --output /tmp/test.pt

The script reuses generate_dataset() from the RF training script so both
models train on exactly the same synthetic data splits.
"""

import argparse
import logging
import os
import sys

import numpy as np

logger = logging.getLogger(__name__)


def build_graph_dataset(X_feats: np.ndarray, y: np.ndarray, pdb_list: list):
    """
    Convert raw PDB strings to PyG Data objects with ground-truth labels.

    Args:
        X_feats: Unused here (kept for API parity with RF script); the graph
            builder extracts its own richer features directly from the PDB.
        y: Label array (1 = Good, 0 = Bad), length N.
        pdb_list: List of N PDB strings.

    Returns:
        List of torch_geometric.data.Data objects with .y set.
    """
    import torch
    from synth_pdb.quality.gnn.graph import build_protein_graph

    graphs = []
    failures = 0
    for i, (pdb_content, label) in enumerate(zip(pdb_list, y)):
        try:
            g = build_protein_graph(pdb_content)
            g.y = torch.tensor([int(label)], dtype=torch.long)
            graphs.append(g)
        except Exception as e:
            failures += 1
            logger.warning("Graph build failed for sample %d: %s", i, e)

    if failures:
        logger.warning("%d / %d graph builds failed.", failures, len(pdb_list))
    return graphs


def generate_pdb_dataset(n_samples: int = 200, random_state: int = 42):
    """
    Generate synthetic PDB strings for the four structural classes used in
    training. Returns (pdb_list, y) rather than (X_feats, y) since the GNN
    builds its own graph features directly from the PDB atoms.
    """
    import biotite.structure.io.pdb as pdb_io
    import io
    from synth_pdb.generator import generate_pdb_content

    rng = np.random.default_rng(random_state)

    n_good = int(n_samples * 0.4)
    n_bad_random = int(n_samples * 0.2)
    n_bad_distorted = int(n_samples * 0.2)
    n_bad_clash = n_samples - n_good - n_bad_random - n_bad_distorted

    logger.info(
        "Generating: %d Good, %d Random, %d Distorted, %d Clashing",
        n_good, n_bad_random, n_bad_distorted, n_bad_clash,
    )

    pdbs, labels = [], []
    failure_counts = {"good": 0, "random": 0, "distorted": 0, "clash": 0}

    # 1. Good (Alpha Helix)
    for i in range(n_good):
        if i % 20 == 0:
            logger.info("  Good %d/%d", i, n_good)
        try:
            pdbs.append(generate_pdb_content(length=20, conformation="alpha", minimize_energy=False))
            labels.append(1)
        except Exception as e:
            failure_counts["good"] += 1
            logger.warning("Good sample %d failed: %s", i, e, exc_info=True)

    # 2. Bad (Random coil)
    for i in range(n_bad_random):
        if i % 10 == 0:
            logger.info("  Random %d/%d", i, n_bad_random)
        try:
            pdbs.append(generate_pdb_content(length=20, conformation="random", minimize_energy=False))
            labels.append(0)
        except Exception as e:
            failure_counts["random"] += 1
            logger.warning("Random sample %d failed: %s", i, e, exc_info=True)

    # 3. Bad (Distorted)
    for i in range(n_bad_distorted):
        if i % 10 == 0:
            logger.info("  Distorted %d/%d", i, n_bad_distorted)
        try:
            clean = generate_pdb_content(length=20, conformation="alpha", minimize_energy=False)
            f = io.StringIO(clean)
            struc_obj = pdb_io.PDBFile.read(f).get_structure(model=1)
            struc_obj.coord += rng.normal(0, 0.5, struc_obj.coord.shape)
            f_out = io.StringIO()
            pdb_file = pdb_io.PDBFile()
            pdb_file.set_structure(struc_obj)
            pdb_file.write(f_out)
            pdbs.append(f_out.getvalue())
            labels.append(0)
        except Exception as e:
            failure_counts["distorted"] += 1
            logger.warning("Distorted sample %d failed: %s", i, e, exc_info=True)

    # 4. Bad (Single Clash)
    for i in range(n_bad_clash):
        if i % 10 == 0:
            logger.info("  Clashing %d/%d", i, n_bad_clash)
        try:
            clean = generate_pdb_content(length=20, conformation="alpha", minimize_energy=False)
            f = io.StringIO(clean)
            struc_obj = pdb_io.PDBFile.read(f).get_structure(model=1)
            ca_idx = [j for j, a in enumerate(struc_obj) if a.atom_name == "CA"]
            if len(ca_idx) >= 5:
                struc_obj.coord[ca_idx[1]] = struc_obj.coord[ca_idx[4]]
            f_out = io.StringIO()
            pdb_file = pdb_io.PDBFile()
            pdb_file.set_structure(struc_obj)
            pdb_file.write(f_out)
            pdbs.append(f_out.getvalue())
            labels.append(0)
        except Exception as e:
            failure_counts["clash"] += 1
            logger.warning("Clash sample %d failed: %s", i, e, exc_info=True)

    total_failures = sum(failure_counts.values())
    if total_failures:
        logger.warning("Generation failures: %s (total=%d)", failure_counts, total_failures)
    else:
        logger.info("All %d samples generated successfully.", len(pdbs))

    if len(pdbs) < int(n_samples * 0.5):
        raise RuntimeError(
            f"Only {len(pdbs)} of {n_samples} samples generated. "
            f"Failures: {failure_counts}"
        )

    return pdbs, np.array(labels, dtype=np.int64)


def train_gnn(output_path: str, n_samples: int = 200, epochs: int = 50,
              hidden_dim: int = 64, lr: float = 1e-3, random_state: int = 42):
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.loader import DataLoader
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, accuracy_score
    except ImportError as exc:
        logger.error("Missing dependency: %s. Run: pip install synth-pdb[gnn]", exc)
        sys.exit(1)

    from synth_pdb.quality.gnn.model import ProteinGNN
    from synth_pdb.quality.gnn.gnn_classifier import GNNQualityClassifier

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------
    logger.info("=== GNN Quality Scorer Training ===")
    pdbs, y = generate_pdb_dataset(n_samples=n_samples, random_state=random_state)

    logger.info("Building protein graphs...")
    graphs = build_graph_dataset(None, y, pdbs)

    if not graphs:
        raise RuntimeError("No graphs were built. Aborting training.")

    # Rebuild labels from graphs (some may have failed building)
    y_graphs = np.array([g.y.item() for g in graphs])

    # Train/test split
    idx = np.arange(len(graphs))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=y_graphs)
    train_graphs = [graphs[i] for i in idx_train]
    test_graphs  = [graphs[i] for i in idx_test]

    logger.info(
        "Dataset: %d train / %d test  (Good: %d, Bad: %d)",
        len(train_graphs), len(test_graphs),
        int(np.sum(y_graphs == 1)), int(np.sum(y_graphs == 0)),
    )

    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
    test_loader  = DataLoader(test_graphs,  batch_size=16, shuffle=False)

    # ------------------------------------------------------------------
    # Model, optimiser, scheduler
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on: %s", device)

    model = ProteinGNN(node_features=8, edge_features=2, hidden_dim=hidden_dim, num_classes=2)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    logger.info("Training for %d epochs...", epochs)
    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            log_probs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = F.nll_loss(log_probs, batch.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            preds = log_probs.argmax(dim=-1)
            correct += (preds == batch.y).sum().item()
            total += batch.num_graphs

        scheduler.step()
        train_acc = correct / total if total > 0 else 0.0

        if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            logger.info(
                "Epoch %3d/%d  loss=%.4f  train_acc=%.3f  lr=%.2e",
                epoch, epochs, total_loss / total, train_acc,
                scheduler.get_last_lr()[0],
            )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            log_probs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            preds = log_probs.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch.y.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    logger.info("\n=== Test Evaluation ===")
    logger.info("Accuracy: %.4f", acc)
    logger.info(
        "\n%s",
        classification_report(all_labels, all_preds, target_names=["Bad", "Good"], labels=[0, 1])
    )

    # ------------------------------------------------------------------
    # Save via GNNQualityClassifier
    # ------------------------------------------------------------------
    clf = GNNQualityClassifier.__new__(GNNQualityClassifier)
    clf.model = model.cpu()
    clf._model_path = None
    clf.save(output_path)
    logger.info("Done! Model saved to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Train GNN Protein Quality Classifier (Graph Attention Network)"
    )
    parser.add_argument(
        "--output",
        default="synth_pdb/quality/models/gnn_quality_v1.pt",
        help="Output path for the .pt checkpoint",
    )
    parser.add_argument("--n-samples", type=int, default=200, help="Training samples to generate")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--hidden-dim", type=int, default=64, help="GNN hidden dimension")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--random-state", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    train_gnn(
        output_path=args.output,
        n_samples=args.n_samples,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        random_state=args.random_state,
    )
