"""
TDD integration tests for GNN training and checkpoint save/load.

Written BEFORE the implementation exists — all tests should fail initially.
"""
import os
import tempfile
import unittest

import pytest
torch = pytest.importorskip("torch", reason="PyTorch not installed")
pyg = pytest.importorskip("torch_geometric", reason="torch_geometric not installed")

from synth_pdb.generator import generate_pdb_content
from synth_pdb.quality.gnn.graph import build_protein_graph


def _quick_dataset(n_good: int = 5, n_bad: int = 5):
    """Tiny balanced dataset for fast integration tests."""
    from torch_geometric.data import Data
    graphs, labels = [], []
    for _ in range(n_good):
        pdb = generate_pdb_content(length=15, conformation="alpha", minimize_energy=False)
        g = build_protein_graph(pdb)
        g.y = torch.tensor([1], dtype=torch.long)
        graphs.append(g)
        labels.append(1)
    for _ in range(n_bad):
        pdb = generate_pdb_content(length=15, conformation="random", minimize_energy=False)
        g = build_protein_graph(pdb)
        g.y = torch.tensor([0], dtype=torch.long)
        graphs.append(g)
        labels.append(0)
    return graphs


class TestGNNTraining(unittest.TestCase):
    """Integration tests for the GNN training loop."""

    def test_gnn_trains_without_error(self):
        """Training for 3 epochs on 10 tiny samples must complete without exception."""
        from synth_pdb.quality.gnn.model import ProteinGNN
        from torch_geometric.loader import DataLoader

        graphs = _quick_dataset(n_good=5, n_bad=5)
        loader = DataLoader(graphs, batch_size=5, shuffle=True)

        model = ProteinGNN(node_features=8, edge_features=2, hidden_dim=32, num_classes=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        model.train()
        for epoch in range(3):
            epoch_loss = 0.0
            for batch in loader:
                optimizer.zero_grad()
                log_probs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = torch.nn.functional.nll_loss(log_probs, batch.y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss)

        # Loss must be finite (no NaN/Inf from bad initialisation or graph problems)
        for i, l in enumerate(losses):
            self.assertTrue(
                np.isfinite(l),
                msg=f"Loss at epoch {i} is non-finite: {l}"
            )

    def test_gnn_saves_and_loads_checkpoint(self):
        """Saving and reloading a checkpoint must produce identical predictions."""
        from synth_pdb.quality.gnn.gnn_classifier import GNNQualityClassifier

        clf = GNNQualityClassifier()
        pdb = generate_pdb_content(length=15, conformation="alpha", minimize_energy=False)

        # Get prediction before save
        is_good_before, prob_before, _ = clf.predict(pdb)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name

        try:
            clf.save(checkpoint_path)
            self.assertTrue(os.path.exists(checkpoint_path), "Checkpoint file was not created")
            self.assertGreater(os.path.getsize(checkpoint_path), 0, "Checkpoint file is empty")

            # Load fresh instance
            clf2 = GNNQualityClassifier(model_path=checkpoint_path)
            is_good_after, prob_after, _ = clf2.predict(pdb)

            self.assertEqual(
                is_good_before,
                is_good_after,
                msg="is_good differs after save/load cycle"
            )
            self.assertAlmostEqual(
                prob_before, prob_after, places=4,
                msg=f"Probability changed after save/load: {prob_before:.6f} → {prob_after:.6f}"
            )
        finally:
            if os.path.exists(checkpoint_path):
                os.unlink(checkpoint_path)


import numpy as np  # used in test_gnn_trains_without_error

if __name__ == "__main__":
    unittest.main()
