"""
TDD tests for synth_pdb/quality/gnn/model.py and gnn_classifier.py.

Written BEFORE the implementation exists — all tests should fail initially.
"""
import unittest
import numpy as np

import pytest
torch = pytest.importorskip("torch", reason="PyTorch not installed")
pyg = pytest.importorskip("torch_geometric", reason="torch_geometric not installed")

from synth_pdb.generator import generate_pdb_content


def _make_helix_pdb(length: int = 15) -> str:
    return generate_pdb_content(length=length, conformation="alpha", minimize_energy=False)


class TestGNNModelForwardPass(unittest.TestCase):
    """Tests for synth_pdb.quality.gnn.model.ProteinGNN."""

    @classmethod
    def setUpClass(cls):
        from synth_pdb.quality.gnn.model import ProteinGNN
        from synth_pdb.quality.gnn.graph import build_protein_graph
        cls.model = ProteinGNN(node_features=8, edge_features=2, hidden_dim=32, num_classes=2)
        cls.model.eval()
        cls.graph = build_protein_graph(_make_helix_pdb(15))

    def test_forward_pass_output_shape(self):
        """Single-graph forward pass must return logits of shape [1, 2]."""
        from torch_geometric.data import Batch
        batch = Batch.from_data_list([self.graph])
        with torch.no_grad():
            out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        self.assertEqual(
            list(out.shape),
            [1, 2],
            msg=f"Expected output shape [1, 2], got {list(out.shape)}"
        )

    def test_output_is_log_probabilities(self):
        """Output must be log-probabilities: all values ≤ 0 and exp() sums to 1."""
        from torch_geometric.data import Batch
        batch = Batch.from_data_list([self.graph])
        with torch.no_grad():
            log_probs = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        # log-softmax output: all values <= 0
        self.assertTrue(
            (log_probs <= 0).all().item(),
            msg="log_softmax outputs must all be ≤ 0"
        )
        # exp() must sum to ~1 per row
        probs = log_probs.exp()
        row_sum = probs.sum(dim=-1).item()
        self.assertAlmostEqual(
            row_sum, 1.0, places=4,
            msg=f"Probability row sum is {row_sum:.6f}, expected 1.0"
        )

    def test_batched_forward_pass(self):
        """Batched forward pass with 3 graphs must return [3, 2] output."""
        from torch_geometric.data import Batch
        from synth_pdb.quality.gnn.graph import build_protein_graph
        graphs = [
            build_protein_graph(_make_helix_pdb(10)),
            build_protein_graph(_make_helix_pdb(12)),
            build_protein_graph(_make_helix_pdb(15)),
        ]
        batch = Batch.from_data_list(graphs)
        with torch.no_grad():
            out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        self.assertEqual(
            list(out.shape),
            [3, 2],
            msg=f"Expected batched output [3, 2], got {list(out.shape)}"
        )


class TestGNNClassifierAPI(unittest.TestCase):
    """Tests that GNNQualityClassifier.predict() matches the RF classifier API."""

    @classmethod
    def setUpClass(cls):
        from synth_pdb.quality.gnn.gnn_classifier import GNNQualityClassifier
        # Untrained model — we just test API shape / types, not accuracy
        cls.clf = GNNQualityClassifier()
        cls.pdb = _make_helix_pdb(20)

    def test_predict_returns_three_tuple(self):
        """predict() must return a (bool, float, dict) triple."""
        result = self.clf.predict(self.pdb)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3, msg="predict() must return (is_good, probability, features)")

    def test_predict_is_good_is_bool(self):
        is_good, prob, feats = self.clf.predict(self.pdb)
        self.assertIsInstance(is_good, (bool, np.bool_))

    def test_predict_probability_in_unit_interval(self):
        _, prob, _ = self.clf.predict(self.pdb)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_predict_features_is_dict(self):
        _, _, feats = self.clf.predict(self.pdb)
        self.assertIsInstance(feats, dict)
        self.assertGreater(len(feats), 0, "Feature dict must be non-empty")


if __name__ == "__main__":
    unittest.main()
