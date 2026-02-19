"""
TDD tests for synth_pdb/quality/gnn/graph.py (PDB → PyG Data conversion).

Written BEFORE the implementation exists — all tests should fail initially.
"""
import unittest
import numpy as np

import pytest
torch = pytest.importorskip("torch", reason="PyTorch not installed")
pyg = pytest.importorskip("torch_geometric", reason="torch_geometric not installed")

from synth_pdb.generator import generate_pdb_content


def _make_helix_pdb(length: int = 20) -> str:
    return generate_pdb_content(length=length, conformation="alpha", minimize_energy=False)


class TestGraphBuilder(unittest.TestCase):
    """Tests for synth_pdb.quality.gnn.graph.build_protein_graph."""

    @classmethod
    def setUpClass(cls):
        # Import here so the error is a test failure, not a collection error
        from synth_pdb.quality.gnn.graph import build_protein_graph
        cls.build = staticmethod(build_protein_graph)
        cls.pdb = _make_helix_pdb(20)
        cls.data = cls.build(cls.pdb)

    def test_graph_has_correct_node_count(self):
        """A 20-residue protein must produce exactly 20 nodes."""
        self.assertEqual(
            self.data.x.shape[0],
            20,
            msg=f"Expected 20 nodes (residues), got {self.data.x.shape[0]}"
        )

    def test_node_features_shape(self):
        """Node feature tensor must be shape [N, 8]."""
        self.assertEqual(
            self.data.x.shape[1],
            8,
            msg=f"Expected 8 node features per residue, got {self.data.x.shape[1]}"
        )

    def test_edges_are_bidirectional(self):
        """For every directed edge (i→j), the reverse edge (j→i) must also exist."""
        edge_index = self.data.edge_index  # shape [2, E]
        edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        for src, dst in list(edges):
            self.assertIn(
                (dst, src),
                edges,
                msg=f"Edge ({src}→{dst}) exists but reverse ({dst}→{src}) is missing. "
                    "Graph must be undirected (bidirectional edges)."
            )

    def test_no_self_loops(self):
        """No residue should have an edge to itself."""
        edge_index = self.data.edge_index
        src, dst = edge_index[0].tolist(), edge_index[1].tolist()
        self_loops = [(s, d) for s, d in zip(src, dst) if s == d]
        self.assertEqual(
            len(self_loops),
            0,
            msg=f"Found {len(self_loops)} self-loop(s): {self_loops[:5]}"
        )

    def test_sin_cos_encoding_bounded(self):
        """Sin/cos dihedral features (indices 0–3) must lie in [-1, 1]."""
        x = self.data.x.numpy()
        for col in range(4):  # sin(phi), cos(phi), sin(psi), cos(psi)
            col_min, col_max = x[:, col].min(), x[:, col].max()
            self.assertGreaterEqual(
                float(col_min), -1.0 - 1e-6,
                msg=f"Column {col} min {col_min:.4f} is below -1 (sin/cos must be bounded)"
            )
            self.assertLessEqual(
                float(col_max), 1.0 + 1e-6,
                msg=f"Column {col} max {col_max:.4f} is above +1 (sin/cos must be bounded)"
            )

    def test_edge_features_shape(self):
        """Edge feature tensor must have shape [E, 2]."""
        self.assertEqual(
            self.data.edge_attr.shape[1],
            2,
            msg=f"Expected 2 edge features (distance, seq_separation), "
                f"got {self.data.edge_attr.shape[1]}"
        )

    def test_ca_distance_edges_below_threshold(self):
        """All edges must connect Cα pairs within 8 Å (the construction threshold)."""
        distances = self.data.edge_attr[:, 0].numpy()  # first edge feature = Cα distance
        max_dist = float(distances.max())
        self.assertLessEqual(
            max_dist,
            8.0 + 1e-3,
            msg=f"Max Cα distance in edges is {max_dist:.2f} Å, exceeds 8 Å threshold"
        )


if __name__ == "__main__":
    unittest.main()
