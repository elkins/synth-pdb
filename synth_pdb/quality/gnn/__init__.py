"""
synth_pdb.quality.gnn
~~~~~~~~~~~~~~~~~~~~~
PyTorch Geometric-based protein quality GNN scorer.

Install requirements:  pip install synth-pdb[gnn]
"""
from .graph import build_protein_graph
from .model import ProteinGNN
from .gnn_classifier import GNNQualityClassifier

__all__ = ["build_protein_graph", "ProteinGNN", "GNNQualityClassifier"]
