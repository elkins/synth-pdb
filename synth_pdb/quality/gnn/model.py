"""
synth_pdb.quality.gnn.model
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Graph Attention Network (GAT) for protein structure quality classification.

─────────────────────────────────────────────────────────────────────────────
EDUCATIONAL BACKGROUND — How do GNNs work?
─────────────────────────────────────────────────────────────────────────────

A Graph Neural Network learns by *message passing*: each node aggregates
information from its neighbours, then updates its own representation.
After several rounds of message passing, every node's embedding encodes
information about its local structural environment.

The general message-passing equation is:

    h_v^(k) = UPDATE( h_v^(k-1),  AGG( { h_u^(k-1) : u ∈ N(v) } ) )

where:
  h_v^(k)  = embedding of node v after k message-passing steps
  N(v)     = neighbours of v (residues within 8 Å of v)
  AGG      = aggregation function (mean, sum, attention-weighted sum …)
  UPDATE   = transformation (MLP, etc.)

After K layers, node v's embedding captures information from all nodes
within K hops — i.e., up to K contacts away in the protein.

─────────────────────────────────────────────────────────────────────────────
WHY GRAPH ATTENTION NETWORK (GAT) over plain GCN?
─────────────────────────────────────────────────────────────────────────────

In a standard Graph Convolutional Network (GCN), every neighbour contributes
*equally* to the aggregation (weighted only by node degree for normalisation).

In a Graph Attention Network (GAT), the aggregation weight for each edge
(u → v) is learned:

    α_{uv} = softmax_u ( LeakyReLU( a^T [W h_u ‖ W h_v ‖ W_e e_{uv}] ) )

where W, W_e are learnable weight matrices and e_{uv} is the edge feature
vector (here: Cα distance and sequence separation).

Concrete benefit for structural biology:
  • Backbone contacts (|i-j|=1) and long-range contacts can receive
    different attention weights — the model is not forced to treat them
    equally.
  • Contacts at 3 Å (steric clash!) can be down-weighted by the attention
    mechanism during training.
  • The attention weights α_{uv} are interpretable: after training you can
    visualise which residue–residue contacts the model considers most
    diagnostic of quality.

─────────────────────────────────────────────────────────────────────────────
MULTI-HEAD ATTENTION (heads=4)
─────────────────────────────────────────────────────────────────────────────

Like Multi-Head Attention in Transformers, we run H=4 independent attention
functions in parallel and average (concat=False) their outputs:

    h_v^(k) = σ( (1/H) Σ_h  Σ_{u∈N(v)} α_{uv}^h · W^h h_u )

Each head can specialise: one head may focus on local-sequence contacts,
another on distant cross-contacts, etc.

─────────────────────────────────────────────────────────────────────────────
READOUT — Global Mean Pooling
─────────────────────────────────────────────────────────────────────────────

After 3 message-passing layers each node has a rich local embedding.
We need a single *graph-level* vector to feed into the classifier.

Global Mean Pool:  z_G = (1/N) Σ_v h_v^(3)

This is permutation-invariant — rotating or renumbering residues does not
change z_G.  More expressive readouts (global_add_pool, Set2Set) exist but
mean pooling works well for small proteins and is easy to interpret
(z_G is literally the average residue embedding).

─────────────────────────────────────────────────────────────────────────────
FULL ARCHITECTURE
─────────────────────────────────────────────────────────────────────────────

  Input node features [N, 8]
         │
  ┌──────▼────────────────────────────────────────────┐
  │  GATConv(8 → 64, heads=4, concat=False,           │
  │           edge_dim=2, dropout=0.1)                │ Layer 1
  │  BatchNorm1d(64) → ELU                            │
  └──────┬────────────────────────────────────────────┘
         │
  ┌──────▼────────────────────────────────────────────┐
  │  GATConv(64 → 64, heads=4, concat=False,          │
  │           edge_dim=2, dropout=0.1)                │ Layer 2
  │  BatchNorm1d(64) → ELU                            │
  └──────┬────────────────────────────────────────────┘
         │
  ┌──────▼────────────────────────────────────────────┐
  │  GATConv(64 → 64, heads=4, concat=False,          │
  │           edge_dim=2, dropout=0.1)                │ Layer 3
  │  BatchNorm1d(64) → ELU                            │
  └──────┬────────────────────────────────────────────┘
         │  node embeddings [N, 64]
  global_mean_pool()  →  graph embedding [batch_size, 64]
         │
  ┌──────▼────────────────────────────────────────────┐
  │  Linear(64 → 32) → ELU → Dropout(0.3)            │
  │  Linear(32 → 2)  → log_softmax                   │ MLP head
  └──────┬────────────────────────────────────────────┘
         │  [batch_size, 2]  log P(Bad), log P(Good)

Loss: Negative Log-Likelihood (NLLLoss), equivalent to cross-entropy
      when the input is log-softmax output.
"""

import logging

logger = logging.getLogger(__name__)


def _check_pyg():
    """Raise a clear ImportError if torch_geometric is not installed."""
    try:
        import torch
        import torch_geometric
    except ImportError as exc:
        raise ImportError(
            "torch and torch_geometric are required. "
            "Install with: pip install synth-pdb[gnn]"
        ) from exc


class ProteinGNN:
    """
    Graph Attention Network classifier for protein structure quality.

    ── Design note on the __new__ pattern ─────────────────────────────
    We want ``ProteinGNN`` to be importable without triggering PyTorch
    imports at module load time (so users without PyTorch can still use
    the rest of synth_pdb.quality).

    Using ``__new__`` means ``ProteinGNN(...)`` *returns* an instance of
    the inner ``_ProteinGNNModule`` (a real ``torch.nn.Module``) rather than
    of ``ProteinGNN`` itself.  This is a factory pattern — the outer class
    exists only as a well-named constructor.  The resulting object behaves
    exactly like a plain ``nn.Module``.
    ─────────────────────────────────────────────────────────────────────

    Usage::

        model = ProteinGNN()           # returns a torch.nn.Module
        model.eval()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        # out: [batch_size, 2] log-probabilities (Good class = index 1)
    """

    def __new__(cls, node_features: int = 8, edge_features: int = 2,
                hidden_dim: int = 64, num_classes: int = 2):
        _check_pyg()
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import GATConv, global_mean_pool

        class _ProteinGNNModule(nn.Module):
            """
            The actual torch.nn.Module returned by ProteinGNN().

            Parameters
            ----------
            node_features : int
                Dimensionality of the input node feature vector (default 8,
                matching graph.py's output).
            edge_features : int
                Dimensionality of the edge feature vector (default 2:
                Cα distance, sequence separation).
            hidden_dim : int
                Width of all GATConv and MLP hidden layers.
            num_classes : int
                Output classes. 2 for binary (Bad / Good) classification.
            """

            def __init__(self):
                super().__init__()

                # ── Message-passing layers ─────────────────────────────────
                # GATConv  arguments:
                #   in_channels  — input feature size
                #   out_channels — output feature size PER HEAD
                #   heads        — number of independent attention heads
                #   concat       — if False, heads are AVERAGED (not concatenated)
                #                  keeping the output size = out_channels
                #   edge_dim     — size of edge feature vector
                #   dropout      — attention coefficient dropout (regularisation)
                #
                # Three layers give a receptive field of 3 hops, meaning each
                # node's final embedding captures all nodes reachable in ≤ 3
                # contact-graph steps.  For a 20-residue alpha helix that is
                # typically the entire protein.
                self.conv1 = GATConv(
                    node_features, hidden_dim,
                    heads=4, concat=False, edge_dim=edge_features, dropout=0.1
                )
                # BatchNorm1d normalises each feature across the mini-batch.
                # This prevents "covariate shift" between layers and stabilises
                # training — especially important when protein sizes vary (so N varies).
                self.bn1 = nn.BatchNorm1d(hidden_dim)

                self.conv2 = GATConv(
                    hidden_dim, hidden_dim,
                    heads=4, concat=False, edge_dim=edge_features, dropout=0.1
                )
                self.bn2 = nn.BatchNorm1d(hidden_dim)

                self.conv3 = GATConv(
                    hidden_dim, hidden_dim,
                    heads=4, concat=False, edge_dim=edge_features, dropout=0.1
                )
                self.bn3 = nn.BatchNorm1d(hidden_dim)

                # ── Graph-level MLP classifier head ────────────────────────
                # After readout, z_G is a fixed-size vector regardless of
                # protein length, so a plain MLP can classify it.
                self.lin1 = nn.Linear(hidden_dim, hidden_dim // 2)
                # Dropout(0.3) randomly zeros 30% of activations during training.
                # This forces the model to be robust — it cannot rely on any
                # single feature — and reduces overfitting on the small dataset.
                self.dropout = nn.Dropout(p=0.3)
                self.lin2 = nn.Linear(hidden_dim // 2, num_classes)

                # Store architecture hyperparameters so they can be serialised
                # into the checkpoint alongside the weights (see gnn_classifier.py).
                self.node_features = node_features
                self.edge_features = edge_features
                self.hidden_dim = hidden_dim
                self.num_classes = num_classes

            def forward(self, x, edge_index, edge_attr, batch):
                """
                Forward pass.

                Parameters
                ----------
                x          : Tensor [total_nodes, node_features]
                    Node feature matrix for all proteins in the batch.
                    (Batching concatenates all node matrices; ``batch`` tracks
                    which protein each node belongs to.)
                edge_index : Tensor [2, total_edges]
                    COO edge index.
                edge_attr  : Tensor [total_edges, edge_features]
                    Edge feature matrix.
                batch      : Tensor [total_nodes]
                    Maps each node to its protein index in the batch.
                    e.g. [0,0,...,0, 1,1,...,1, 2,...] for a batch of 3.

                Returns
                -------
                Tensor [batch_size, num_classes]
                    Log-probabilities for each class per protein.
                    Use .exp() to get probabilities.  argmax(-1) for label.
                """

                # ── Layer 1: local geometry ────────────────────────────────
                # After layer 1, each node has aggregated information from its
                # direct neighbours — the first "shell" of contacts.
                # For a helix residue, this includes i±1, i±2, i±3, i+4.
                x = self.conv1(x, edge_index, edge_attr)
                x = self.bn1(x)
                # ELU (Exponential Linear Unit) is chosen over ReLU because:
                #   • ELU has non-zero gradient for negative inputs → avoids
                #     "dying neuron" problem common in deep GNNs
                #   • Smooth at 0, unlike ReLU's sharp kink
                x = F.elu(x)

                # ── Layer 2: neighbourhood of neighbourhoods ───────────────
                # Each node now has information from nodes 2 hops away, i.e.
                # contacts-of-contacts.  This captures secondary structure
                # patterns (e.g. helix turns) and local packing.
                x = self.conv2(x, edge_index, edge_attr)
                x = self.bn2(x)
                x = F.elu(x)

                # ── Layer 3: medium-range interactions ─────────────────────
                # 3-hop receptive field.  For small peptides (≤ 20 residues)
                # this effectively pools global structural information into
                # each node's embedding.
                x = self.conv3(x, edge_index, edge_attr)
                x = self.bn3(x)
                x = F.elu(x)

                # ── Readout: node embeddings → graph embedding ─────────────
                # global_mean_pool sums node embeddings per graph and divides
                # by the number of nodes:
                #     z_G = (1/N) Σ_v h_v^(3)
                # The ``batch`` vector tells PyG which protein each node
                # belongs to so it averages correctly across the batch.
                x = global_mean_pool(x, batch)   # [batch_size, hidden_dim]

                # ── MLP classification head ────────────────────────────────
                x = self.lin1(x)
                x = F.elu(x)
                # Dropout is only active during .train() mode; disabled in
                # .eval() mode (inference), which is the correct behaviour.
                x = self.dropout(x)
                x = self.lin2(x)   # raw logits [batch_size, 2]

                # log_softmax is numerically more stable than log(softmax(x)).
                # Combined with nn.NLLLoss it is mathematically equivalent to
                # nn.CrossEntropyLoss (which takes raw logits).
                # We use log_softmax + NLLLoss to make the probability extraction
                # step (.exp()) explicit and readable in gnn_classifier.py.
                return F.log_softmax(x, dim=-1)

        return _ProteinGNNModule()
