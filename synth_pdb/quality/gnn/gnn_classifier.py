"""
synth_pdb.quality.gnn.gnn_classifier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Drop-in replacement for :class:`synth_pdb.quality.classifier.ProteinQualityClassifier`
using the GNN model.

─────────────────────────────────────────────────────────────────────────────
EDUCATIONAL BACKGROUND — Inference with a GNN classifier
─────────────────────────────────────────────────────────────────────────────

Once the GNN is trained, using it at inference time follows this flow:

    PDB string
        │
        ▼  (graph.py)
    Data object (nodes, edges, features)
        │
        ▼  (Batch.from_data_list)
    PyG Batch  ← wraps a single graph in a batch of size 1
        │
        ▼  (model.forward)
    log-probabilities  [1, 2]    ← log P(Bad), log P(Good)
        │
        ▼  (.exp()[:, 1])
    prob_good  ∈ [0, 1]          ← probability that the structure is high-quality
        │
        ▼  (> 0.5 threshold)
    is_good  True / False        ← binary quality judgement

─────────────────────────────────────────────────────────────────────────────
DESIGN CONTRACT — Same API as ProteinQualityClassifier (RF)
─────────────────────────────────────────────────────────────────────────────

Both classifiers expose:
    predict(pdb_str) → (is_good: bool, probability: float, features: dict)

This lets downstream code swap between the RF and GNN model without changes:

    from synth_pdb.quality.classifier    import ProteinQualityClassifier   # RF
    from synth_pdb.quality.gnn.gnn_classifier import GNNQualityClassifier  # GNN

    clf = GNNQualityClassifier()       # or ProteinQualityClassifier()
    is_good, prob, feats = clf.predict(pdb_string)

─────────────────────────────────────────────────────────────────────────────
CHECKPOINT FORMAT (.pt)
─────────────────────────────────────────────────────────────────────────────

GNN weights are saved with torch.save() as a dict:
    {
      "state_dict"   : OrderedDict of parameter tensors,
      "node_features": int,    ← architecture metadata
      "edge_features": int,
      "hidden_dim"   : int,
      "num_classes"  : int,
    }

We store architecture metadata alongside weights so the model can be
re-instantiated without any external configuration file.  This is the
standard pattern for "self-describing" PyTorch checkpoints.

Compare this to scikit-learn's joblib format (RF), which pickles the entire
fitted estimator object (weights + architecture + preprocessing all merged).
The PyTorch approach is more portable across Python / PyTorch versions.
"""

import logging
import os
from typing import Optional, Tuple, Dict

import numpy as np

logger = logging.getLogger(__name__)

# Default checkpoint path (bundled inside the package after training)
_DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(__file__), "..", "models", "gnn_quality_v1.pt"
)

# Feature names matching graph.py's node feature ordering.
# Used to build the feature dict returned by predict() — useful for debugging
# and for producing human-readable explanations of a prediction.
_FEATURE_NAMES = [
    "sin_phi",        # backbone dihedral φ — sine component
    "cos_phi",        # backbone dihedral φ — cosine component
    "sin_psi",        # backbone dihedral ψ — sine component
    "cos_psi",        # backbone dihedral ψ — cosine component
    "b_factor_norm",  # normalised crystallographic temperature factor
    "seq_position",   # normalised sequence position (0=N-term, 1=C-term)
    "is_n_terminus",  # 1 if this is the N-terminal residue, else 0
    "is_c_terminus",  # 1 if this is the C-terminal residue, else 0
]


class GNNQualityClassifier:
    """
    GNN-based protein structure quality classifier.

    Predicts whether a PDB structure is "High Quality" (biophysically plausible,
    good Ramachandran geometry, no steric clashes) or "Low Quality".

    ── When is a GNN better than a Random Forest? ─────────────────────────
    The RF classifier uses hand-crafted, per-structure summary statistics
    (e.g. "fraction of residues in favoured Ramachandran regions").

    The GNN works directly on the full residue interaction graph, so it can:
      • Learn WHICH specific contacts are problematic, not just aggregate counts
      • Capture spatial patterns (e.g. a single clashing i/i+4 contact pair)
      • Generalise to protein classes or contact patterns not seen in training
        (because the pattern recogniser is learned, not hand-engineered)

    The trade-off is training time and interpretability:
      • RF:  ~0.03 s to train, ~0.1 ms/sample inference, instant setup
      • GNN: ~3 s to train,   ~0.3 ms/sample inference, richer features
    ────────────────────────────────────────────────────────────────────────
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: Path to a .pt checkpoint written by GNNQualityClassifier.save().
                        If None, looks for the default bundled checkpoint.
                        If no checkpoint is found, initialises a random-weight model
                        (useful for testing graph construction without training).
        """
        self.model = None
        self._model_path: Optional[str] = None

        if model_path:
            self.load(model_path)
        else:
            default = os.path.normpath(_DEFAULT_CHECKPOINT)
            if os.path.exists(default):
                self.load(default)
            else:
                logger.info(
                    "No pre-trained GNN checkpoint found at %s. "
                    "Classifier initialised with a random-weight model. "
                    "Run scripts/train_gnn_quality_filter.py to train.",
                    default,
                )
                # A freshly initialised (untrained) model still produces valid
                # probability outputs — they just won't be meaningful.  This
                # allows the predict() API to be called for graph-construction
                # testing without requiring a trained model.
                self._init_fresh_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, pdb_content: str) -> Tuple[bool, float, Dict[str, float]]:
        """
        Predict the quality of a PDB structure.

        ── What happens inside ──────────────────────────────────────────
        1. build_protein_graph(pdb_content) → PyG Data object
        2. Batch.from_data_list([graph])    → single-element batch
        3. model.forward(...)               → log-probabilities [1, 2]
        4. .exp()[0, 1]                     → P(Good) ∈ [0, 1]
        5. > 0.5 threshold                  → is_good bool

        The feature dict is derived from the NODE features, not from the
        model's internal activations.  It shows the mean value of each
        input feature across all residues — useful for debugging why the
        model assigned a particular score.
        ─────────────────────────────────────────────────────────────────

        Args:
            pdb_content: PDB-format string.

        Returns:
            is_good (bool): True if P(Good) > 0.5.
            probability (float): P(Good), in [0, 1].  Values near 0.5 indicate
                the model is uncertain; values near 0 or 1 are confident.
            features (dict): Mean per-feature summary of the input graph's
                node feature matrix.  Useful for logging and introspection.

        Raises:
            ImportError: If torch or torch_geometric are not installed.
            ValueError: If the PDB contains too few residues to build a graph.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "torch is required for GNNQualityClassifier. "
                "Install with: pip install synth-pdb[gnn]"
            ) from exc

        from .graph import build_protein_graph
        from torch_geometric.data import Batch

        # Step 1 — Build graph from PDB coordinates
        graph = build_protein_graph(pdb_content)

        # Step 2 — Wrap in a batch (required by PyG's DataLoader API even for
        # single graphs; the model's global_mean_pool needs a ``batch`` vector)
        batch = Batch.from_data_list([graph])

        # Step 3 — Forward pass (no gradient needed at inference time)
        self.model.eval()
        with torch.no_grad():
            log_probs = self.model(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )
            # .exp() undoes the log: converts log P(class) → P(class)
            # Index 1 = "Good" class (matching the training label convention)
            prob_good = float(log_probs.exp()[0, 1].item())

        # Decision boundary at 0.5.  In production you might tune this
        # threshold on a validation set to optimise precision/recall trade-off.
        is_good = prob_good > 0.5

        # Build the feature dict by averaging each node feature column.
        # This is a summary, not a per-residue breakdown — for per-residue
        # interpretation you would extract individual rows of graph.x.
        node_feats = graph.x.numpy()
        feat_dict = {
            name: float(np.mean(node_feats[:, i]))
            for i, name in enumerate(_FEATURE_NAMES)
        }

        return bool(is_good), prob_good, feat_dict

    def save(self, path: str) -> None:
        """
        Save model weights and architecture config to a ``.pt`` checkpoint.

        The checkpoint is a plain Python dict serialised with torch.save().
        It contains the architecture hyperparameters alongside the weight
        tensors so the model can be reconstructed without any external config.

        Args:
            path: Destination file path (should end in .pt).
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required to save a GNN checkpoint.") from exc

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                # state_dict: OrderedDict mapping parameter names → tensors
                # This is the standard PyTorch checkpoint format.
                "state_dict": self.model.state_dict(),
                # Architecture metadata: necessary to re-instantiate the model
                # correctly on load without requiring the caller to remember
                # the original constructor arguments.
                "node_features": self.model.node_features,
                "edge_features": self.model.edge_features,
                "hidden_dim":    self.model.hidden_dim,
                "num_classes":   self.model.num_classes,
            },
            path,
        )
        self._model_path = path
        logger.info("GNN checkpoint saved to %s", path)

    def load(self, path: str) -> None:
        """
        Load model weights from a ``.pt`` checkpoint.

        The architecture is reconstructed from the metadata stored in the
        checkpoint, then the saved state_dict is loaded into the new model.
        This means the checkpoint is fully self-describing — you do not need
        to know the original hidden_dim, heads, etc.

        Args:
            path: Path to a .pt checkpoint written by GNNQualityClassifier.save().

        Raises:
            FileNotFoundError / RuntimeError: If the checkpoint is missing or
                corrupt.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required to load a GNN checkpoint.") from exc

        from .model import ProteinGNN

        try:
            # weights_only=False allows loading the full checkpoint dict
            # (metadata + weights).  Set to True if you only need weights and
            # want extra security against arbitrary code execution via pickle.
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

            # Re-create the model architecture from stored metadata
            self.model = ProteinGNN(
                node_features=checkpoint["node_features"],
                edge_features=checkpoint["edge_features"],
                hidden_dim=checkpoint["hidden_dim"],
                num_classes=checkpoint["num_classes"],
            )
            # Copy the trained weights into the fresh model skeleton
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.eval()
            self._model_path = path
            logger.info("GNN classifier loaded from %s", path)
        except Exception as exc:
            logger.error("Failed to load GNN checkpoint from %s: %s", path, exc, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_fresh_model(self) -> None:
        """
        Initialise a randomly-weighted model.

        Used when no checkpoint is available — for example, during graph-
        construction unit tests where accuracy doesn't matter, or as the
        starting point before calling train_gnn_quality_filter.py.

        A freshly-initialised model produces random log-probabilities that
        are close to log(0.5) ≈ -0.693 for both classes (random guessing),
        because the weights are initialised near zero by PyTorch's defaults.
        """
        from .model import ProteinGNN
        self.model = ProteinGNN(node_features=8, edge_features=2, hidden_dim=64, num_classes=2)
        self.model.eval()
