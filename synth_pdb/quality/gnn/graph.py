"""
synth_pdb.quality.gnn.graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Converts a PDB string into a PyTorch Geometric `Data` graph for the GNN quality scorer.

─────────────────────────────────────────────────────────────────────────────
EDUCATIONAL BACKGROUND — Why represent a protein as a graph?
─────────────────────────────────────────────────────────────────────────────

A conventional neural network expects a fixed-size vector (e.g. scikit-learn's
Random Forest receives [n_samples, n_features]).  Proteins are NOT fixed-size: a
20-residue peptide and a 500-residue enzyme need to be handled by the same model.

Graph Neural Networks (GNNs) solve this elegantly:

    Protein → Graph G = (V, E)
    V = {residues}                 ← nodes
    E = {spatial contacts}         ← edges

The model then operates over the graph topology, regardless of size.  This is the
same abstraction used by AlphaFold 2's "triangle attention", GVP-GNN, and most
modern structure models.

─────────────────────────────────────────────────────────────────────────────
GRAPH STRUCTURE
─────────────────────────────────────────────────────────────────────────────

  Nodes  — one per residue, identified by Cα position.
  Edges  — bidirectional between every Cα pair within 8 Å of each other.
           8 Å is a biologically meaningful cutoff: it captures direct
           contacts (side-chain interactions, H-bonds) without connecting
           residues that are too distant to interact.

  Node features  (8-dimensional):
    [0] sin(φ)          Backbone torsion — periodic encoding avoids the
    [1] cos(φ)          ±180° discontinuity. Alpha helices cluster at
    [2] sin(ψ)          φ ≈ -60°, ψ ≈ -45°. Beta strands at φ ≈ -120°,
    [3] cos(ψ)          ψ ≈ +120°. The GNN learns these clusters without
                        us hard-coding them.
    [4] B-factor        Crystallographic temperature factor (flexibility).
                        High B-factor → atomic displacement → possible
                        disorder. Normalised to [0, 1].
    [5] seq_position    Normalised index (0 = N-term, 1 = C-term). Lets
                        the model distinguish terminal from buried residues.
    [6] is_N_terminus   One-hot flag — termini have different chemistry
    [7] is_C_terminus   (free NH3+ / COO-) and often higher B-factors.

  Edge features  (2-dimensional):
    [0] Cα–Cα distance  Physical distance in Å.  Nearby residues interact
                        more strongly — the GNN can learn to weight edges
                        by proximity.
    [1] seq_separation  |i − j|.  Distinguishes local contacts (i, i+3 in
                        a helix) from long-range contacts (cross-strand).
"""

import logging
import math
import io
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def build_protein_graph(pdb_content: str, ca_distance_threshold: float = 8.0):
    """
    Parse *pdb_content* and return a :class:`torch_geometric.data.Data` object
    representing the protein as a residue-level contact graph.

    ── How this fits into the GNN pipeline ─────────────────────────────────
    The pipeline is:

        PDB string
            │
            ▼
        build_protein_graph()     ← YOU ARE HERE
            │  produces a PyG Data object:
            │     data.x          — node feature matrix [N, 8]
            │     data.edge_index — edge connectivity   [2, E]
            │     data.edge_attr  — edge features       [E, 2]
            │
            ▼
        ProteinGNN.forward()      — message passing over the graph
            │
            ▼
        log-softmax scores        — [batch_size, 2]  (Bad / Good)

    ────────────────────────────────────────────────────────────────────────

    Args:
        pdb_content: PDB-format string.
        ca_distance_threshold: Maximum Cα–Cα distance (Å) for an edge to be
            created. 8 Å is a standard contact-map cutoff in structural biology.

    Returns:
        torch_geometric.data.Data with attributes:
            - ``x``          : float32 node feature matrix [N, 8]
            - ``edge_index`` : long edge index [2, E]
            - ``edge_attr``  : float32 edge features [E, 2]
            - ``num_nodes``  : int N

    Raises:
        ValueError: If fewer than 2 residues with Cα atoms are found.
        ImportError: If torch or torch_geometric are not installed.
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:
        raise ImportError(
            "torch and torch_geometric are required for the GNN quality scorer. "
            "Install with: pip install synth-pdb[gnn]"
        ) from exc

    # ------------------------------------------------------------------
    # Step 1 — Parse PDB to extract per-residue structural data
    # ------------------------------------------------------------------
    # We read raw ATOM records rather than using a heavy library so that
    # this module has only numpy as a dependency (PyG handles the tensor
    # construction below).
    residues = _parse_backbone(pdb_content)

    if len(residues) < 2:
        raise ValueError(
            f"Found only {len(residues)} Cα atom(s) in PDB. "
            "Need at least 2 residues to build a graph."
        )

    n = len(residues)
    logger.debug("Building graph for %d residues (threshold=%.1f Å)", n, ca_distance_threshold)

    # ------------------------------------------------------------------
    # Step 2 — Build node feature matrix  x  of shape [N, 8]
    # ------------------------------------------------------------------
    # Each row describes one residue.  The features are chosen to capture:
    #   • Backbone geometry  (φ/ψ dihedrals — most discriminative signal)
    #   • Flexibility        (B-factor)
    #   • Sequence context   (position, terminus flags)
    # ------------------------------------------------------------------

    # Collect Cα coordinates — needed for the pairwise distance matrix
    ca_coords = np.array([r["ca"] for r in residues], dtype=np.float32)  # [N, 3]

    # ── B-factor normalisation ────────────────────────────────────────
    # Raw B-factors vary across structures (e.g. 5–80 Å²).  We normalise
    # to [0, 1] within each structure so the model sees relative flexibility,
    # not absolute values that depend on the refinement protocol.
    b_factors = np.array([r["b_factor"] for r in residues], dtype=np.float32)
    b_range = b_factors.max() - b_factors.min()
    b_norm = (b_factors - b_factors.min()) / (b_range if b_range > 1e-6 else 1.0)

    # ── Sequence position ─────────────────────────────────────────────
    # Linspace gives a clean gradient 0→1 that the GNN can use to
    # distinguish N-terminal from C-terminal residues.
    seq_pos = np.linspace(0.0, 1.0, n, dtype=np.float32)

    # ── Dihedral sin/cos encoding ─────────────────────────────────────
    # Why sin AND cos, not just the raw angle?
    # Because -180° and +180° are the SAME conformation, but numerically
    # very far apart.  Encoding as (sin θ, cos θ) makes the representation
    # continuous and circular — a point on the unit circle.
    # This is the same trick used in positional encodings for Transformers.
    node_feats = np.zeros((n, 8), dtype=np.float32)
    for i, res in enumerate(residues):
        phi, psi = res["phi"], res["psi"]
        node_feats[i, 0] = math.sin(math.radians(phi)) if phi is not None else 0.0
        node_feats[i, 1] = math.cos(math.radians(phi)) if phi is not None else 0.0
        node_feats[i, 2] = math.sin(math.radians(psi)) if psi is not None else 0.0
        node_feats[i, 3] = math.cos(math.radians(psi)) if psi is not None else 0.0
        node_feats[i, 4] = b_norm[i]
        node_feats[i, 5] = seq_pos[i]
        node_feats[i, 6] = 1.0 if i == 0 else 0.0       # N-terminus flag
        node_feats[i, 7] = 1.0 if i == n - 1 else 0.0   # C-terminus flag

    # ------------------------------------------------------------------
    # Step 3 — Build edge index and edge features
    # ------------------------------------------------------------------
    # The edge index is a [2, E] tensor where:
    #   edge_index[0] = source residue indices
    #   edge_index[1] = destination residue indices
    #
    # This COO (coordinate) sparse format is what PyTorch Geometric expects.
    # We create BIDIRECTIONAL edges: if residue i contacts j, we add both
    # (i→j) AND (j→i).  This ensures every node can aggregate information
    # from all its neighbours during message passing (see model.py).
    #
    # Why 8 Å?
    #   • Sequential Cα–Cα distance in a chain ≈ 3.8 Å
    #   • Alpha helix:   i and i+4 are ~5 Å apart → all captured
    #   • Beta strand:   cross-strand H-bond partners ≈ 4.5 Å → captured
    #   • 8 Å captures most functionally relevant contacts while keeping
    #     the graph sparse (avoids n² edges linking every residue to every
    #     other residue in long loops).
    # ------------------------------------------------------------------

    # Compute pairwise Cα distance matrix using broadcasting: O(N²) memory
    # which is fine for the short peptides synth-pdb generates (≤ 50 res).
    diff = ca_coords[:, None, :] - ca_coords[None, :, :]   # [N, N, 3]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))         # [N, N]

    src_list, dst_list, edge_attr_list = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                # No self-loops: a residue does not send messages to itself
                # (GATConv handles self-information via the node's own
                # embedding, not through an explicit self-edge)
                continue
            d = dist_matrix[i, j]
            if d < ca_distance_threshold:
                src_list.append(i)
                dst_list.append(j)
                # Edge feature vector: [distance, sequence_separation]
                # The GNN uses these to modulate attention weights —
                # e.g. close contacts and local-sequence contacts can be
                # weighted differently.
                edge_attr_list.append([d, float(abs(i - j))])

    if not src_list:
        # Safety fallback: if all Cα atoms happened to be > 8 Å apart
        # (extremely distorted structure), connect sequential neighbours so
        # the graph is never edgeless (an edgeless graph means no message
        # passing and the GNN degenerates to an MLP on average node features).
        logger.warning(
            "No edges found within %.1f Å threshold — adding sequential backbone edges.",
            ca_distance_threshold,
        )
        for i in range(n - 1):
            d = float(dist_matrix[i, i + 1])
            src_list.extend([i, i + 1])
            dst_list.extend([i + 1, i])
            edge_attr_list.extend([[d, 1.0], [d, 1.0]])

    # ------------------------------------------------------------------
    # Step 4 — Pack into a PyG Data object
    # ------------------------------------------------------------------
    # torch_geometric.data.Data is a property-graph container.  It holds
    # tensors as named attributes and knows how to batch multiple graphs
    # together (used in dataloaders during training).
    x = torch.tensor(node_feats, dtype=torch.float)
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n)


# ---------------------------------------------------------------------------
# Internal: PDB backbone parser
# ---------------------------------------------------------------------------

def _parse_backbone(pdb_content: str) -> List[Dict]:
    """
    Extract per-residue Cα coordinates, B-factors, and backbone dihedrals
    from a raw PDB string.

    Returns a list of dicts (one per residue, sorted by chain then residue
    number):
        {"ca": [x, y, z], "b_factor": float, "phi": float|None, "psi": float|None}

    ── PDB format reminder ──────────────────────────────────────────────
    Each ATOM line is fixed-width:
      cols  1– 6  Record type ("ATOM  ")
      cols 13–16  Atom name  ("CA  " for alpha carbon)
      cols 22–26  Residue sequence number
      cols 31–38  X coordinate (Å)
      cols 39–46  Y coordinate (Å)
      cols 47–54  Z coordinate (Å)
      cols 61–66  B-factor
    ─────────────────────────────────────────────────────────────────────
    """
    # Collect raw ATOM records keyed by (chain, res_num)
    atom_records: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
    b_records: Dict[Tuple[str, int], float] = {}

    for line in pdb_content.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        chain = line[21].strip() or "A"
        try:
            res_num = int(line[22:26].strip())
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            b = float(line[60:66]) if len(line) > 60 else 0.0
        except ValueError:
            continue

        key = (chain, res_num)
        if key not in atom_records:
            atom_records[key] = {}
        atom_records[key][atom_name] = np.array([x, y, z], dtype=np.float64)
        if atom_name == "CA":
            b_records[key] = b

    if not atom_records:
        return []

    # Sort residues by (chain, residue number) to ensure consistent ordering
    sorted_keys = sorted(atom_records.keys())

    # ------------------------------------------------------------------
    # Compute backbone dihedrals φ and ψ
    # ------------------------------------------------------------------
    # φ (phi) = dihedral(C_{i-1}, N_i, Cα_i, C_i)
    # ψ (psi) = dihedral(N_i, Cα_i, C_i, N_{i+1})
    #
    # These are undefined for the first (φ) and last (ψ) residues, where
    # we leave them as None → encoded as 0 in both sin/cos channels.
    # This is a soft indicator to the model that these are terminal residues
    # (which also have explicit is_N/C_terminus flags).
    residues = []
    for idx, key in enumerate(sorted_keys):
        atoms = atom_records[key]
        if "CA" not in atoms:
            continue

        phi = None
        psi = None

        if idx > 0:
            prev_key = sorted_keys[idx - 1]
            prev = atom_records.get(prev_key, {})
            if "C" in prev and "N" in atoms and "CA" in atoms and "C" in atoms:
                phi = _dihedral(prev["C"], atoms["N"], atoms["CA"], atoms["C"])

        if idx < len(sorted_keys) - 1:
            next_key = sorted_keys[idx + 1]
            nxt = atom_records.get(next_key, {})
            if "N" in atoms and "CA" in atoms and "C" in atoms and "N" in nxt:
                psi = _dihedral(atoms["N"], atoms["CA"], atoms["C"], nxt["N"])

        residues.append({
            "ca": atoms["CA"].astype(np.float32),
            "b_factor": b_records.get(key, 0.0),
            "phi": phi,
            "psi": psi,
        })

    return residues


def _dihedral(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """
    Compute the dihedral angle (degrees) defined by four 3D points.

    Algorithm — the "two normal vectors" method:
    ─────────────────────────────────────────────
    Given four points p1–p4 defining three bond vectors b1, b2, b3:

        b1 = p2 - p1
        b2 = p3 - p2  ← central bond (the rotation axis)
        b3 = p4 - p3

    The dihedral is the angle between the planes (b1,b2) and (b2,b3):
        n1 = b1 × b2   ← normal to plane 1
        n2 = b2 × b3   ← normal to plane 2
        θ  = arccos(n1 · n2)

    Sign is determined by whether n1 and b3 point in the same direction
    (the "right-hand rule" convention used in IUPAC backbone dihedrals).
    """
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    n1 = np.cross(b1, b2)   # normal to the b1-b2 plane
    n2 = np.cross(b2, b3)   # normal to the b2-b3 plane

    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm < 1e-10 or n2_norm < 1e-10:
        # Degenerate case: three collinear points → dihedral undefined
        return 0.0

    n1 = n1 / n1_norm
    n2 = n2 / n2_norm
    b2_unit = b2 / np.linalg.norm(b2)

    # Clamp to [-1, 1] to guard against floating-point drift before arccos
    cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
    angle = math.degrees(math.acos(cos_angle))

    # Determine sign: if n1 and b3 point in the same direction, the dihedral
    # is negative by the IUPAC right-hand convention.
    if np.dot(n1, b3) < 0:
        angle = -angle

    return angle
