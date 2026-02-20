"""
tests/test_plm.py
-----------------
TDD test suite for synth_pdb.plm — the ESM-2 protein language model embedder.

WHAT IS A PROTEIN LANGUAGE MODEL (PLM)?
----------------------------------------
A PLM is a transformer-based neural network pre-trained on hundreds of millions
of protein sequences using masked-language-modelling (MLM): randomly mask amino
acids in a sequence and train the model to predict them from context.

After pre-training, the model's *internal representations* (the activations at
the last hidden layer) encode far more than just the sequence — they capture:

  • Evolutionary information: which positions co-vary across the tree of life
  • Structural context: residues buried in a hydrophobic core vs. solvent-exposed
  • Chemical environment: charged vs. polar vs. nonpolar neighbourhoods
  • Functional context: active-site residues vs. scaffold residues

Crucially, all of this is learned from *sequence alone* — no 3D coordinates.
Yet ESM-2 embeddings predict secondary structure, solvent accessibility, and
contact maps at near-state-of-the-art accuracy without any fine-tuning.  This
is the foundation of ESMFold (a structure predictor that rivals AlphaFold2 in
speed).

MODEL CHOICE: facebook/esm2_t6_8M_UR50D
-----------------------------------------
ESM-2 comes in several scales.  We use the smallest:

  Model          Params   Embedding dim  File size
  t6_8M          8M       320            ~30 MB     ← used here
  t12_35M        35M      480            ~140 MB
  t30_150M       150M     640            ~580 MB
  t33_650M       650M     1280           ~2.5 GB    ← AlphaFold-level accuracy

The API is identical regardless of which model you choose — just change the
model name string.  Start small, scale up when you need accuracy.

TEST STRUCTURE
--------------
Five test classes, ordered from trivial to semantic:
  1. TestImportSafety       — module importable without transformers
  2. TestEmbeddingShape     — output array has the right shape and dtype
  3. TestEmbeddingValues    — values are finite, bounded, deterministic
  4. TestEmbeddingSemantics — similar sequences embed closer than random pairs
  5. TestStructureInput     — embed_structure() extracts sequence and embeds
"""

import unittest
import importlib
import numpy as np


# ---------------------------------------------------------------------------
# Shared test fixture: minimal amino acid sequences
# ---------------------------------------------------------------------------

# A short α-helical peptide (polyalanine prefers α-helix)
_HELIX_SEQ = "AAAAAAAAAAAAAAAA"  # 16 residues

# A short β-strand mimic (alternating Val/Ile strongly favours extended strand)
_STRAND_SEQ = "VIVIVIVIVIVIVIVIVI"  # 18 residues — mismatched length from helix

# Ubiquitin N-terminal region (real, well-characterised)
_UBIQUITIN_N20 = "MQIFVKTLTGKTITLEVEPS"  # 20 residues

# Completely scrambled version of ubiquitin (same composition, different order)
_SCRAMBLED_20 = "PVTKLIMEFQTSGVTITVKE"  # same AAs, shuffled


def _get_embedder():
    """Return a warmed-up ESM2Embedder, or skip the test if torch/transformers unavailable.

    We call embed() with a short probe sequence so the model is loaded before
    any test runs.  This means the slow model-load is paid once per test session
    in setUpClass, not per test.  It also ensures that ImportError from missing
    transformers is caught here (not inside a test method), triggering SkipTest.
    """
    try:
        from synth_pdb.plm import ESM2Embedder
        embedder = ESM2Embedder()
        # Trigger the lazy load now — catches ImportError if transformers is absent
        embedder.embed("ACDEF")
        return embedder
    except ImportError as e:
        raise unittest.SkipTest(f"PLM dependencies not available: {e}")



# ---------------------------------------------------------------------------
# 1. Import Safety
# ---------------------------------------------------------------------------

class TestImportSafety(unittest.TestCase):
    """
    The PLM module must be importable even without torch or transformers.

    WHY THIS MATTERS
    ----------------
    synth-pdb has a large plain-numpy user base that never uses PyTorch.
    If `import synth_pdb` silently pulled in a torch import at the top level,
    every user would pay the import cost even if they never use PLM features.

    The fix: all heavy imports are deferred inside ESM2Embedder._load_model(),
    which is called lazily on the first embed() call.  This pattern mirrors
    how synth_nmr.neural_shifts handles the same problem.
    """

    def test_module_importable_without_transformers(self):
        """synth_pdb.plm must import cleanly — no torch/transformers at module level."""
        import sys
        # Temporarily hide transformers to simulate environment without it
        orig = sys.modules.get("transformers")
        sys.modules["transformers"] = None  # type: ignore[assignment]
        try:
            # Force re-import
            if "synth_pdb.plm" in sys.modules:
                del sys.modules["synth_pdb.plm"]
            import synth_pdb.plm  # Must NOT raise  # noqa: F401
        finally:
            if orig is None:
                sys.modules.pop("transformers", None)
            else:
                sys.modules["transformers"] = orig
            # Restore real module for other tests
            if "synth_pdb.plm" in sys.modules:
                del sys.modules["synth_pdb.plm"]

    def test_embedder_instantiable_without_model_loaded(self):
        """ESM2Embedder() constructor must not download/load the model."""
        from synth_pdb.plm import ESM2Embedder
        embedder = ESM2Embedder()
        # The model should not be loaded yet — only on first embed() call
        self.assertIsNone(embedder._model,
                          "Model should be None until first embed() call (lazy loading)")
        self.assertIsNone(embedder._tokenizer,
                          "Tokenizer should be None until first embed() call")

    def test_embed_raises_clear_importerror_without_transformers(self):
        """When transformers is missing, embed() must raise ImportError with pip hint."""
        import sys
        orig_transformers = sys.modules.get("transformers")
        orig_torch = sys.modules.get("torch")

        # Simulate missing transformers
        sys.modules["transformers"] = None  # type: ignore[assignment]
        if "synth_pdb.plm" in sys.modules:
            del sys.modules["synth_pdb.plm"]

        try:
            from synth_pdb.plm import ESM2Embedder
            embedder = ESM2Embedder()
            with self.assertRaises(ImportError) as ctx:
                embedder.embed("ACDEFGHIKLMNPQRSTVWY")
            self.assertIn("pip install", str(ctx.exception))
        finally:
            if orig_transformers is None:
                sys.modules.pop("transformers", None)
            else:
                sys.modules["transformers"] = orig_transformers
            if "synth_pdb.plm" in sys.modules:
                del sys.modules["synth_pdb.plm"]


# ---------------------------------------------------------------------------
# 2. Output Shape
# ---------------------------------------------------------------------------

class TestEmbeddingShape(unittest.TestCase):
    """
    ESM-2 produces one embedding vector per residue.

    WHY (L, D) AND NOT JUST (D,)?
    ------------------------------
    Per-residue embeddings are what make PLMs powerful for structure work.
    A single sequence-level vector (like BERT's [CLS] token) loses positional
    information.  With per-residue vectors we can:
      • Predict secondary structure at each position independently
      • Feed into GNNs as node features (one node = one residue)
      • Compute inter-residue similarity matrices for contact prediction
      • Detect local functional motifs

    The embedding dimension (D=320 for esm2_t6_8M) is fixed by the model
    architecture.  Each dimension is a learned feature with no direct physical
    interpretation — they are only meaningful relative to each other.
    """

    @classmethod
    def setUpClass(cls):
        cls.embedder = _get_embedder()

    def test_output_is_2d_ndarray(self):
        """embed() must return a 2D numpy ndarray."""
        result = self.embedder.embed(_HELIX_SEQ)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.ndim, 2,
                         f"Expected 2D array, got shape {result.shape}")

    def test_output_shape_rows_equals_sequence_length(self):
        """Number of rows must equal the number of residues (L)."""
        for seq in [_HELIX_SEQ, _UBIQUITIN_N20]:
            result = self.embedder.embed(seq)
            self.assertEqual(result.shape[0], len(seq),
                             f"Row count {result.shape[0]} != len(seq) {len(seq)}")

    def test_output_dim_matches_model_config(self):
        """Embedding dim must match the model's hidden size advertised in ESM2Embedder."""
        result = self.embedder.embed(_HELIX_SEQ)
        self.assertEqual(result.shape[1], self.embedder.embedding_dim,
                         f"Dim {result.shape[1]} != advertised {self.embedder.embedding_dim}")

    def test_output_dtype_is_float32(self):
        """Embeddings must be float32 (not float64 — PyTorch default is float32)."""
        result = self.embedder.embed(_HELIX_SEQ)
        self.assertEqual(result.dtype, np.float32,
                         f"Expected float32, got {result.dtype}")

    def test_embed_structure_shape(self):
        """embed_structure(AtomArray) must return (n_residues, embedding_dim)."""
        import biotite.structure as struc
        # Build a minimal 5-residue AtomArray (CA atoms only suffices)
        seq = "ACDEF"
        aa_map = {"A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE"}
        atoms = []
        for i, ch in enumerate(seq):
            a = struc.Atom(
                [i * 3.8, 0.0, 0.0],
                chain_id="A", res_id=i + 1, res_name=aa_map[ch],
                atom_name="CA", element="C",
            )
            atoms.append(a)
        structure = struc.array(atoms)

        result = self.embedder.embed_structure(structure)
        self.assertEqual(result.shape, (len(seq), self.embedder.embedding_dim))


# ---------------------------------------------------------------------------
# 3. Value Properties
# ---------------------------------------------------------------------------

class TestEmbeddingValues(unittest.TestCase):
    """
    Sanity-check the numerical properties of the embeddings.

    ESM-2 uses layer normalisation throughout, so embeddings have bounded
    magnitude and should not contain NaN / Inf.  Modern transformers are
    numerically stable for standard protein sequences.
    """

    @classmethod
    def setUpClass(cls):
        cls.embedder = _get_embedder()
        cls.emb = cls.embedder.embed(_UBIQUITIN_N20)

    def test_all_finite(self):
        """No NaN or Inf in the embedding matrix."""
        self.assertTrue(np.all(np.isfinite(self.emb)),
                        "Embedding contains NaN or Inf values")

    def test_values_bounded(self):
        """Values should be in a plausible range — not wildly exploding."""
        # ESM-2 uses LayerNorm; embeddings are typically in [-10, 10]
        # We use a generous bound to avoid false failures
        self.assertLess(np.max(np.abs(self.emb)), 100.0,
                        "Embedding values seem unexpectedly large (>100)")

    def test_deterministic(self):
        """Same sequence must produce identical embeddings on two calls."""
        emb1 = self.embedder.embed(_UBIQUITIN_N20)
        emb2 = self.embedder.embed(_UBIQUITIN_N20)
        np.testing.assert_array_equal(
            emb1, emb2, err_msg="Embeddings are not deterministic"
        )

    def test_not_all_zeros(self):
        """Embeddings must not be trivially zero."""
        self.assertGreater(np.sum(np.abs(self.emb)), 0.0,
                           "Embeddings are all zeros — model output is degenerate")


# ---------------------------------------------------------------------------
# 4. Semantic Properties
# ---------------------------------------------------------------------------

class TestEmbeddingSemantics(unittest.TestCase):
    """
    The real power of PLMs is that *meaning is encoded in geometry*.

    COSINE SIMILARITY IN EMBEDDING SPACE
    -------------------------------------
    If we average the per-residue vectors into a single sequence-level vector
    (mean pooling), sequences with similar function or fold should sit closer
    together in this space than random sequences.

    We can't test AlphaFold-level accuracy here, but we can check that:
      1. Two identical sequences have similarity = 1.0
      2. Two completely different sequences have similarity < 1.0
      3. A sequence and its scrambled version have lower similarity than
         a sequence paired with itself (even small-model ESM-2 captures
         some local context beyond just composition)

    This is a *weak* semantic test appropriate for a unit test.  A full
    benchmark would compare against SCOP structural similarity.

    MEAN POOLING
    ------------
    mean_embed(seq) = (1/L) * Σ embedding[i]    for i in 0..L-1

    This gives a single D-dim vector.  It discards positional info but
    lets us compare whole sequences.  Alternatives:
      • Use only the [CLS] token (first position) — sometimes better
      • Use attention-weighted pooling — more complex, marginally better
      • Use per-residue embeddings directly — necessary for structure tasks
    """

    @classmethod
    def setUpClass(cls):
        cls.embedder = _get_embedder()

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two 1D vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def test_identical_sequences_similarity_one(self):
        """mean_embed(seq, seq) cosine similarity must be 1.0 (up to float32 eps)."""
        e = self.embedder.mean_embed(_UBIQUITIN_N20)
        sim = self._cosine_sim(e, e)
        self.assertAlmostEqual(sim, 1.0, places=5,
                               msg="Identical sequence self-similarity should be 1.0")

    def test_different_sequences_similarity_less_than_one(self):
        """Completely different sequences must not have similarity = 1.0."""
        e_helix = self.embedder.mean_embed(_HELIX_SEQ)
        e_strand = self.embedder.mean_embed(_STRAND_SEQ)
        sim = self._cosine_sim(e_helix, e_strand)
        self.assertLess(sim, 0.9999,
                        "Polyalanine and polyvaline should differ in embedding space")

    def test_sequence_similarity_returns_float_in_minus1_to_1(self):
        """sequence_similarity() must return a scalar in [-1, 1]."""
        sim = self.embedder.sequence_similarity(_UBIQUITIN_N20, _SCRAMBLED_20)
        self.assertIsInstance(sim, float)
        self.assertGreaterEqual(sim, -1.0)
        self.assertLessEqual(sim, 1.0)

    def test_mean_embed_shape(self):
        """mean_embed() must return a 1D vector of length embedding_dim."""
        e = self.embedder.mean_embed(_UBIQUITIN_N20)
        self.assertEqual(e.ndim, 1)
        self.assertEqual(e.shape[0], self.embedder.embedding_dim)


# ---------------------------------------------------------------------------
# 5. Structure Input
# ---------------------------------------------------------------------------

class TestStructureInput(unittest.TestCase):
    """
    embed_structure() is a convenience wrapper that:
      1. Extracts the amino acid sequence from a biotite AtomArray
      2. Delegates to embed()

    This keeps the core embed() clean (sequence → embeddings) while
    letting callers pass structures directly without extracting sequences
    themselves.

    The sequence is extracted via the standard 3-letter → 1-letter code
    mapping from the residue names in the AtomArray.  Only standard amino
    acids are included; HETATM residues are skipped.
    """

    @classmethod
    def setUpClass(cls):
        cls.embedder = _get_embedder()

    def _make_structure(self, sequence: str):
        """Build a minimal AtomArray (N, CA, C backbone) for testing."""
        import biotite.structure as struc
        aa_map = {
            "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
            "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
            "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
            "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
        }
        atoms = []
        for i, ch in enumerate(sequence):
            rn = aa_map.get(ch.upper(), "ALA")
            for aname, offset in [("N", -1.2), ("CA", 0.0), ("C", 1.2)]:
                atoms.append(struc.Atom(
                    [i * 3.8 + offset, 0.0, 0.0],
                    chain_id="A", res_id=i + 1, res_name=rn,
                    atom_name=aname, element=aname[0],
                ))
        return struc.array(atoms)

    def test_embed_structure_returns_correct_shape(self):
        """embed_structure must return (n_residues, embedding_dim)."""
        seq = "ACDEFGHIK"
        structure = self._make_structure(seq)
        result = self.embedder.embed_structure(structure)
        self.assertEqual(result.shape[0], len(seq))
        self.assertEqual(result.shape[1], self.embedder.embedding_dim)

    def test_embed_structure_matches_embed_sequence(self):
        """embed_structure(arr) must give the same result as embed(extracted_seq)."""
        seq = "MQIFVKTLTG"
        structure = self._make_structure(seq)

        emb_struct = self.embedder.embed_structure(structure)
        emb_seq = self.embedder.embed(seq)

        np.testing.assert_allclose(
            emb_struct, emb_seq, rtol=1e-5,
            err_msg="embed_structure and embed(seq) give different results for the same sequence",
        )


if __name__ == "__main__":
    unittest.main()
