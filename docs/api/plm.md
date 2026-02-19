# `synth_pdb.plm` — Protein Language Model Embeddings

ESM-2 per-residue embeddings via HuggingFace Transformers.

**Install the optional dependency first:**

```bash
pip install synth-pdb[plm]
```

---

## Quick Start

```python
from synth_pdb.plm import ESM2Embedder

embedder = ESM2Embedder()   # lazy — model loads on first embed() call

# Per-residue embeddings
emb = embedder.embed("MQIFVKTLTGKTITLEVEPS")
print(emb.shape)   # (20, 320) — 20 residues × 320-dim float32

# From a biotite AtomArray
emb = embedder.embed_structure(atom_array)   # same shape as embed()

# Sequence-level cosine similarity
sim = embedder.sequence_similarity("ACDEF", "ACDEF")   # → 1.0
sim = embedder.sequence_similarity("ACDEF", "VWLYG")   # → ~0.7–0.9
```

!!! note "Lazy loading"
    `ESM2Embedder()` does nothing until you call `embed()`. This means
    `from synth_pdb.plm import ESM2Embedder` is always safe, even
    without `torch` or `transformers` installed.

---

## Using a Larger Model

All ESM-2 variants share the same API:

```python
# Default (8M params, 320-dim, ~30 MB)
embedder = ESM2Embedder()

# Better accuracy (35M params, 480-dim)
embedder = ESM2Embedder(model_name="facebook/esm2_t12_35M_UR50D")

# Near-production (150M params, 640-dim)
embedder = ESM2Embedder(model_name="facebook/esm2_t30_150M_UR50D")
```

---

## API Reference

::: synth_pdb.plm.ESM2Embedder
    options:
      show_source: false
      members:
        - embed
        - embed_structure
        - mean_embed
        - sequence_similarity
        - embedding_dim

---

## Practical Examples

### Feed into GNN as node features

```python
from synth_pdb.plm import ESM2Embedder
import numpy as np

plm = ESM2Embedder()
plm_features = plm.embed_structure(structure)   # (L, 320)

# Concatenate with your existing per-residue geometry features
node_features = np.concatenate([geometry_features, plm_features], axis=-1)
```

### Secondary structure linear probe

```python
import torch
import torch.nn as nn

plm = ESM2Embedder()
emb = torch.tensor(plm.embed("MQIFVKTLTGKTITLEVEPS"))   # (20, 320)

probe = nn.Linear(320, 3)   # 3 classes: Helix / Strand / Coil
logits = probe(emb)          # (20, 3)
probs = logits.softmax(-1)
```

### Pairwise similarity matrix over a sequence library

```python
import numpy as np

sequences = ["ACDEF", "ACDEF", "VWLYG", "RRKKK"]
plm = ESM2Embedder()
mean_embs = np.stack([plm.mean_embed(s) for s in sequences])   # (N, 320)

# Normalise rows, then dot-product → cosine similarity matrix
norms = np.linalg.norm(mean_embs, axis=1, keepdims=True)
normed = mean_embs / (norms + 1e-8)
sim_matrix = normed @ normed.T   # (N, N)
```

---

## Background

See [Protein Language Models](../science/plm.md) for the full scientific background,
model architecture diagram, and explanation of what the embedding dimensions encode.
