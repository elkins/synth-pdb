# Protein Language Models (PLMs)

## What is a Protein Language Model?

A **protein language model (PLM)** is a transformer neural network pre-trained on hundreds of millions of protein sequences. The training task — masked language modelling (MLM) — is simple: randomly mask amino acids and train the network to predict the missing ones from the surrounding context.

$$P(\text{residue}_i \mid \text{sequence}_{1..L \setminus i})$$

After pre-training, the *internal activations* at the last hidden layer form **per-residue embedding vectors**. These representations encode far more than just the amino acid identity — they capture:

| Signal encoded | How it arises |
|---|---|
| **Evolutionary conservation** | Co-evolving positions across millions of homologous sequences |
| **Structural context** | Buried residues have different contexts than exposed ones |
| **Chemical environment** | Charged, polar, hydrophobic neighbourhood patterns |
| **Functional signals** | Active site residues co-vary with catalytic residues |

Crucially, **all of this is learned from sequence alone** — no 3D coordinates are used during training.

---

## ESM-2: The Model Used in synth-pdb

[ESM-2](https://www.science.org/doi/10.1126/science.ade2574) (Evolutionary Scale Modelling, v2) is a family of transformer models from Meta AI, trained on UniRef50 (≈250 million non-redundant protein sequences).

### Architecture

```
Input: "MQIFVKTLTG..."
       ↓
Tokenisation: each AA → integer token  +  [CLS] prefix, [EOS] suffix
       ↓
Token embedding lookup: (L+2, D)
       ↓
Rotary Position Embedding (RoPE)   ←  encodes relative position, not absolute
       ↓
N × Transformer encoder layers:
    Multi-Head Self-Attention
    LayerNorm + residual
    Feed-Forward Network + LayerNorm + residual
       ↓
Last hidden state: (L+2, D)
Slice off [CLS] and [EOS]: → (L, D)   ←  one vector per amino acid
```

### Model Variants

synth-pdb defaults to the smallest variant, ideal for education and fast experiments:

| Model | Params | Embed dim | Download | Best for |
|---|---|---|---|---|
| `esm2_t6_8M_UR50D` | 8M | 320 | ~30 MB | **Education**, fast CPU inference ← default |
| `esm2_t12_35M_UR50D` | 35M | 480 | ~140 MB | Better structure prediction |
| `esm2_t30_150M_UR50D` | 150M | 640 | ~580 MB | Near-production quality |
| `esm2_t33_650M_UR50D` | 650M | 1280 | ~2.5 GB | AlphaFold-rivalling accuracy |

!!! tip "Upgrading models"
    All variants use exactly the same API. To use a more powerful model, just change the `model_name` argument:
    ```python
    embedder = ESM2Embedder(model_name="facebook/esm2_t12_35M_UR50D")
    ```
    Weights are cached after the first download at `~/.cache/huggingface/`.

---

## Interpreting the Embedding Space

### Per-Residue Embeddings: `(L, D)` Matrix

Each row is a D-dimensional vector for one residue. The 320 dimensions have **no direct physical interpretation** — each is a learned feature. But in aggregate they encode rich structural information:

```
Residue 1:  [0.21, -1.4, 0.83, ... ]  ← encodes "Met in position 1, near hydrophobic core"
Residue 2:  [0.10, -0.9, 0.55, ... ]  ← encodes "Gln, solvent-exposed, next to Met"
...
```

Two residues with similar contexts across the protein universe will have similar vectors, regardless of which protein they belong to.

### Mean Pooling: `(D,)` Vector

Averaging the per-residue matrix collapses it to a single sequence-level vector:

$$\text{mean\_embed}(\text{seq}) = \frac{1}{L} \sum_{i=1}^{L} \text{embed}(\text{seq})_i$$

This loses positional specificity but enables **fast sequence comparison** using cosine similarity.

### Cosine Similarity

$$\text{sim}(A, B) = \frac{\text{mean\_embed}(A) \cdot \text{mean\_embed}(B)}{|\text{mean\_embed}(A)| \cdot |\text{mean\_embed}(B)|}$$

Cosine similarity is **magnitude-invariant**: longer proteins produce higher-norm embeddings simply because there are more residues, not because they are more "similar". Cosine corrects for this by measuring the *angle* between vectors, not their length.

---

## Downstream Uses

### 1. GNN Node Features (Structure Quality)

The per-residue embedding matrix `(L, 320)` can directly enrich the existing GNN quality scorer with evolutionary and chemical context that geometry features alone cannot capture:

```python
from synth_pdb.plm import ESM2Embedder

plm = ESM2Embedder()
plm_features = plm.embed_structure(atom_array)   # (L, 320)
# Concatenate with existing geometric features per node
```

### 2. Secondary Structure Prediction

A linear probe on top of each residue embedding is sufficient for near-state-of-the-art SS prediction — demonstrating that structural information is already encoded in the sequence model:

```python
import torch.nn as nn

probe = nn.Linear(320, 3)   # 3 classes: H, E, C
logits = probe(torch.tensor(embeddings))   # (L, 3)
```

### 3. Contact Map Prediction

Outer product of per-residue embeddings gives a pairwise representation:

```
emb_i:  (L, D)
emb_j:  (L, D)
outer:  (L, L, 2D)    →  CNN →  (L, L) contact probability
```

This is the core idea behind the EvoFormer in AlphaFold 2.

### 4. Sequence Similarity Search

```python
sim = plm.sequence_similarity(seq_query, seq_candidate)
# 0.95+ → likely same fold family
# 0.80–0.95 → similar function, possibly different fold
# <0.70 → potentially unrelated
```

---

## Reference

Lin, Z. et al. (2023) "Evolutionary-scale prediction of atomic-level protein structure with a language model." *Science* 379, 1123–1130. [doi:10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574)
