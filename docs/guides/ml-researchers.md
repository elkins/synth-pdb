# Guide for AI/ML Researchers

synth-pdb is designed as a **high-performance data factory** for training protein AI models. This guide shows you how to leverage its unique capabilities for machine learning workflows.

## Why Use synth-pdb for ML?

### 1. **Zero-Copy Handover** 🚀

Direct NumPy → PyTorch/JAX/MLX without memory copying:

```python
from synth_pdb.batch_generator import BatchedGenerator
import torch

# Generate 1,000 structures in milliseconds
generator = BatchedGenerator("ALA-GLY-SER-LEU-VAL", n_batch=1000)
batch = generator.generate_batch(drift=5.0)

# Zero-copy PyTorch handover
coords_tensor = torch.from_numpy(batch.coords).float()
print(f"Contiguous: {coords_tensor.is_contiguous()}")  # True
print(f"Shares memory: {coords_tensor.data_ptr() == batch.coords.ctypes.data}")  # True
```

### 2. **Vectorized Generation** ⚡

20x faster than serial generation:

```python
import time

# Serial (slow)
start = time.time()
for _ in range(1000):
    gen = PeptideGenerator("ALA-GLY-SER")
    peptide = gen.generate()
serial_time = time.time() - start

# Batched (fast)
start = time.time()
batch = BatchedGenerator("ALA-GLY-SER", n_batch=1000).generate_batch()
batched_time = time.time() - start

print(f"Speedup: {serial_time / batched_time:.1f}x")  # ~20x
```

### 3. **Hard Decoys** 🎯

Generate challenging negative samples for robust training:

=== "Sequence Threading"

    Force a sequence onto the wrong backbone:

    ```bash
    # Thread Poly-Ala onto Poly-Pro backbone
    synth-pdb --mode decoys --sequence AAAAA --template-sequence PPPPP --hard
    ```

=== "Torsion Drift"

    Add controlled noise to Ramachandran angles:

    ```bash
    # Add 5° drift to all phi/psi angles
    synth-pdb --mode decoys --drift 5.0
    ```

=== "Label Shuffling"

    Generate valid structure, then shuffle residue identities:

    ```bash
    synth-pdb --mode decoys --sequence ACDEF --shuffle-sequence
    ```

### 4. **Rotation-Invariant Features** 🔄

Built-in distogram and orientogram export:

```python
from synth_pdb.distogram import calculate_distogram
from synth_pdb.orientogram import calculate_orientogram

# Distogram (NxN distance matrix)
distogram = calculate_distogram(peptide.structure)

# Orientogram (6D inter-residue orientations)
orientogram = calculate_orientogram(peptide.structure)
```

## Quick Start: Batch Generation

### Basic Batch Generation

```python
from synth_pdb.batch_generator import BatchedGenerator

# Generate 1,000 diverse structures
generator = BatchedGenerator(
    sequence="ALA-GLY-SER-LEU-VAL-ILE-MET",
    n_batch=1000,
    full_atom=False  # Backbone only for speed
)

batch = generator.generate_batch(drift=5.0)

print(f"Shape: {batch.coords.shape}")  # (1000, 7, 5, 3)
# Dimensions: (batch, residues, atoms_per_residue, xyz)
```

### PyTorch DataLoader Integration

```python
import torch
from torch.utils.data import Dataset, DataLoader

class SynthPDBDataset(Dataset):
    def __init__(self, sequences, n_samples_per_seq=100):
        self.data = []
        for seq in sequences:
            gen = BatchedGenerator(seq, n_batch=n_samples_per_seq)
            batch = gen.generate_batch(drift=3.0)
            self.data.append(torch.from_numpy(batch.coords).float())
        self.data = torch.cat(self.data, dim=0)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# Create dataset
sequences = ["ALA-GLY-SER", "LEU-VAL-ILE", "PHE-TYR-TRP"]
dataset = SynthPDBDataset(sequences, n_samples_per_seq=100)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train loop
for batch in dataloader:
    # batch shape: (32, residues, atoms, 3)
    predictions = model(batch)
    loss = criterion(predictions, targets)
    loss.backward()
```

## Advanced: Dataset Factory

Generate massive datasets for pre-training:

```bash
# Generate 10,000 structures with contact maps
synth-pdb --mode dataset \
    --dataset-format npz \
    --num-samples 10000 \
    --min-length 10 \
    --max-length 50 \
    --output ./training_data
```

Output structure:
```
training_data/
├── dataset_manifest.csv
├── train/
│   ├── synth_000001.npz  # coords, sequence, contact_map
│   ├── synth_000002.npz
│   ...
└── test/
    ├── synth_008001.npz
    ...
```

Load in PyTorch:

```python
import numpy as np
from pathlib import Path

class NPZDataset(Dataset):
    def __init__(self, data_dir):
        self.files = list(Path(data_dir).glob("*.npz"))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        return {
            'coords': torch.from_numpy(data['coords']).float(),
            'sequence': torch.from_numpy(data['sequence']).float(),
            'contact_map': torch.from_numpy(data['contact_map']).float()
        }

dataset = NPZDataset("training_data/train")
```

## Use Cases

### 1. Structure Prediction (AlphaFold-style)

Train a model to predict 3D coordinates from sequence:

```python
# Generate training data
sequences = generate_random_sequences(n=10000, length_range=(10, 50))
for seq in sequences:
    gen = PeptideGenerator(seq)
    peptide = gen.generate(conformation="random")
    # Save (sequence, coords) pairs
```

### 2. Protein Design (Inverse Folding)

Train a model to predict sequence from structure:

```python
# Generate (structure, sequence) pairs
batch = BatchedGenerator("ALA-GLY-SER", n_batch=1000).generate_batch()
# Train model: coords → sequence
```

### 3. Contact Prediction

Train a model to predict residue-residue contacts:

```python
from synth_pdb.contact import calculate_contact_map

# Generate training data
peptide = gen.generate()
coords = peptide.structure.coord
contact_map = calculate_contact_map(coords, cutoff=8.0)
# Train model: sequence → contact_map
```

### 4. Dynamics Prediction

Train a model to predict NMR relaxation rates:

```python
# Generate structure + dynamics data
synth-pdb --length 30 --gen-relax --output structure.pdb

# Parse NEF file for R1, R2, NOE values
# Train model: structure → dynamics
```

## Tutorials

Explore these interactive notebooks:

- [ML Handover Demo](../tutorials/ml_handover_demo.ipynb) - Benchmark and integration
- [Hard Decoy Challenge](../tutorials/hard_decoy_challenge.ipynb) - Negative sampling strategies
- [Dataset Factory](../tutorials/dataset_factory.ipynb) - Bulk generation workflows
- [Neural NMR Pipeline](../tutorials/neural_nmr_pipeline.ipynb) - Multi-modal training

## Framework-Specific Examples

- [JAX Handover](https://github.com/elkins/synth-pdb/blob/main/examples/ml_loading/jax_handover.ipynb)
- [PyTorch Handover](https://github.com/elkins/synth-pdb/blob/main/examples/ml_loading/pytorch_handover.ipynb)
- [MLX Handover](https://github.com/elkins/synth-pdb/blob/main/examples/ml_loading/mlx_handover.ipynb) (Apple Silicon)

## Performance Tips

!!! tip "Maximize Throughput"
    
    1. **Use `full_atom=False`** for backbone-only generation (10x faster)
    2. **Disable minimization** during training data generation
    3. **Use Numba** for 50-100x speedup on geometry calculations
    4. **Batch size**: Aim for 1000-10000 structures per batch
    5. **Multiprocessing**: Use `--mode dataset` with automatic parallelization


## Protein Language Model Embeddings (PLM) 🧬

synth-pdb includes an optional ESM-2 integration that gives every residue a
320-dimensional vector encoding **evolutionary, structural, and chemical context**
— learned from 250 million proteins, with no training required on your part.

!!! note "Installation"
    ```bash
    pip install synth-pdb[plm]
    ```

### What you get

```python
from synth_pdb.plm import ESM2Embedder

embedder = ESM2Embedder()

# Per-residue embeddings — shape (L, 320), dtype float32
emb = embedder.embed("MQIFVKTLTGKTITLEVEPS")
print(emb.shape)   # (20, 320)

# Directly from a structure
emb = embedder.embed_structure(atom_array)   # (n_residues, 320)

# Sequence-level cosine similarity (mean-pooled)
sim = embedder.sequence_similarity("ACDEF", "VWLYG")   # float in [-1, 1]
```

### Use as GNN node features

Drop-in enrichment for the quality GNN:

```python
plm = ESM2Embedder()
plm_feats = plm.embed_structure(structure)        # (L, 320)
node_feats = np.concatenate([geom_feats, plm_feats], axis=-1)
```

### Linear probe for secondary structure

A single linear layer on PLM embeddings achieves near-SOTA secondary structure prediction — no 3D coordinates needed:

```python
import torch.nn as nn

probe = nn.Linear(320, 3)   # Helix / Strand / Coil
logits = probe(torch.tensor(emb))   # (L, 3)
```

### Upgrading to a more powerful model

The API is identical regardless of model size:

```python
# 35M params, 480-dim — better accuracy, same code
embedder = ESM2Embedder(model_name="facebook/esm2_t12_35M_UR50D")
```

For the full scientific background see [Protein Language Models](../science/plm.md).
For the API reference see [api/plm](../api/plm.md).

## Next Steps

- [API Reference: batch_generator](../api/batch_generator.md)
- [API Reference: plm](../api/plm.md)
- [Science: Protein Language Models](../science/plm.md)
- [Examples: Advanced Features](../examples/advanced.md)
- [Scientific Background: Ramachandran Plots](../science/ramachandran.md)
