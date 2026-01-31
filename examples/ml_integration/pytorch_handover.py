"""
ML Integration Example: PyTorch Handover
----------------------------------------
This script demonstrates how to generate a batch of thousands of protein structures
using synth_pdb and feed them directly into a PyTorch model for training or inference.

The BatchedGenerator produces contiguous NumPy arrays which are 'zero-copy' compatible
with PyTorch (meaning they share the same physical memory).
"""

import numpy as np
import torch
import torch.nn as nn
from synth_pdb.batch_generator import BatchedGenerator
import time

def demo_pytorch_integration():
    print("--- synth-pdb -> PyTorch Handover Demo ---")
    
    # 1. Generate a large batch of structures (e.g., 5000 structures of length 20)
    # We use full_atom=False for simplicity, but it works exactly the same for full-atom.
    sequence = "ALA-GLY-SER-TRP-HIS-LYS-CYS-ASP-GLU-PHE" * 2
    n_batch = 5000
    
    print(f"Generating {n_batch} structures of length {len(sequence.split('-'))}...")
    start_time = time.time()
    
    generator = BatchedGenerator(sequence, n_batch=n_batch)
    batch = generator.generate_batch(drift=2.0) # Add some 'structural noise' for the model
    
    gen_time = time.time() - start_time
    print(f"Generation Complete in {gen_time:.4f}s ({n_batch/gen_time:.1f} structures/sec)")

    # 2. Handover to PyTorch
    # torch.from_numpy() creates a tensor that shares the memory with the numpy array.
    # This is "Zero-Copy" - no data is moved in RAM.
    coords_tensor = torch.from_numpy(batch.coords).float()
    
    print(f"Handover to PyTorch Complete.")
    print(f"Tensor Shape: {coords_tensor.shape} (Batch, Atoms, XYZ)")
    print(f"Tensor Memory Contiguity: {coords_tensor.is_contiguous()}")

    # 3. Simple Biophysical "Surrogate Model"
    # Let's build a model that predicts the "Radius of Gyration" from coordinates.
    class RyGModel(nn.Module):
        def __init__(self, n_atoms):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_atoms * 3, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1) # Predict a single float (Radius of Gyration)
            )
            
        def forward(self, x):
            return self.net(x)

    model = RyGModel(batch.n_atoms)
    
    # 4. Inference Run
    # We process the entire 5000 structure batch in a single forward pass
    with torch.no_grad():
        predictions = model(coords_tensor)
        
    print(f"Model Inference Success. Output Shape: {predictions.shape}")
    print(f"Sample Prediction (Structure 0): {predictions[0].item():.4f} Å")

    # 5. Calculate "Ground Truth" (Mean Distance from Centroid)
    # This shows how easy it is to mix PyTorch and NumPy for structural analysis
    centroid = coords_tensor.mean(dim=1, keepdim=True)
    rg = torch.sqrt(((coords_tensor - centroid)**2).sum(dim=-1).mean(dim=1))
    
    print(f"Batch-wise Radius of Gyration calculated in PyTorch.")
    print(f"Mean Rg for Batch: {rg.mean().item():.2f} Å")

if __name__ == "__main__":
    try:
        demo_pytorch_integration()
    except ImportError:
        print("Error: PyTorch not found. Please install via 'pip install torch'.")
