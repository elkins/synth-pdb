
import os
import csv
import random
import logging
from pathlib import Path
from typing import Optional
import numpy as np

# Import internal dependencies
# We use generator for PDB content
from .generator import generate_pdb_content
# We use export for contact maps, but we need the contact map DATA first.
# Contact map calculation is in contact_map.py? No, it's usually part of pdbstat or internal.
# Wait, `export.py` takes a numpy array. We need to CALCULATE it first.
# Checking synth_pdb/contact.py or similar.
# Ah, `synth_pdb/contact.py` likely has `calculate_contact_map`.
from .contact import compute_contact_map
from .export import export_constraints
from .data import STANDARD_AMINO_ACIDS

import biotite.structure.io.pdb as pdb
import io

logger = logging.getLogger(__name__)

class DatasetGenerator:
    """
    Generates large-scale synthetic protein datasets for AI model training.
    
    educational_note:
    -----------------
    AI models for protein folding (like AlphaFold, RoseTTAFold) require massive datasets 
    of (Structure, Sequence) pairs to learn the patterns of protein physics.
    Real PDB data is limited (~200k structures). Synthetic data allows us to:
    1. Augment training data with unlimited diversity.
    2. Balance the dataset (e.g., more examples of rare secondary structures).
    3. Create "uncurated" datasets to test model robustness.
    
    This generator produces:
    - PDB files (coordinates)
    - Contact Maps (distance constraints)
    - Metadata Manifest (CSV)
    """
    
    def __init__(
        self, 
        output_dir: str, 
        num_samples: int = 100,
        min_length: int = 10,
        max_length: int = 50,
        train_ratio: float = 0.8,
        seed: Optional[int] = None
    ):
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.min_length = min_length
        self.max_length = max_length
        self.train_ratio = train_ratio
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
    def prepare_directories(self):
        """Create the directory structure for the dataset."""
        train_dir = self.output_dir / "train"
        test_dir = self.output_dir / "test"
        
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize manifest if it doesn't exist
        manifest_path = self.output_dir / "dataset_manifest.csv"
        if not manifest_path.exists():
            with open(manifest_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "length", "conformation", "split", "pdb_path", "cmap_path"])

    def generate(self):
        """Run the generation loop."""
        logger.info(f"Starting bulk generation of {self.num_samples} samples...")
        self.prepare_directories()
        
        manifest_path = self.output_dir / "dataset_manifest.csv"
        
        # Open manifest for appending
        # In production, we might want to batch this, but line-by-line is safer for interruptions
        with open(manifest_path, "a", newline="") as f:
            writer = csv.writer(f)
            
            for i in range(self.num_samples):
                sample_id = f"synth_{i:06d}"
                
                # 1. Randomize Parameters
                length = random.randint(self.min_length, self.max_length)
                
                # weighted choice for conformation complexity
                conformations = ["alpha", "beta", "random", "mixed"] 
                # Note: 'mixed' usually requires structure string. For simplicity we stick to basic or mixed logic.
                # If we pick 'mixed', we construct a structure string.
                conf_type = random.choices(
                    ["alpha", "beta", "random", "ppii", "extended"],
                    weights=[0.3, 0.3, 0.3, 0.05, 0.05]
                )[0]
                
                # Determine split
                is_train = random.random() < self.train_ratio
                split = "train" if is_train else "test"
                save_dir = self.output_dir / split
                
                try:
                    # 2. Generate Structure
                    # We pass optimize_sidechains=False for speed usually, or maybe True?
                    # For Bulk, speed is key. Let's do False unless requested.
                    pdb_content = generate_pdb_content(
                        length=length,
                        conformation=conf_type,
                        optimize_sidechains=False
                    )
                    
                    # 3. Calculate Contact Map
                    # We need to parse PDB content to AtomArray to calculate map
                    pdb_file = pdb.PDBFile.read(io.StringIO(pdb_content))
                    structure = pdb_file.get_structure(model=1)
                    
                    # Calculate map (threshold 8.0A standard)
                    # compute_contact_map requires power=0 for binary output suitable for CASP?
                    # export_constraints takes raw numeric or binary. CASP usually wants probabilities or distances.
                    # Standard contact map usually binary 0/1. export.py lines 37-38 check `val > 0.0`.
                    # Let's use power=0 to get clean binary contacts.
                    cmap = compute_contact_map(structure, threshold=8.0, power=0)
                    
                    # Get sequence for export header
                    # Biotite structure.res_name has 3-letter codes
                    from .data import ONE_TO_THREE_LETTER_CODE
                    three_to_one = {v: k for k, v in ONE_TO_THREE_LETTER_CODE.items()}
                    # Filter for CA to get one per residue
                    ca = structure[structure.atom_name == "CA"]
                    seq_str = "".join([three_to_one.get(res, 'X') for res in ca.res_name])
                    
                    # 4. Save Files
                    pdb_save_path = save_dir / f"{sample_id}.pdb"
                    cmap_save_path = save_dir / f"{sample_id}.casp"
                    
                    with open(pdb_save_path, "w") as out:
                        out.write(pdb_content)
                        
                    cmap_content = export_constraints(cmap, seq_str, fmt="casp")
                    with open(cmap_save_path, "w") as out:
                        out.write(cmap_content)
                        
                    # 5. Log to Manifest
                    writer.writerow([
                        sample_id,
                        length,
                        conf_type,
                        split,
                        str(pdb_save_path.relative_to(self.output_dir)),
                        str(cmap_save_path.relative_to(self.output_dir))
                    ])
                    
                    if (i+1) % 100 == 0:
                        logger.info(f"Generated {i+1}/{self.num_samples} samples.")
                        
                except Exception as e:
                    logger.error(f"Failed to generate sample {sample_id}: {e}")
                    continue
                    
        logger.info("Bulk generation complete.")
