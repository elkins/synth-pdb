
import os
import csv
import random
import logging
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import concurrent.futures

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
from .data import STANDARD_AMINO_ACIDS, ONE_TO_THREE_LETTER_CODE

import biotite.structure.io.pdb as pdb
import io

logger = logging.getLogger(__name__)

def _generate_single_sample_task(args):
    """
    Helper function to generate a single sample.
    Arguments are passed as a tuple to be compatible with map/submit if needed,
    but we'll use unpacking for clarity.
    """
    sample_id, length, conf_type, split, output_dir = args
    
    save_dir = Path(output_dir) / split
    pdb_save_path = save_dir / f"{sample_id}.pdb"
    cmap_save_path = save_dir / f"{sample_id}.casp"
    
    try:
        # 1. Generate Structure
        # optimize_sidechains=False for speed in bulk generation
        pdb_content = generate_pdb_content(
            length=length,
            conformation=conf_type,
            optimize_sidechains=False
        )
        
        # 2. Calculate Contact Map
        pdb_file = pdb.PDBFile.read(io.StringIO(pdb_content))
        structure = pdb_file.get_structure(model=1)
        
        # educational_note:
        # -----------------
        # Why Distance Matrices instead of Binary Contact Maps?
        #
        # Binary Maps (0/1) are common for "Contact Prediction" (a classification task). 
        # However, they discard the detailed geometry of the protein. Modern 
        # Structure Prediction models (like AlphaFold) use "Distograms" (weighted 
        # distance binned distributions) or raw distances to learn a continuous 
        # representation of the energy landscape.
        #
        # By passing power=None here, we generate raw distances. This allows the 
        # CASP exporter to report the exact ground-truth distance for every pair 
        # within the threshold, rather than just marking them all as "8.0".
        cmap = compute_contact_map(structure, threshold=8.0)
        
        # Get sequence for export header
        three_to_one = {v: k for k, v in ONE_TO_THREE_LETTER_CODE.items()}
        # Filter for CA to get one per residue
        ca = structure[structure.atom_name == "CA"]
        seq_str = "".join([three_to_one.get(res, 'X') for res in ca.res_name])
        
        # 3. Save Files
        with open(pdb_save_path, "w") as out:
            out.write(pdb_content)
            
        cmap_content = export_constraints(cmap, seq_str, fmt="casp", threshold=8.0)
        with open(cmap_save_path, "w") as out:
            out.write(cmap_content)
            
        # Return success and metadata for manifest
        return {
            "success": True,
            "sample_id": sample_id,
            "length": length,
            "conformation": conf_type,
            "split": split,
            "pdb_path": str(pdb_save_path.relative_to(output_dir)),
            "cmap_path": str(cmap_save_path.relative_to(output_dir))
        }

    except Exception as e:
        return {
            "success": False,
            "sample_id": sample_id,
            "error": str(e)
        }

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
        seed: Optional[int] = None,
        max_workers: Optional[int] = None
    ):
        self.output_dir = Path(output_dir).absolute()
        self.num_samples = num_samples
        self.min_length = min_length
        self.max_length = max_length
        self.train_ratio = train_ratio
        self.max_workers = max_workers
        
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
        """Run the generation loop using multiprocessing."""
        import multiprocessing
        
        # Determine CPUs
        if self.max_workers is None:
            self.max_workers = max(1, multiprocessing.cpu_count() - 1)
            
        logger.info(f"Starting bulk generation of {self.num_samples} samples using {self.max_workers} cores...")
        self.prepare_directories()
        
        manifest_path = self.output_dir / "dataset_manifest.csv"
        
        # Prepare Tasks
        tasks = []
        for i in range(self.num_samples):
            sample_id = f"synth_{i:06d}"
            
            # 1. Randomize Parameters (in main process for determinism with seed)
            length = random.randint(self.min_length, self.max_length)
            
            # weighted choice for conformation complexity
            # Note: 'mixed' usually requires structure string. For simplicity we stick to basic or mixed logic.
            # If we pick 'mixed', we construct a structure string.
            conf_type = random.choices(
                ["alpha", "beta", "random", "ppii", "extended"],
                weights=[0.3, 0.3, 0.3, 0.05, 0.05]
            )[0]
            
            is_train = random.random() < self.train_ratio
            split = "train" if is_train else "test"
            
            tasks.append((sample_id, length, conf_type, split, str(self.output_dir)))

        # Execute
        completed_count = 0
        # Open manifest for appending
        # In production, we might want to batch this, but line-by-line is safer for interruptions
        with open(manifest_path, "a", newline="") as f:
            writer = csv.writer(f)
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_id = {executor.submit(_generate_single_sample_task, task): task[0] for task in tasks}
                
                for future in concurrent.futures.as_completed(future_to_id):
                    sample_id = future_to_id[future]
                    try:
                        result = future.result()
                        if result["success"]:
                            writer.writerow([
                                result["sample_id"],
                                result["length"],
                                result["conformation"],
                                result["split"],
                                result["pdb_path"],
                                result["cmap_path"]
                            ])
                            completed_count += 1
                        else:
                            logger.error(f"Failed to generate {sample_id}: {result.get('error')}")
                            
                        # Logging progress
                        if completed_count % 100 == 0:
                            logger.info(f"Progress: {completed_count}/{self.num_samples} ({completed_count/self.num_samples*100:.1f}%)")
                            
                    except Exception as exc:
                        logger.error(f"Generate task generated an exception: {exc}")
                    
        logger.info(f"Bulk generation complete. Generated {completed_count}/{self.num_samples} samples.")
