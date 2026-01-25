import logging
import numpy as np
import biotite.structure as struc
from .generator import generate_pdb_content
from .packing import optimize_sidechains as run_optimization
from .physics import EnergyMinimizer
import biotite.structure.io.pdb as pdb
import io

logger = logging.getLogger(__name__)

class DecoyGenerator:
    """
    Generates ensembles of protein structures (decoys) with specific properties.
    """
    
    def __init__(self):
        pass

    def generate_ensemble(
        self,
        sequence: str,
        n_decoys: int,
        out_dir: str,
        rmsd_min: float = 0.0,
        rmsd_max: float = 999.0,
        optimize: bool = False,
        minimize: bool = False,
        forcefield: str = 'amber14-all.xml'
    ):
        """
        Generates N unique decoys for a given sequence within a target RMSD range.
        
        EDUCATIONAL NOTE - RMSD (Root Mean Square Deviation):
        -----------------------------------------------------
        RMSD is the standard metric for comparing two protein structures.
        It measures the average distance between corresponding atoms (usually C-alpha)
        after the structures have been optimally superimposed (rotated/translated).
        
        Formula: RMSD = sqrt( sum(dist_i^2) / N )

        Interpretation:
        *   RMSD < 2.0 A: Very similar structures (within experimental error).
        *   RMSD 2-5 A:   Similar fold, but distinct local conformation.
        *   RMSD > 5 A:   Different folds or very unfolded states.
        
        RMSD is calculated against the first generated decoy (the 'reference').
        
        EDUCATIONAL NOTE - "Decoys" vs "NMR Ensembles":
        -----------------------------------------------
        *   **NMR Ensemble**: A set of structures that all satisfy experimental restraints (NOEs)
            and converge to a similar fold (low RMSD). They represent PRECISION.
        *   **Decoys**: Independent random conformations generated to sample the conformational space.
            They often have high RMSD (diversity) and represent the SEARCH SPACE.
            
        This generator produces *Decoys* (independent samples). To mimic an NMR ensemble,
        one would need to filter these by RMSD or use simulated annealing refinement.
        
        Args:
            sequence: Amino acid sequence.
            n_decoys: Number of decoys to generate.
            out_dir: Output directory.
            rmsd_min: Minimum RMSD from reference.
            rmsd_max: Maximum RMSD from reference.
            optimize: Side-chain optimization flag.
            minimize: Energy minimization flag.
            forcefield: Forcefield for minimization.
        """
        import os
        os.makedirs(out_dir, exist_ok=True)
        
        generated_decoys = []
        reference_structure = None
        
        attempts = 0
        max_attempts = n_decoys * 10 # Avoid infinite loop
        
        logger.info(f"Generating {n_decoys} decoys for sequence length {len(sequence)}...")
        
        while len(generated_decoys) < n_decoys and attempts < max_attempts:
            attempts += 1
            
            # Generate candidate
            # We vary conformation parameters randomly to get diversity
            # Ideally generator should support 'random' seeding or 'random' conformation
            # Currently 'random' conformation does random sampling of Ramachandran
            try:
                pdb_content = generate_pdb_content(
                    sequence_str=sequence,
                    conformation='random', # Force random for diversity
                    optimize_sidechains=optimize,
                    minimize_energy=minimize,
                    forcefield=forcefield
                )
                
                # Parse to AtomArray for RMSD calc
                pdb_file = pdb.PDBFile.read(io.StringIO(pdb_content))
                current_structure = pdb_file.get_structure(model=1)
                
                # Biopython/Biotite RMSD usually needs CA atoms
                current_ca = current_structure[current_structure.atom_name == "CA"]
                
                if reference_structure is None:
                    # First structure is reference
                    reference_structure = current_structure
                    reference_ca = current_ca
                    
                    # Accept it
                    filename = os.path.join(out_dir, f"decoy_0.pdb")
                    with open(filename, 'w') as f:
                        f.write(pdb_content)
                    generated_decoys.append(filename)
                    logger.info(f"Decoy 0 (Reference) generated.")
                    
                else:
                    # Calculate RMSD against reference
                    # Superimpose first to minimize RMSD
                    # Returns (fitted, transform)
                    fitted_ca, _ = struc.superimpose(reference_ca, current_ca)
                    rmsd = struc.rmsd(reference_ca, fitted_ca)
                    
                    if rmsd_min <= rmsd <= rmsd_max:
                        idx = len(generated_decoys)
                        filename = os.path.join(out_dir, f"decoy_{idx}.pdb")
                        with open(filename, 'w') as f:
                            f.write(pdb_content)
                        generated_decoys.append(filename)
                        logger.info(f"Decoy {idx} accepted (RMSD={rmsd:.2f}A).")
                    else:
                        logger.debug(f"Decoy rejected (RMSD={rmsd:.2f}A outside {rmsd_min}-{rmsd_max}).")
                        
            except Exception as e:
                logger.warning(f"Decoy generation failed attempt {attempts}: {e}")
                continue
                
        logger.info(f"Finished. Generated {len(generated_decoys)} decoys.")
        return generated_decoys
