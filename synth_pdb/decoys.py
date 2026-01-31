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
        forcefield: str = 'amber14-all.xml',
        hard_mode: bool = False,
        template_sequence: str = None,
        shuffle_sequence: bool = False,
        drift: float = 0.0,
        seed: int = None
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

        EDUCATIONAL NOTE - "Hard Decoys" for AI Models:
        ----------------------------------------------
        In AI protein scoring (e.g., training AlphaFold or Rosetta), "Hard Decoys" 
        are negative samples that look physically realistic but are biologically 
        impossible or topologically incorrect.
        
        Why they matter:
        *   **Contrastive Learning**: AI models need to learn why one structure is 
            "better" than another. Simple random noise (Soft Decoys) is too easy 
            for modern models to distinguish. 
        *   **The Mismatch Problem**: Hard decoys test if a model can detect:
            - **Threading Mismatch**: A sequence forced into a fold it shouldn't adopt.
            - **Label Shuffling**: A valid backbone where the sidechain identities 
              don't match the chemical environment.
            - **Near-Native Noise**: Structures that are *almost* right but have 
              subtle torsion errors (Drift).
        
        These decoys help AI avoid "memorizing" what a protein looks like and 
        force it to learn the underlying physics of fold-sequence compatibility.
        
        Args:
            sequence: Amino acid sequence.
            n_decoys: Number of decoys to generate.
            out_dir: Output directory.
            rmsd_min: Minimum RMSD from reference.
            rmsd_max: Maximum RMSD from reference.
            minimize: Energy minimization flag.
            forcefield: Forcefield for minimization.
            hard_mode: Enable hard decoy mechanisms.
            template_sequence: Sequence to use for backbone folding (threading).
            shuffle_sequence: If True, shuffles the final residue labels.
            drift: Torsion angle perturbation in degrees.
            seed: Random seed for reproducibility.
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
                # 1. Determine local generation parameters
                gen_sequence = sequence
                phi_list, psi_list, omega_list = None, None, None
                
                # Handling Threading (Hard Decoy)
                if hard_mode and template_sequence:
                    # Generate a template-fold structure first
                    template_pdb = generate_pdb_content(
                        sequence_str=template_sequence,
                        conformation='random',
                        seed=attempts + (seed if seed else 0)
                    )
                    phi_list, psi_list, omega_list = self._extract_backbone_dihedrals(template_pdb)
                    # We thread 'sequence' on this fold
                    gen_sequence = sequence
                
                pdb_content = generate_pdb_content(
                    sequence_str=gen_sequence,
                    conformation='random',
                    optimize_sidechains=optimize,
                    minimize_energy=minimize,
                    forcefield=forcefield,
                    seed=attempts + (seed if seed else 0),
                    drift=drift,
                    phi_list=phi_list,
                    psi_list=psi_list,
                    omega_list=omega_list
                )

                # Handling Shuffling (Hard Decoy)
                if hard_mode and shuffle_sequence:
                    pdb_content = self._shuffle_pdb_sequence(pdb_content)
                
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

    def _extract_backbone_dihedrals(self, pdb_content: str):
        """Extracts phi, psi, omega lists from PDB content."""
        pdb_file = pdb.PDBFile.read(io.StringIO(pdb_content))
        structure = pdb_file.get_structure(model=1)
        # Handle cases where structure might be multiple chains
        if len(struc.get_chains(structure)) > 1:
            structure = structure[structure.chain_id == struc.get_chains(structure)[0]]
            
        phi, psi, omega = struc.dihedral_backbone(structure)
        # Convert to degrees and lists
        # Biotite returns radians or already degrees? Checks docs... usually radians.
        # dihedral_backbone returns radians.
        return np.rad2deg(phi).tolist(), np.rad2deg(psi).tolist(), np.rad2deg(omega).tolist()

    def _shuffle_pdb_sequence(self, pdb_content: str):
        """Shuffles residue names in PDB content while keeping backbone intact."""
        pdb_file = pdb.PDBFile.read(io.StringIO(pdb_content))
        structure = pdb_file.get_structure(model=1)
        
        res_ids = np.unique(structure.res_id)
        if len(res_ids) == 0:
            return pdb_content
            
        res_names = []
        for rid in res_ids:
             res_names.append(structure[structure.res_id == rid].res_name[0])
        
        # Shuffle labels
        shuffled_names = res_names.copy()
        import random
        random.shuffle(shuffled_names)
        
        # Map back
        name_map = dict(zip(res_ids, shuffled_names))
        for i in range(len(structure)):
             structure.res_name[i] = name_map[structure.res_id[i]]
             
        out_f = pdb.PDBFile()
        out_f.set_structure(structure)
        out = io.StringIO()
        out_f.write(out)
        return out.getvalue()
