"""
CLI entry point for the synth-pdb tool.

This module provides the main() function that serves as the command-line interface
for generating PDB files.
"""

import sys
import logging
import argparse
import datetime
import os
from pathlib import Path

from .generator import generate_pdb_content
from .decoys import DecoyGenerator
from .docking import DockingPrep
import os
from .validator import PDBValidator
from .pdb_utils import extract_atomic_content, assemble_pdb_content, extract_header_records
from .viewer import view_structure_in_browser

# Get a logger for this module
logger = logging.getLogger(__name__)


def _build_command_string(args: argparse.Namespace) -> str:
    """Build a command string from parsed arguments for PDB header."""
    cmd_parts = ["synth-pdb"]
    if args.sequence:
        cmd_parts.append(f"--sequence {args.sequence}")
    else:
        cmd_parts.append(f"--length {args.length}")
    
    if args.plausible_frequencies:
        cmd_parts.append("--plausible-frequencies")
    if args.conformation != 'alpha':  # Only add if not default
        cmd_parts.append(f"--conformation {args.conformation}")
    if hasattr(args, 'structure') and args.structure:  # NEW: add structure if provided
        cmd_parts.append(f"--structure '{args.structure}'")
    if args.validate:
        cmd_parts.append("--validate")
    if args.guarantee_valid:
        cmd_parts.append("--guarantee-valid")
        cmd_parts.append(f"--max-attempts {args.max_attempts}")
    if args.best_of_N > 1:
        cmd_parts.append(f"--best-of-N {args.best_of_N}")
    if args.refine_clashes > 0:
        cmd_parts.append(f"--refine-clashes {args.refine_clashes}")
    if args.optimize:
        cmd_parts.append("--optimize")
    if args.minimize:
        cmd_parts.append("--minimize")
        cmd_parts.append(f"--forcefield {args.forcefield}")
    
    if args.cyclic:
        cmd_parts.append("--cyclic")
    
    # Phase 7/8/9 flags
    if args.gen_nef:
        cmd_parts.append("--gen-nef")
        cmd_parts.append(f"--noe-cutoff {args.noe_cutoff}")
        if args.nef_output:
            cmd_parts.append(f"--nef-output {args.nef_output}")
            
    if args.gen_relax:
        cmd_parts.append("--gen-relax")
        cmd_parts.append(f"--field {args.field}")
        cmd_parts.append(f"--tumbling-time {args.tumbling_time}")
        
    if args.gen_shifts:
        cmd_parts.append("--gen-shifts")
        if args.shift_output:
            cmd_parts.append(f"--shift-output {args.shift_output}")
            
    if args.output:
        cmd_parts.append(f"--output {args.output}")
    
    return " ".join(cmd_parts)


def main() -> None:
    """
    Main function to parse arguments and generate the PDB file.
    """
    parser = argparse.ArgumentParser(
        description="Generate a PDB file with a random linear amino acid sequence."
    )
    parser.add_argument(
        "--length",
        type=int,
        default=None,
        help="Length of the amino acid sequence (number of residues). Default: 10 (or inferred from --structure if provided).",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional: Output PDB filename. If not provided, a default name will be generated.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )

    parser.add_argument(
        "--sequence",
        type=str,
        help="Specify an amino acid sequence (e.g., 'AGV' or 'ALA-GLY-VAL'). Overrides random generation.",
    )
    parser.add_argument(
        "--plausible-frequencies",
        action="store_true",
        help="Use biologically plausible amino acid frequencies for random sequence generation (ignored if --sequence is provided).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation checks (bond lengths and angles, Ramachandran) on the generated PDB.",
    )
    parser.add_argument(
        "--guarantee-valid",
        action="store_true",
        help="If set, repeatedly generate PDB files until a valid one (no violations) is produced. Implies --validate. Will stop after --max-attempts if no valid PDB is found.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=100,
        help="Maximum number of regeneration attempts when --guarantee-valid is set.",
    )
    parser.add_argument(
        "--best-of-N",
        type=int,
        default=1,
        help="Generate N PDBs, validate each, and select the one with the fewest violations. Implies --validate. Overrides --guarantee-valid.",
    )
    parser.add_argument(
        "--refine-clashes",
        type=int,
        default=0,  # Default to 0, meaning no refinement
        help="Number of iterations to refine generated PDB by minimally adjusting clashing atoms. Implies --validate. Applied after --guarantee-valid or --best-of-N selection.",
    )
    parser.add_argument(
        "--conformation",
        type=str,
        default="alpha",
        choices=["alpha", "beta", "ppii", "extended", "random"],
        help="Secondary structure conformation to generate. Options: alpha (default, alpha helix), beta (beta sheet), ppii (polyproline II), extended (stretched), random (random sampling).",
    )
    parser.add_argument(
        "--metal-ions",
        type=str,
        default="auto",
        choices=["auto", "none"],
        help="Mechanism for handling metal cofactors (e.g. Zinc). 'auto' (default) scans for binding motifs and inserts ions. 'none' disables this.",
    )
    parser.add_argument(
        "--structure",
        type=str,
        default=None,
        help="Per-region conformation specification. Format: 'start-end:conformation,...'. Supports secondary structures (alpha, beta) and Turn types (typeI, typeII, typeVIII). Example: '1-10:alpha,11-14:typeII,15-20:beta'.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open generated structure in browser-based 3D viewer (uses 3Dmol.js). Interactive visualization with rotation, zoom, and style controls.",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run Monte Carlo side-chain optimization to minimize steric clashes (Advanced).",
    )
    parser.add_argument(
        "--minimize",
        action="store_true",
        help="Run physics-based energy minimization using OpenMM (Phase 2). Requires 'openmm' installed.",
    )
    parser.add_argument(
        "--cyclic",
        action="store_true",
        help="Generate a head-to-tail cyclic peptide. Implies --minimize and disables --cap-termini.",
    )
    parser.add_argument(
        "--forcefield",
        type=str,
        default="amber14-all.xml",
        help="Forcefield to use for minimization (default: amber14-all.xml).",
    )
    
    # Phase 3: Research Utilities Arguments
    # Using 'mode' argument to distinguish workflows without breaking BC (default is 'generate')
    parser.add_argument(
        "--mode",
        type=str,
        default="generate",
        choices=["generate", "decoys", "docking", "pymol", "dataset"],
        help="Operation mode: 'generate' (default) single structure, 'decoys' ensemble, 'docking' preparation (PQR), 'pymol' visualization script, 'dataset' bulk generation.",
    )
    parser.add_argument(
        "--n-decoys",
        type=int,
        default=10,
        help="Number of decoys to generate (for --mode decoys).",
    )
    parser.add_argument(
        "--rmsd-range",
        type=str,
        default="0.0-999.0",
        help="Target RMSD range in Angstroms 'min-max' (for --mode decoys).",
    )
    parser.add_argument(
        "--input-pdb",
        type=str,
        help="Input PDB file path (required for --mode docking and --mode pymol).",
    )
    parser.add_argument(
        "--input-nef",
        type=str,
        help="Input NEF file path (required for --mode pymol).",
    )
    parser.add_argument(
        "--output-pml",
        type=str,
        help="Output PyMOL script path (for --mode pymol).",
    )
    
    # Phase 7: Synthetic NMR Data (NEF)
    parser.add_argument(
        "--gen-nef",
        action="store_true",
        help="Generate synthetic NMR data (NOE restraints) in NEF format.",
    )
    parser.add_argument(
        "--noe-cutoff",
        type=float,
        default=5.0,
        help="Distance cutoff (Angstroms) for synthetic NOEs (default 5.0).",
    )
    parser.add_argument(
        "--nef-output",
        type=str,
        help="Optional: Output NEF filename.",
    )
    parser.add_argument(
        "--gen-pymol",
        action="store_true",
        help="Generate a PyMOL script (.pml) to visualize the synthetic NEF restraints on the structure.",
    )

    # Phase 8: Synthetic Relaxation Data
    parser.add_argument(
        "--gen-relax",
        action="store_true",
        help="Generate synthetic NMR relaxation data (R1, R2, NOE) in NEF format.",
    )
    parser.add_argument(
        "--field",
        type=float,
        default=600.0,
        help="Proton Larmor frequency in MHz for relaxation calculation (default 600.0).",
    )
    parser.add_argument(
        "--tumbling-time",
        type=float,
        default=10.0,
        help="Global rotational correlation time (tau_m) in nanoseconds (default 10.0).",
    )

    # Phase 9: Synthetic Chemical Shifts
    parser.add_argument(
        "--gen-shifts",
        action="store_true",
        help="Generate synthetic Chemical Sshift data (H, N, CA, CB, C) based on secondary structure.",
    )
    parser.add_argument(
        "--shift-output",
        type=str,
        help="Optional: Output NEF filename for chemical shifts.",
    )
    
    # Phase 9.5: J-Couplings
    parser.add_argument(
        "--gen-couplings",
        action="store_true",
        help="Generate synthetic 3J(HN-HA) scalar couplings based on phi angles.",
    )
    parser.add_argument(
        "--coupling-output",
        type=str,
        help="Optional: Output CSV filename for J-couplings.",
    )
    
    # Phase 10: Constraint Export
    parser.add_argument(
        "--export-constraints",
        type=str,
        help="Export contact map constraints for modeling (e.g. AlphaFold, CASP). Specify output filename.",
    )
    parser.add_argument(
        "--constraint-format",
        type=str,
        default="casp",
        choices=["casp", "csv"],
        help="Format for constraint export (default: casp). casp=RR format.",
    )
    parser.add_argument(
        "--constraint-cutoff",
        type=float,
        default=8.0,
        help="Distance cutoff (Angstroms) for binary contacts (default: 8.0).",
    )

    # Phase 11: Torsion Export
    parser.add_argument(
        "--export-torsion",
        type=str,
        help="Export backbone torsion angles (Phi, Psi, Omega) to a file. Specify output filename.",
    )
    parser.add_argument(
        "--torsion-format",
        type=str,
        choices=["csv", "json"],
        default="csv",
        help="Format for torsion angle export (default: csv).",
    )

    # Phase 12: Synthetic MSA (Evolution)
    parser.add_argument(
        "--gen-msa",
        action="store_true",
        help="Generate synthetic Multiple Sequence Alignment (MSA) via simulated evolution.",
    )
    parser.add_argument(
        "--msa-depth",
        type=int,
        default=100,
        help="Number of sequences to generate for MSA (default: 100).",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.1,
        help="Mutation rate per position per sequence (default: 0.1).",
    )
    
    # Phase 13: Distogram Export (AI Trinity #3)
    parser.add_argument(
        "--export-distogram",
        type=str,
        help="Export NxN Distance Matrix (Distogram) to a file. Specify output filename.",
    )
    parser.add_argument(
        "--distogram-format",
        type=str,
        choices=["json", "csv", "npz"],
        default="json",
        help="Format for distogram export (default: json).",
    )
    
    # Phase 14: Biophysical Realism (Capping & pH)
    parser.add_argument(
        "--cap-termini",
        action="store_true",
        help="Add N-terminal Acetyl (ACE) and C-terminal N-methylamide (NME) caps."
    )
    parser.add_argument(
        "--ph",
        type=float,
        default=7.4,
        help="pH for determining protonation states (default: 7.4). Affects Histidine (HIS/HIP/HIE)."
    )
    parser.add_argument(
        "--cis-proline-frequency",
        type=float,
        default=0.05,
        help="Probability (0.0-1.0) of Proline adopting the Cis conformation (default: 0.05).",
    )
    parser.add_argument(
        "--phosphorylation-rate",
        type=float,
        default=0.0,
        help="Probability (0.0-1.0) of Ser/Thr/Tyr phosphorylation (default: 0.0).",
    )
    
    # Phase 2: MD Equilibration
    parser.add_argument(
        "--equilibrate",
        action="store_true",
        help="Run Molecular Dynamics equilibration (at 300K) after minimization. Requires OpenMM."
    )
    
    # Phase 15: Bulk Dataset Generation (AI)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to generate for the dataset (for --mode dataset). Default: 100."
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=10,
        help="Minimum sequence length for dataset samples (for --mode dataset). Default: 10."
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=50,
        help="Maximum sequence length for dataset samples (for --mode dataset). Default: 50."
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of samples to split into training set (for --mode dataset). Default: 0.8."
    )
    parser.add_argument(
        "--md-steps",
        type=int,
        default=1000,
        help="Number of MD steps for equilibration (default: 1000 approx 2ps)."
    )
    parser.add_argument(
        "--dataset-format",
        type=str,
        default="pdb",
        choices=["pdb", "npz"],
        help="Output format for dataset generation (default: pdb). 'npz' produces compressed arrays.",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible structure generation. If set, the same command will produce the exact same PDB file.",
    )

    args = parser.parse_args()

    # Set the logging level based on user input
    log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    logging.getLogger().setLevel(log_level)
    logger.debug("Logging level set to %s.", args.log_level.upper())
    
    logger.info("Starting PDB file generation process.")
    logger.debug(
        "Parsed arguments: length=%s, output='%s', sequence='%s', plausible_frequencies=%s, validate=%s",
        args.length,
        args.output,
        args.sequence,
        args.plausible_frequencies,
        args.validate,
    )

    # If --best-of-N is set, it overrides --guarantee-valid and implies --validate.
    if args.best_of_N > 1:
        args.validate = True
        args.guarantee_valid = False # Disable guarantee-valid if best-of-N is used
        logger.info(f"--best-of-N is set to {args.best_of_N}. Generating multiple PDBs to find the one with fewest violations.")
    elif args.guarantee_valid: # Only apply if best-of-N is not active
        args.validate = True
        logger.info("--guarantee-valid is set. Will attempt to generate a valid PDB.")
    
    if args.refine_clashes > 0:
        args.validate = True # Refinement implies validation during initial generation
        logger.info(f"--refine-clashes is set to {args.refine_clashes}. Validation will be performed.")

    if args.cyclic:
        logger.info("--cyclic is set. Enabling energy minimization for ring closure and disabling terminal caps.")
        args.minimize = True
        args.cap_termini = False

    # Validate length only if no sequence is provided
    if args.sequence is None:
        if args.length is None or args.length <= 0:
            # Check if we can infer length from structure parameter
            if args.structure:
                # Parse structure to find maximum residue number
                try:
                    max_residue = 0
                    for region in args.structure.split(','):
                        region = region.strip()
                        if ':' in region:
                            range_part = region.split(':', 1)[0]
                            if '-' in range_part:
                                _, end_str = range_part.split('-', 1)
                                end = int(end_str)
                                max_residue = max(max_residue, end)
                    
                    if max_residue > 0:
                        args.length = max_residue
                        logger.info(f"Inferred length={max_residue} from --structure parameter")
                    else:
                        logger.error("Could not infer length from --structure parameter")
                        sys.exit(1)
                except Exception as e:
                    logger.error(f"Failed to parse --structure parameter: {e}")
                    sys.exit(1)
            else:
                # No structure parameter, use default length of 10
                args.length = 10
                logger.debug("Using default length=10")
        elif args.length <= 0:
            logger.error("Length must be a positive integer.")
            sys.exit(1)

    # Dispatch to specific modes if not generating a new structure
    if args.mode == "docking":
        if not args.input_pdb:
             logger.error("Docking mode requires --input-pdb.")
             sys.exit(1)
        prep = DockingPrep(args.input_pdb)
        pqr_file = prep.generate_pqr()
        logger.info(f"Docking preparation complete. PQR file: {pqr_file}")
        return

    if args.mode == "pymol":
        if not args.input_pdb or not args.input_nef or not args.output_pml:
            logger.error("PyMOL mode requires --input-pdb, --input-nef, and --output-pml.")
            sys.exit(1)
        
        from .visualization import generate_pymol_script
        from .nef_io import read_nef_restraints
        
        try:
            # Read restraints first (the function now expects a list)
            restraints = read_nef_restraints(args.input_nef)
            generate_pymol_script(args.input_pdb, restraints, args.output_pml)
            logger.info(f"PyMOL script generated successfully: {args.output_pml}")
        except Exception as e:
            logger.error(f"Failed to generate PyMOL script: {e}")
            sys.exit(1)
        return # Exit after visualization generation
    
    if args.mode == "decoys":
        if not args.sequence and not args.length:
             logger.error("Decoy generation requires --sequence or --length.")
             sys.exit(1)
        
        target_sequence = args.sequence
        if not target_sequence:
            # Generate random sequence if not provided
            import random
            from .data import ONE_TO_THREE_LETTER_CODE, AMINO_ACID_FREQUENCIES
            
            rng = random.Random(args.seed)
            three_to_one = {v: k for k, v in ONE_TO_THREE_LETTER_CODE.items()}
            
            if args.plausible_frequencies:
                # Map frequencies to 1-letter codes
                residues = list(time_to_one.values()) # No, map keys.
                # Harder. Let's sample 3-letter and convert.
                residues_3 = list(AMINO_ACID_FREQUENCIES.keys())
                weights = list(AMINO_ACID_FREQUENCIES.values())
                chosen_3 = rng.choices(residues_3, weights=weights, k=args.length)
                target_sequence = "".join([three_to_one[r] for r in chosen_3])
            else:
                residues_1 = list(ONE_TO_THREE_LETTER_CODE.keys())
                target_sequence = "".join(rng.choices(residues_1, k=args.length))
            
            logger.info(f"Generated random sequence for decoys: {target_sequence}")


        generator = DecoyGenerator()
        try:
             rmsd_parts = args.rmsd_range.split('-')
             rmsd_min = float(rmsd_parts[0])
             rmsd_max = float(rmsd_parts[1])
        except Exception:
             logger.warning(f"Invalid RMSD range '{args.rmsd_range}', using default 0-999.")
             rmsd_min, rmsd_max = 0.0, 999.0

        out_dir = args.output if args.output else "decoys"
        
        generator.generate_ensemble(
            sequence=target_sequence,
            n_decoys=args.n_decoys,
            out_dir=out_dir,
            rmsd_min=rmsd_min,
            rmsd_max=rmsd_max,
            optimize=args.optimize,
            minimize=args.minimize,
            forcefield=args.forcefield
        )
        return

    if args.mode == "dataset":
        from .dataset import DatasetGenerator
        out_dir = args.output if args.output else "dataset"
        
        logger.info(f"Starting bulk dataset generation in '{out_dir}'...")
        generator = DatasetGenerator(
            output_dir=out_dir,
            num_samples=args.num_samples,
            min_length=args.min_length,
            max_length=args.max_length,
            train_ratio=args.train_ratio,
            seed=args.seed,
            dataset_format=args.dataset_format
        )
        generator.generate()
        logger.info(f"Dataset generation complete. Output directory: {os.path.abspath(out_dir)}")
        return


    length_for_generator = args.length if args.sequence is None else None

    final_pdb_content = None
    final_violations = None
    min_violations_count = float('inf')

    generation_attempts = 1 if not args.guarantee_valid and args.best_of_N <= 1 else args.max_attempts
    if args.best_of_N > 1:
        generation_attempts = args.best_of_N

    for attempt_num in range(1, generation_attempts + 1):
        logger.info(f"Generation attempt {attempt_num}/{generation_attempts}.")
        current_pdb_content = None
        current_violations = []

        try:
            current_pdb_content = generate_pdb_content(
                length=length_for_generator,
                sequence_str=args.sequence,
                use_plausible_frequencies=args.plausible_frequencies,
                conformation=args.conformation,
                structure=args.structure,  # NEW: per-region conformation support
                optimize_sidechains=args.optimize,
                minimize_energy=args.minimize,

                forcefield=args.forcefield,
                seed=args.seed,
                ph=args.ph,
                cap_termini=args.cap_termini,
                equilibrate=args.equilibrate,
                equilibrate_steps=args.md_steps,
                metal_ions=args.metal_ions,
                cis_proline_frequency=args.cis_proline_frequency,
                phosphorylation_rate=args.phosphorylation_rate,
                cyclic=args.cyclic,
            )

            if not current_pdb_content:
                logger.warning(f"Failed to generate PDB content in attempt {attempt_num}. Skipping.")
                continue

            if args.validate:
                logger.info("Performing PDB validation checks for current generation...")
                logger.debug(f"PDB content passed to validator (attempt {attempt_num}):\n{current_pdb_content}")
                validator = PDBValidator(current_pdb_content)
                validator.validate_all()
                current_violations = validator.get_violations()
                logger.debug(f"PDBValidator returned {len(current_violations)} violations for attempt {attempt_num}. Content: {current_violations}")
            
            if args.guarantee_valid:
                if not current_violations:
                    logger.info(f"Successfully generated a valid PDB file after {attempt_num} attempts.")
                    final_pdb_content = current_pdb_content
                    final_violations = current_violations
                    break # Exit loop, valid PDB found
                else:
                    logger.warning(f"PDB generated in attempt {attempt_num} has {len(current_violations)} violations. Retrying...")
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("--- PDB Validation Report for failed attempt ---")
                        for violation in current_violations:
                            logger.debug(violation)
                        logger.debug("--- End Validation Report ---")
            elif args.best_of_N > 1:
                if len(current_violations) < min_violations_count:
                    min_violations_count = len(current_violations)
                    final_pdb_content = current_pdb_content
                    final_violations = current_violations
                    logger.info(f"Attempt {attempt_num} yielded {len(current_violations)} violations (new minimum).")
                else:
                    logger.info(f"Attempt {attempt_num} yielded {len(current_violations)} violations. Current minimum is {min_violations_count}.")
            else: # No guarantee-valid or best-of-N, just take the first one
                final_pdb_content = current_pdb_content
                final_violations = current_violations
                break

        except (ValueError, TypeError, RuntimeError, Exception) as e:
            logger.error(f"Error processing sequence during generation: {e}")
            sys.exit(1)

    # Parse structure definitions for highlighting in Viewer
    highlights = []
    if args.structure:
        try:
            # Format: "1-5:beta,6-9:typeII"
            parts = args.structure.split(",")
            for part in parts:
                if ":" in part:
                     rng, s_type = part.split(":")
                     if "-" in rng:
                         start_s, end_s = rng.split("-")
                         start, end = int(start_s), int(end_s)
                         
                         # Only highlight specific turns or interesting features
                         if "type" in s_type or "beta" in s_type and "turn" in s_type: # e.g. typeII, beta-turn
                             highlights.append({
                                 'start': start, 
                                 'end': end, 
                                 'color': 'purple', 
                                 'style': 'stick', # Stick makes turn geometry visible
                                 'label': s_type
                             })
                         elif "helix" in s_type or "alpha" in s_type:
                             highlights.append({
                                 'start': start, 
                                 'end': end, 
                                 'color': 'magenta', 
                                 'style': 'cartoon',
                                 'label': 'Alpha Helix'
                             })
        except Exception as e:
            logger.warning(f"Could not parse structure for highlighting: {e}")

    if final_pdb_content is None:
        logger.error(f"Failed to generate a suitable PDB file after {generation_attempts} attempts.")
        sys.exit(1)
    else:
        # Extract atomic content from the initially selected PDB for subsequent refinement or final assembly.
        final_pdb_atomic_content = extract_atomic_content(final_pdb_content)
        
        # PRESERVE HEADER RECORDS (SSBOND)
        # generator.py creates them, but extract_atomic_content strips them.
        # We must re-inject them during assembly.
        preserved_ssbonds = extract_header_records(final_pdb_content, "SSBOND")
        preserved_conects = extract_header_records(final_pdb_content, "CONECT")

        # Apply refinement if requested
        if args.refine_clashes > 0:
            args.validate = True # Refinement implies validation
            logger.info(f"Starting steric clash refinement for {args.refine_clashes} iterations.")
            
            # current_refined_atomic_content will hold only ATOM/TER lines
            current_refined_atomic_content = final_pdb_atomic_content
            current_refined_violations = final_violations
            initial_violations_count = len(final_violations)

            for refine_iter in range(args.refine_clashes):
                logger.info(f"Refinement iteration {refine_iter + 1}/{args.refine_clashes}. Violations: {len(current_refined_violations)}")
                if not current_refined_violations:
                    logger.info("No violations remain, stopping refinement early.")
                    break

                # Parse atoms from current atomic PDB content
                # PDBValidator._parse_pdb_atoms can work directly on atomic lines.
                parsed_atoms_for_refinement = PDBValidator._parse_pdb_atoms(current_refined_atomic_content)
                
                # Apply steric clash tweak
                modified_atoms = PDBValidator._apply_steric_clash_tweak(parsed_atoms_for_refinement)

                # Convert modified atoms back to atomic PDB content (no header/footer)
                new_atomic_content_after_tweak = PDBValidator.atoms_to_pdb_content(modified_atoms)

                # Re-validate the tweaked atomic PDB content.
                # For validation, PDBValidator expects a full PDB string.
                # Build command string for temporary header
                cmd_string = _build_command_string(args)
                temp_full_pdb = assemble_pdb_content(
                    new_atomic_content_after_tweak, 1, command_args=cmd_string,
                    extra_records=preserved_ssbonds # Keeps context valid if validator checks bonds
                )
                temp_validator = PDBValidator(pdb_content=temp_full_pdb)
                temp_validator.validate_all()
                new_violations = temp_validator.get_violations()

                if len(new_violations) < len(current_refined_violations):
                    logger.info(f"Refinement iteration {refine_iter + 1}: Reduced violations from {len(current_refined_violations)} to {len(new_violations)}.")
                    current_refined_atomic_content = new_atomic_content_after_tweak
                    current_refined_violations = new_violations
                else:
                    logger.info(f"Refinement iteration {refine_iter + 1}: No further reduction in violations ({len(new_violations)}). Stopping refinement.")
                    break # No improvement, stop refinement

            final_pdb_atomic_content = current_refined_atomic_content # This is now atomic-only
            final_violations = current_refined_violations
            if initial_violations_count > len(final_violations):
                logger.info(f"Refinement process completed. Reduced total violations from {initial_violations_count} to {len(final_violations)}.")
            elif initial_violations_count == len(final_violations):
                logger.info(f"Refinement process completed. No change in total violations ({len(final_violations)}).")
            else: # Should not happen if logic is correct, but for completeness
                logger.warning(f"Refinement process completed. Violations increased from {initial_violations_count} to {len(final_violations)}. This indicates an issue with the refinement logic.")
        # If no refinement was requested, final_pdb_atomic_content was already set from the initial extraction.
 
        # After successful generation (and optional validation)
        # Only proceed to file writing if final_pdb_atomic_content is not None


        # Final output handling
        if final_pdb_atomic_content is not None:
            # Determine the sequence length for the final header, especially if it was inferred from sequence string.
            final_sequence_length = args.length
            if args.sequence:
                final_sequence_length = len(args.sequence.replace("-", ""))
            elif args.length is None:
                # Infer length from the atomic content if not explicitly set
                # Temporarily create a PDBValidator with minimal header to get sequence length
                cmd_string = _build_command_string(args)
                temp_full_pdb_for_length_inference = assemble_pdb_content(
                    final_pdb_atomic_content, 1, command_args=cmd_string
                )
                temp_validator_for_length = PDBValidator(pdb_content=temp_full_pdb_for_length_inference)
                # Assuming a single chain 'A' for simplicity, as per current generator
                inferred_sequence = temp_validator_for_length._get_sequences_by_chain().get('A', [])
                final_sequence_length = len(inferred_sequence) if inferred_sequence else "VARIABLE"

            # Assemble the full PDB content with header and footer
            cmd_string = _build_command_string(args)
            final_full_pdb_content_to_write = assemble_pdb_content(
                final_pdb_atomic_content, final_sequence_length, command_args=cmd_string,
                extra_records=preserved_ssbonds,
                conect_records=preserved_conects
            )

            if args.output:
                output_filename = args.output
                logger.debug("Using user-provided output filename: %s", output_filename)
            else:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                if args.sequence:
                    # Use a simplified sequence string for filename to avoid very long names
                    sequence_tag = args.sequence.replace("-", "")[:10]
                    output_filename = f"custom_peptide_{sequence_tag}_{timestamp}.pdb"
                else:
                    output_filename = f"random_linear_peptide_{args.length}_{timestamp}.pdb"
                logger.debug("Generated default output filename: %s", output_filename)

            try:
                with open(output_filename, "w") as f:
                    f.write(final_full_pdb_content_to_write)
                logger.info(
                    "Successfully generated PDB file: %s", os.path.abspath(output_filename)
                )

                # Print final validation report
                if final_violations:
                    logger.warning(f"--- PDB Validation Report for {os.path.abspath(output_filename)} ---")
                    logger.warning(f"Final PDB has {len(final_violations)} violations.")
                    for violation in final_violations:
                        logger.warning(violation)
                    logger.warning("--- End Validation Report ---")
                elif args.validate:
                    logger.info(f"No violations found in the final PDB for {os.path.abspath(output_filename)}.")

                # Phase 7, 8, & 9 + 10: Synthetic NMR Data & Exports
                # We perform calculations first, so we can capture data (like restraints) for visualization if needed.
                generated_restraints = None # To hold restraints for viewer
                
                if args.gen_nef or args.gen_relax or args.gen_shifts or args.gen_couplings or args.export_constraints or args.export_torsion or args.gen_msa or args.export_distogram:
                    if args.mode != "generate":
                        logger.warning("Synthetic Data Generation is currently only supported in single structure 'generate' mode.")
                    else:
                        from .nmr import calculate_synthetic_noes
                        from .nef_io import write_nef_file, write_nef_relaxation, write_nef_chemical_shifts
                        from .relaxation import calculate_relaxation_rates
                        from .chemical_shifts import predict_chemical_shifts
                        # NEW IMPORTS for Export
                        from .contact import compute_contact_map
                        from .export import export_constraints
                        from .torsion import calculate_torsion_angles, export_torsion_angles
                        from .evolution import generate_msa_sequences, write_msa
                        from .distogram import calculate_distogram, export_distogram
                        
                        import biotite.structure.io.pdb as pdb_io
                        import io
                        import numpy as np
                        
                        logger.info("Generating Synthetic Data...")
                        
                        # We need the generated structure as an AtomArray
                        pdb_file = pdb_io.PDBFile.read(io.StringIO(final_full_pdb_content_to_write))
                        structure = pdb_file.get_structure(model=1)
                        
                        # Sequence inference
                        res_names = [structure[structure.res_id == i][0].res_name for i in sorted(list(set(structure.res_id)))]
                        from .data import ONE_TO_THREE_LETTER_CODE
                        three_to_one = {v: k for k, v in ONE_TO_THREE_LETTER_CODE.items()}
                        seq_str = "".join([three_to_one.get(r, "X") for r in res_names])

                        # Validation: Check for Hydrogens
                        if not np.any(structure.element == "H") and (args.gen_nef or args.gen_relax):
                             logger.error("Structure has no hydrogens! NEF/Relaxation requires protons. Use --minimize.")
                        else:
                            # 1. NOE Restraints (Phase 7)
                            if args.gen_nef:
                                logger.info("Calculating NOE Restraints...")
                                restraints = calculate_synthetic_noes(structure, cutoff=args.noe_cutoff)
                                generated_restraints = restraints # Capture for viewer
                                
                                nef_filename = args.nef_output
                                if not nef_filename:
                                    nef_filename = output_filename.replace(".pdb", ".nef")
                                
                                write_nef_file(nef_filename, seq_str, restraints)
                                logger.info(f"NEF Restraints generated: {os.path.abspath(nef_filename)}")
                                
                                if args.gen_pymol:
                                    from .visualization import generate_pymol_script
                                    pml_filename = output_filename.replace(".pdb", ".pml")
                                    generate_pymol_script(os.path.basename(output_filename), restraints, pml_filename)
                                    logger.info(f"PyMOL Visualization Script generated: {os.path.abspath(pml_filename)}")

                            # 2. Relaxation Data (Phase 8)
                            if args.gen_relax:
                                rates = calculate_relaxation_rates(structure, field_mhz=args.field, tau_m_ns=args.tumbling_time)
                                relax_filename = output_filename.replace(".pdb", "_relax.nef")
                                write_nef_relaxation(relax_filename, seq_str, rates, field_freq_mhz=args.field)

                            # 3. Chemical Shifts (Phase 9)
                            if args.gen_shifts:
                                shifts = predict_chemical_shifts(structure)
                                shift_filename = args.shift_output if args.shift_output else output_filename.replace(".pdb", "_shifts.nef")
                                write_nef_chemical_shifts(shift_filename, seq_str, shifts)
                                shift_filename = args.shift_output if args.shift_output else output_filename.replace(".pdb", "_shifts.nef")
                                write_nef_chemical_shifts(shift_filename, seq_str, shifts)
                                logger.info(f"NEF Chemical Shift Data generated: {os.path.abspath(shift_filename)}")

                            # 3.5 J-Couplings (Phase 9.5)
                            if args.gen_couplings:
                                from .coupling import predict_couplings_from_structure
                                # Reuse torsion calc if not already done
                                # Ideally we'd optimize to not recalc, but calculation is cheap.
                                angles_list = calculate_torsion_angles(structure)
                                
                                # Convert generic List[Dict] to Dict[int, float] for phis
                                phi_map = {}
                                for angle_data in angles_list:
                                    if angle_data['phi'] is not None:
                                        phi_map[angle_data['res_id']] = angle_data['phi']
                                    else:
                                        phi_map[angle_data['res_id']] = np.nan
                                        
                                couplings = predict_couplings_from_structure(phi_map)
                                
                                coupling_csv = args.coupling_output if args.coupling_output else output_filename.replace(".pdb", "_couplings.csv")
                                
                                with open(coupling_csv, "w") as f:
                                    f.write("res_id,residue,J_HN_HA\n")
                                    # Write sorted by resid
                                    for angle_data in angles_list:
                                        rid = angle_data['res_id']
                                        res = angle_data['residue']
                                        jval = couplings.get(rid, np.nan)
                                        f.write(f"{rid},{res},{jval:.4f}\n")
                                        
                                logger.info(f"J-Couplings exported: {os.path.abspath(coupling_csv)}")
                                
                            # 4. Constraint Export (Phase 10)
                            if args.export_constraints:
                                logger.info(f"Exporting Constraints in {args.constraint_format.upper()} format...")
                                # Format: RR or CSV
                                # We use compute_contact_map relative to cutoff
                                # For Export, we typically want BINARY map if using CASP
                                
                                # Calculate Distance Matrix for export
                                matrix = compute_contact_map(
                                    structure, 
                                    method="ca", 
                                    threshold=args.constraint_cutoff
                                )
                                
                                content = export_constraints(matrix, seq_str, fmt=args.constraint_format, threshold=args.constraint_cutoff)
                                
                                export_file = args.export_constraints
                                with open(export_file, "w") as f:
                                    f.write(content)
                                logger.info(f"Constraints exported to: {os.path.abspath(export_file)}")

                            # 5. Torsion Export (Phase 11)
                            if args.export_torsion:
                                angles = calculate_torsion_angles(structure)
                                export_torsion_angles(angles, args.export_torsion, fmt=args.torsion_format)
                                logger.info(f"Torsion angles exported to: {os.path.abspath(args.export_torsion)}")
                            
                            # 6. MSA Generation (Phase 12)
                            if args.gen_msa:
                                sequences = generate_msa_sequences(structure, n_seqs=args.msa_depth, mutation_rate=args.mutation_rate)
                                msa_filename = output_filename.replace(".pdb", ".fasta")
                                write_msa(sequences, msa_filename)
                                logger.info(f"Synthetic MSA generated: {os.path.abspath(msa_filename)}")

                            # 7. Distogram Export (Phase 13)
                            if args.export_distogram:
                                matrix = calculate_distogram(structure)
                                export_distogram(matrix, args.export_distogram, fmt=args.distogram_format)
                                logger.info(f"Distogram exported to: {os.path.abspath(args.export_distogram)}")

                # Open 3D viewer if requested (MOVED AFTER NMR calc to access generated_restraints)
                if args.visualize:
                    logger.info("Opening 3D molecular viewer in browser...")
                    try:
                        view_structure_in_browser(
                            final_full_pdb_content_to_write,
                            filename=output_filename,
                            style="cartoon",
                            color="spectrum",
                            restraints=generated_restraints, # Pass captured restraints
                            highlights=highlights, # Pass beta-turn highlights
                            show_hbonds=True
                        )
                    except Exception as e:
                        logger.error(f"Failed to open 3D viewer: {e}") 

            except Exception as e:
                logger.error("An unexpected error occurred during file writing: %s", e)
                sys.exit(1)
        else:
            # If final_pdb_atomic_content is None (implies final_pdb_content was None originally)
            logger.error("No suitable PDB content was generated for writing.")
            sys.exit(1)


if __name__ == "__main__":
    main()
