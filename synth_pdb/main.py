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
from .pdb_utils import extract_atomic_content, assemble_pdb_content
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
        "--structure",
        type=str,
        default=None,
        help="Per-region conformation specification (NEW!). Format: 'start-end:conformation,...' Example: '1-10:alpha,11-20:beta'. Allows mixed secondary structures. Unspecified residues use --conformation default.",
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
        choices=["generate", "decoys", "docking"],
        help="Operation mode: 'generate' (default) single structure, 'decoys' ensemble, 'docking' preparation (PQR).",
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
        help="Input PDB file path (required for --mode docking).",
    )
    
    # Phase 7: Synthetic NMR Data
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
            )

            if not current_pdb_content:
                logger.warning(f"Failed to generate PDB content in attempt {attempt_num}. Skipping.")
                continue

            if args.validate:
                logger.info("Performing PDB validation checks for current generation...")
                logger.debug(f"PDB content passed to validator (attempt {attempt_num}):\n{current_pdb_content}")
                validator = PDBValidator(current_pdb_content)
                validator.validate_bond_lengths()
                validator.validate_bond_angles()
                validator.validate_ramachandran()
                validator.validate_steric_clashes()
                validator.validate_peptide_plane()
                validator.validate_sequence_improbabilities()
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

        except ValueError as ve:
            logger.error(f"Error processing sequence during generation (attempt {attempt_num}): {ve}. Skipping.")
        except Exception as e:
            logger.error(f"An unexpected error occurred during generation (attempt {attempt_num}): {e}. Skipping.")

    if final_pdb_content is None:
        logger.error(f"Failed to generate a suitable PDB file after {generation_attempts} attempts.")
        sys.exit(1)
    else:
        # Extract atomic content from the initially selected PDB for subsequent refinement or final assembly.
        final_pdb_atomic_content = extract_atomic_content(final_pdb_content)

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
                    new_atomic_content_after_tweak, 1, command_args=cmd_string
                )
                temp_validator = PDBValidator(pdb_content=temp_full_pdb)
                temp_validator.validate_bond_lengths()
                temp_validator.validate_bond_angles()
                temp_validator.validate_ramachandran()
                temp_validator.validate_steric_clashes()
                temp_validator.validate_peptide_plane()
                temp_validator.validate_sequence_improbabilities()
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
                final_pdb_atomic_content, final_sequence_length, command_args=cmd_string
            )

            if args.output:
                output_filename = args.output
                logger.debug("Using user-provided output filename: %s", output_filename)
            else:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                if args.sequence:
                    # Use a simplified sequence string for filename to avoid very long names
                    sequence_tag = args.sequence.replace("-", "")[
                        :10
                    ]  # Take first 10 chars, remove hyphens
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

                # Open 3D viewer if requested
                if args.visualize:
                    logger.info("Opening 3D molecular viewer in browser...")
                    try:
                        view_structure_in_browser(
                            final_full_pdb_content_to_write,
                            filename=output_filename,
                            style="cartoon",
                            color="spectrum"
                        )
                    except Exception as e:
                        logger.error(f"Failed to open 3D viewer: {e}")
                        # Don't fail the entire program if visualization fails
                
                # Phase 7: Generate NEF if requested
                if args.gen_nef:
                    if args.mode != "generate":
                        logger.warning("NEF generation is currently only supported in single structure 'generate' mode (for now).")
                    else:
                        from .nmr import calculate_synthetic_noes
                        from .nef_io import write_nef_file
                        import biotite.structure.io.pdb as pdb_io
                        
                        logger.info("Generating Synthetic NMR Data (NEF)...")
                        
                        # We need the generated structure as an AtomArray
                        # Ideally we reuse the object, but here we have the string.
                        # Re-parsing ensures we use exactly what was written to disk.
                        pdb_file = pdb_io.PDBFile.read(io.StringIO(final_full_pdb_content_to_write))
                        structure = pdb_file.get_structure(model=1)
                        
                        # Validation: Check for Hydrogens
                        if not np.any(structure.element == "H"):
                             logger.error("Generated structure has no hydrogens! NEF generation requires protons. Did you use --minimize (Phase 2)?")
                        else:
                            restraints = calculate_synthetic_noes(structure, cutoff=args.noe_cutoff)
                            
                            nef_filename = args.nef_output
                            if not nef_filename:
                                nef_filename = output_filename.replace(".pdb", ".nef")
                                
                            # Sequence: get from args or infer
                            seq_str = args.sequence if args.sequence else ""
                            # If random, we need to infer.
                            if not seq_str:
                                # Quick inference from structure
                                # (omitted for brevity, relying on args.sequence or minimal fallback in nef_io if needed)
                                # Actually nef_io handles this if we pass the inferred sequence length? No, it needs string.
                                # Let's try to get it from the 1-letter code map or just pass "UNK" if strictly random without tracking.
                                # Better: generator returns sequence. But here in main we might have lost it.
                                # For V1.4, let's assume args.sequence OR we parse residue names.
                                # Parse residues:
                                res_names = [structure[structure.res_id == i][0].res_name for i in sorted(list(set(structure.res_id)))]
                                from .data import THREE_TO_ONE_LETTER_CODE
                                seq_str = "".join([THREE_TO_ONE_LETTER_CODE.get(r, "X") for r in res_names])

                            write_nef_file(nef_filename, seq_str, restraints)
                            logger.info(f"NEF file generated: {os.path.abspath(nef_filename)}")


            except Exception as e:
                logger.error("An unexpected error occurred during file writing: %s", e)
                sys.exit(1)
        else:
            # If final_pdb_atomic_content is None (implies final_pdb_content was None originally)
            logger.error("No suitable PDB content was generated for writing.")
            sys.exit(1)


if __name__ == "__main__":
    main()
