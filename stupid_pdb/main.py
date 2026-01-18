"""
CLI entry point for the stupid-pdb tool.
"""

import argparse
import logging
import datetime
import os
import sys

from .generator import generate_pdb_content
from .validator import PDBValidator

# Get a logger for this module
logger = logging.getLogger(__name__)


def main():
    """
    Main function to parse arguments and generate the PDB file.
    """
    parser = argparse.ArgumentParser(
        description="Generate a PDB file with a random linear amino acid sequence."
    )
    parser.add_argument(
        "--length",
        type=int,
        default=10,
        help="Length of the amino acid sequence (number of residues).",
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
        "--full-atom",
        action="store_true",
        help="Generate a full atomic representation (N, CA, C, O, and side chain atoms) instead of only C-alpha atoms.",
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

    args = parser.parse_args()

    # Set the logging level based on user input
    log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    logging.getLogger().setLevel(log_level)
    logger.debug("Logging level set to %s.", args.log_level.upper())

    logger.info("Starting PDB file generation process.")
    logger.debug(
        "Parsed arguments: length=%s, output='%s', full_atom=%s, sequence='%s', plausible_frequencies=%s, validate=%s",
        args.length,
        args.output,
        args.full_atom,
        args.sequence,
        args.plausible_frequencies,
        args.validate,
    )

    # Validate length only if no sequence is provided
    if args.sequence is None and (args.length is None or args.length <= 0):
        logger.error("Length must be a positive integer when no sequence is provided.")
        sys.exit(1)

    # If a sequence is provided, its length will be used by the generator, so we can pass None for length
    # or the specified length for consistency if needed by the generator function for logging.
    # The generator will determine the actual length from the sequence if provided.
    length_for_generator = args.length if args.sequence is None else None

    try:
        pdb_content = generate_pdb_content(
            length=length_for_generator,
            full_atom=args.full_atom,
            sequence_str=args.sequence,
            use_plausible_frequencies=args.plausible_frequencies,
        )

        if not pdb_content:
            logger.error("Failed to generate PDB content. Exiting.")
            sys.exit(1)

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

        with open(output_filename, "w") as f:
            f.write(pdb_content)
        logger.info(
            "Successfully generated PDB file: %s", os.path.abspath(output_filename)
        )

        # Perform validation if requested
        if args.validate:
            logger.info("Performing PDB validation checks...")
            validator = PDBValidator(pdb_content)
            validator.validate_bond_lengths()
            validator.validate_bond_angles()
            validator.validate_ramachandran()
            validator.validate_steric_clashes()
            validator.validate_peptide_plane()
            validator.validate_sequence_improbabilities()

            violations = validator.get_violations()
            if violations:
                logger.warning("--- PDB Validation Report ---")
                for violation in violations:
                    logger.warning(violation)
                logger.warning("--- End Validation Report ---")
            else:
                logger.info(
                    "No bond length, angle, Ramachandran, steric, peptide plane, or sequence improbability violations found."
                )
    except ValueError as ve:
        logger.error("Error processing sequence: %s", ve)
        sys.exit(1)
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
