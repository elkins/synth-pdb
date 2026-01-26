
    # Phase 9.5: AI Constraint Export
    parser.add_argument(
        "--export-constraints",
        type=str,
        help="Export contact map constraints for AI modeling (e.g. AlphaFold, CASP). Specify output filename.",
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
