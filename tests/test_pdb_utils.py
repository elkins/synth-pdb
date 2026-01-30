
import pytest
from synth_pdb.pdb_utils import extract_header_records, assemble_pdb_content

def test_extract_header_records_ssbond():
    """Verify EXTRACTING SSBOND records from full PDB works."""
    pdb_content = (
        "HEADER    PEPTIDE\n"
        "SSBOND   1 CYS A    1    CYS A    6\n"
        "SSBOND   2 CYS A   10    CYS A   15\n"
        "ATOM      1  N   ALA A   1\n"
    )
    
    extracted = extract_header_records(pdb_content, "SSBOND")
    assert "SSBOND   1 CYS A    1    CYS A    6" in extracted
    assert "SSBOND   2 CYS A   10    CYS A   15" in extracted
    assert "ATOM" not in extracted

def test_extract_header_records_no_match():
    """Verify behavior when no records match."""
    pdb_content = "ATOM      1  N   ALA A   1\n"
    extracted = extract_header_records(pdb_content, "SSBOND")
    assert extracted == ""

def test_assemble_pdb_content_preservation():
    """Verify assemble_pdb_content respects extra_records."""
    atomic_content = "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N"
    extra = "SSBOND   1 CYS A    1    CYS A    6"
    
    full_pdb = assemble_pdb_content(atomic_content, 1, extra_records=extra)
    
    assert "HEADER" in full_pdb
    assert "SSBOND   1 CYS A    1    CYS A    6" in full_pdb
    assert "ATOM" in full_pdb
