
import numpy as np

def export_constraints(contact_map: np.ndarray, sequence: str, fmt: str = "casp", separation_cutoff: int = 0) -> str:
    """
    Export a Contact Map to text format for AI modeling.
    
    Parameters:
    -----------
    contact_map : np.ndarray
        NxN matrix. Values can be Binary (0/1) or Probabilities (0.0-1.0).
    sequence : str
        The protein sequence (required for CASP header).
    fmt : str
        "casp" (CASP RR format) or "csv" (Simple list).
    separation_cutoff : int
        Minimum sequence separation |i-j| to include. 
        Default 0 includes neighbors. CASP often requires 6.
        
    Returns:
    --------
    content : str
        The textual content of the file.
    """
    n_res = contact_map.shape[0]
    lines = []
    
    if fmt == "casp":
        # CASP RR Format
        # SEQ
        # MODEL 1
        # i j d_min d_max prob
        lines.append(sequence)
        
        for i in range(n_res):
            for j in range(i + 1 + separation_cutoff, n_res):
                val = contact_map[i, j]
                if val > 0.0:
                    # Residues are 1-indexed in CASP
                    res_i = i + 1
                    res_j = j + 1
                    # d_min, d_max, prob
                    # Standard contact definition is < 8 Angstroms
                    # If input is binary 1.0, we assume it means d < 8.0
                    lines.append(f"{res_i} {res_j} 0.0 8.0 {val:.5f}")
    
    elif fmt == "csv":
        # CSV Format
        # Res1,ResName1,Res2,ResName2,Value
        lines.append("Res1,Res2,Value")
        for i in range(n_res):
            for j in range(i + 1 + separation_cutoff, n_res):
                val = contact_map[i, j]
                if val > 0.0:
                    lines.append(f"{i+1},{j+1},{val:.5f}")
    
    else:
        raise ValueError(f"Unknown format: {fmt}")
        
    return "\n".join(lines)
