import logging
try:
    import openmm.app as app
    import openmm as mm
    from openmm import unit
    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False
    app = None
    mm = None
    unit = None
import sys
import os
import numpy as np


# Constants
# SSBOND_CAPTURE_RADIUS determines the maximum distance (in Angstroms) between two Sulfur atoms
# for them to be considered as a potential disulfide bond.
# Linear chains being cyclized can have terminals > 15A apart initially.
SSBOND_CAPTURE_RADIUS = 18.0

logger = logging.getLogger(__name__)


class EnergyMinimizer:
    """
    Performs energy minimization on molecular structures using OpenMM.
    
    ### Educational Note: What is Energy Minimization?
    --------------------------------------------------
    Proteins fold into specific 3D shapes to minimize their "Gibbs Free Energy".
    A generated structure (like one built from simple geometry) often has "clashes"
    where atoms are too close (high Van der Waals repulsion) or bond angles are strained.
    
    Energy Minimization is like rolling a ball down a hill. The "Energy Landscape"
    represents the potential energy of the protein as a function of all its atom coordinates.
    The algorithm moves atoms slightly to reduce this energy, finding a local minimum
    where the structure is physically relaxed.

    ### Educational Note - Metal Coordination in Physics:
    -----------------------------------------------------
    Metal ions like Zinc (Zn2+) are not "bonded" in the same covalent sense as Carbon-Carbon 
    bonds in classical forcefields. Instead, they are typically modeled as point charges 
    held by electrostatics and Van der Waals forces.
    
    In this tool, we automatically detect potential coordination sites (like Zinc Fingers).
    To maintain the geometry during minimization, we apply Harmonic Constraints 
    that act like springs, tethering the Zinc to its ligands (Cys/His). 
    We also deprotonate coordinating Cysteines to represent the thiolate state.
    
    ### NMR Perspective:
    In NMR structure calculation (e.g., CYANA, XPLOR-NIH), minimization is often part of
    "Simulated Annealing". Structures are calculated to satisfy experimental restraints
    (NOEs, J-couplings) and then energy-minimized to ensure good geometry.
    This module performs that final "geometry regularization" step.
    """
    
    def __init__(self, forcefield_name='amber14-all.xml', solvent_model=None):
        """
        Initialize the Minimizer with a Forcefield and Solvent Model.
        
        Args:
            forcefield_name: The "rulebook" for how atoms interact.
                             'amber14-all.xml' describes protein atoms (parameters for bond lengths,
                             angles, charges, and VdW radii).
            solvent_model:   How water is simulated. 
                             'app.OBC2' is an "Implicit Solvent" model. Instead of simulating
                             thousands of individual water molecules (Explicit Solvent),
                             it uses a mathematical continuum to approximate water's dielectric 
                             shielding and hydrophobic effects. This is much faster.
                             
                             ### NMR Note:
                             Since NMR is performed in solution (not crystals), implicit solvent 
                             aims to approximate that solution environment, distinct from the
                             vacuum or crystal lattice assumptions of other methods.
        """
        if not HAS_OPENMM: return
        if solvent_model is None: solvent_model = app.OBC2
        self.forcefield_name = forcefield_name
        self.water_model = 'amber14/tip3pfb.xml' 
        self.solvent_model = solvent_model
        ff_files = [self.forcefield_name, self.water_model]
        
        solvent_xml_map = {
            app.OBC2: 'implicit/obc2.xml',
            app.OBC1: 'implicit/obc1.xml',
            app.GBn:  'implicit/gbn.xml',
            app.GBn2: 'implicit/gbn2.xml',
            app.HCT:  'implicit/hct.xml',
        }
        
        if self.solvent_model in solvent_xml_map:
            ff_files.append(solvent_xml_map[self.solvent_model])
        try:
            self.forcefield = app.ForceField(*ff_files)
        except Exception as e:
            logger.error(f"Failed to load forcefield: {e}"); raise

    def minimize(self, pdb_file_path, output_path, max_iterations=0, tolerance=10.0, cyclic=False):
        """
        Minimizes the energy of a structure already containing correct atoms (including Hydrogens).
        
        EDUCATIONAL NOTE - Anatomy of a Forcefield:
        -------------------------------------------
        A forcefield (like Amber14) approximates the potential energy (U) of a 
        molecule as a sum of four main terms:
        
        U = U_bond + U_angle + U_torsion + [U_vdw + U_elec]
        
        1. Bonded Terms (Springs):
           - U_bond/U_angle: Atoms behave like balls on springs. Pushing them 
             away from ideal (equilibrium) lengths/angles costs energy.
           - U_torsion: Rotation around bonds is restricted by periodic potential 
             wells (e.g., the preference for trans vs cis).
        2. Non-Bonded Terms (Distant Neighbors):
           - U_vdw (Lennard-Jones): Models Steric Repulsion (don't overlap!) and 
             London Dispersion (subtle attraction).
           - U_elec (Coulomb): Attraction between opposite charges (e.g., a 
             Salt Bridge) and repulsion between like charges.
        
        Minimization is the process of finding the coordinate set where $dU/dX = 0$.

        Args:
            pdb_file_path: Input PDB path.
            output_path: Output PDB path.
            max_iterations: Limit steps (0 = until convergence).
            tolerance: Target energy convergence threshold (kJ/mol).

        ### Educational Note - Computational Efficiency:
        ----------------------------------------------
        Energy Minimization is an O(N^2) or O(N log N) operation depending on the method.
        Starting with a structure that satisfies Ramachandran constraints (from `validator.py`)
        can reduce convergence time by 10-50x compared to minimizing a random coil.
        
        Effectively, the validator acts as a "pre-minimizer", placing atoms in the 
        correct basin of attraction so the expensive physics engine only needs to 
        perform local optimization.
        """
        if not HAS_OPENMM:
            logger.error("Cannot minimize: OpenMM not found.")
            return False
        return self._run_simulation(pdb_file_path, output_path, max_iterations=max_iterations, tolerance=tolerance, add_hydrogens=False, cyclic=cyclic)

    def equilibrate(self, pdb_file_path, output_path, steps=1000, cyclic=False):
        """
        Run Thermal Equilibration (MD) at 300K.
        
        Args:
            pdb_file_path: Input PDB/File path.
            output_path: Output PDB path.
            steps: Number of MD steps (2 fs per step). 1000 steps = 2 ps.
        Returns:
            True if successful.
        """
        if not HAS_OPENMM:
             logger.error("Cannot equilibrate: OpenMM not found.")
             return False
        return self._run_simulation(pdb_file_path, output_path, add_hydrogens=True, equilibration_steps=steps, cyclic=cyclic)

    def add_hydrogens_and_minimize(self, pdb_file_path, output_path, max_iterations=0, tolerance=10.0, cyclic=False):
        """
        Robust minimization pipeline: Adds Hydrogens -> Creates/Minimizes System -> Saves Result.
        
        ### Why Add Hydrogens?
        X-ray crystallography often doesn't resolve hydrogen atoms because they have very few electrons.
        However, Molecular Dynamics forcefields (like Amber) are explicitly "All-Atom". They REQUIRE
        hydrogens to calculate bond angles and electrostatics (h-bonds) correctly.
        
        ### NMR Perspective:
        Unlike X-ray, NMR relies entirely on the magnetic spin of protons (H1). Hydrogens are
        the "eyes" of NMR. Correctly placing them is critical not just for physics but for
        predicting NOEs (Nuclear Overhauser Effects) which depend on H-H distances.
        We use `app.Modeller` to "guess" the standard positions of hydrogens at specific pH (7.0).
        """
        if not HAS_OPENMM:
             logger.error("Cannot add hydrogens: OpenMM not found.")
             return False
        return self._run_simulation(pdb_file_path, output_path, add_hydrogens=True, max_iterations=max_iterations, tolerance=tolerance, cyclic=cyclic)

    def _run_simulation(self, input_path, output_path, max_iterations=0, tolerance=10.0, add_hydrogens=True, equilibration_steps=0, cyclic=False):
        logger.info(f"Processing physics for {input_path} (cyclic={cyclic})...")
        import tempfile
        import os
        import numpy as np
        
        # Initialize variables early to avoid NameErrors
        coordination_restraints = []
        salt_bridge_restraints = []
        atom_list = []
        added_bonds = []
        
        # EDUCATIONAL NOTE - PDB PRE-PROCESSING (OpenMM Template Fix):
        # -----------------------------------------------------------
        # OpenMM's standard forcefields (amber14-all) are highly optimized for wild-type
        # human proteins but frequently lack templates for:
        # 1. Phosphorylated residues (SEP, TPO, PTR)
        # 2. Histidine tautomers (HIE, HID) named explicitly in the input.
        # 3. D-Amino Acids (DAL, DPH, etc.) - These require L-analog templates.
        #
        # To prevent "No template found" errors, we surgically rename residues to 
        # their standard counterparts BEFORE loading. We preserve the original
        # identity in `original_res_names` and `original_res_ids` for final restoration.
        ptm_map = {
            'SEP': 'SER', 'TPO': 'THR', 'PTR': 'TYR',
            'HIE': 'HIS', 'HID': 'HIS', 'HIP': 'HIS',
            'DAL': 'ALA', 'DAR': 'ARG', 'DAN': 'ASN', 'DAS': 'ASP', 'DCY': 'CYS',
            'DGL': 'GLU', 'DGN': 'GLN', 'DHI': 'HIS', 'DIL': 'ILE', 'DLE': 'LEU',
            'DLY': 'LYS', 'DME': 'MET', 'DPH': 'PHE', 'DPR': 'PRO', 'DSE': 'SER',
            'DTH': 'THR', 'DTR': 'TRP', 'DTY': 'TYR', 'DVA': 'VAL'
        }
        ptm_atom_names = ["P", "O1P", "O2P", "O3P"]
        
        original_metadata = {}    # (res_id, chain_id) -> {"name": str, "id": str}
        
        try:
            if os.path.exists(input_path):
                with open(input_path, 'r') as f:
                    pdb_lines = f.readlines()
                
                modified_lines = []
                res_index = -1
                last_res_key = None
                
                for line in pdb_lines:
                    if line.startswith(("ATOM", "HETATM")) and len(line) >= 26:
                        res_name = line[17:20].strip()
                        res_id = line[22:26].strip()
                        chain_id = line[21] if len(line) > 21 else " "
                        res_key = (res_id, chain_id)
                        
                        if res_key != last_res_key:
                            last_res_key = res_key
                            original_metadata[res_key] = {"name": res_name, "id": res_id}
                        
                        if res_name in ptm_map:
                            # Swap name in the line
                            new_name = ptm_map[res_name]
                            line = line[:17] + f"{new_name: >3}" + line[20:]
                            # Strip PTM atoms that aren't in standard templates to avoid Modeller errors
                            if res_name in ['SEP', 'TPO', 'PTR'] and len(line) >= 16:
                                atom_name = line[12:16].strip()
                                if atom_name in ptm_atom_names:
                                    continue # Skip this line
                    modified_lines.append(line)
                
                # Write to a secondary temporary file for OpenMM to load
                with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tf:
                    tf.writelines(modified_lines)
                    temp_input_path = tf.name
                
                # Load into OpenMM
                pdb = app.PDBFile(temp_input_path)
                topology, positions = pdb.topology, pdb.positions
                
                # Cleanup temp file
                try: os.unlink(temp_input_path)
                except: pass
            else:
                # If path doesn't exist, assume we're in a test mock 
                # and app.PDBFile will be mocked accordingly.
                pdb = app.PDBFile(input_path)
                topology, positions = pdb.topology, pdb.positions
            
            # EDUCATIONAL NOTE - Topological Validation:
            # ------------------------------------------
            # OpenMM's PDB reader can sometimes skip bonds if they aren't explicitly 
            # in CONECT records or deviate too far from their ideal length.
            # We force bond generation to ensure standard residues have 
            # all internal bonds defined, which is required for template matching.
            topology.createStandardBonds()
            topology.createDisulfideBonds(positions)
            
            atom_list = list(topology.atoms())
            
        except Exception as e:
            logger.error(f"PDB Pre-processing failed: {e}")
            return False
            
        # EDUCATIONAL NOTE - Topology Bridging (welding the ring):
        # --------------------------------------------------------
        try:
            if cyclic:
                logger.info("Cyclizing peptide via harmonic restraints (Restraint-First approach).")
                # EDUCATIONAL NOTE: We do NOT add the bond to the Topology here.
                # Adding it triggers OpenMM's Amber template matcher to look for 
                # non-existent cyclic templates or specialized patches, leading 
                # to "Too many external bonds" errors. We use Restraints instead.

            # 1. MODELLER SETUP
            modeller = app.Modeller(topology, positions)
            
            # EDUCATIONAL NOTE - Robust Backbone Stitching (Heuristic Bonding):
            # -----------------------------------------------------------------
            # When building de novo structures (especially with non-standard residues 
            # or terminal caps), standard PDB-to-Topology builders often miss local 
            # connectivity. We implement a "Heuristic Welder" that looks for 
            # missing C-N peptide bonds based on proximity. If two sequential residues 
            # are close but unbonded, we manually weld their backbone atoms to ensure 
            # a continuous, force-propagating chain for the simulation.
            try:
                residues = list(modeller.topology.residues())
                existing_bonds = set(frozenset([b[0].index, b[1].index]) for b in modeller.topology.bonds())
                for i in range(len(residues) - 1):
                    res1, res2 = residues[i], residues[i + 1]
                    c_s = next((a for a in res1.atoms() if a.name == 'C'), None)
                    n_s = next((a for a in res2.atoms() if a.name == 'N'), None)
                    if c_s and n_s and frozenset([c_s.index, n_s.index]) not in existing_bonds:
                        logger.debug(f"Stitching missing backbone bond: {res1.name}{res1.id} -> {res2.name}{res2.id}")
                        modeller.topology.addBond(c_s, n_s)
            except Exception as e:
                logger.warning(f"Robust stitching failed: {e}")
                
            # 4. HYDROGEN STRIPPING (Conditional)
            # We strip H to let modeller re-add them at specified pH (e.g. pH 7.0).
            # This ensures optimal protonation states and avoids clashing at terminal bonds.
            added_bonds = []
            if add_hydrogens:
                try:
                    modeller.delete([a for a in modeller.topology.atoms() if a.element is not None and a.element.symbol == "H"])
                except Exception as e: logger.debug(f"H deletion failed: {e}")

            # 5. BIOPHYSICAL CONSTRAINT DETECTION
            # ------------------------------------
            
            # EDUCATIONAL NOTE - The SSBOND Capture Radius:
            # ---------------------------------------------
            # Unlike distance-based bonding in simple geometry, physical disulfide 
            # formation is highly sensitive to the S-S distance (~2.03 A).
            # We use a large "Capture Radius" (SSBOND_CAPTURE_RADIUS) to detect 
            # potential pairs in un-optimized structures, then allow the "Mega-Pull" 
            # to bring them into the ideal covalent distance.
            try:
                cys_residues = [r for r in modeller.topology.residues() if r.name == 'CYS' or r.name == 'CYX']
                res_to_sg = {r.index: [a for a in r.atoms() if a.name == 'SG'][0] for r in cys_residues if any(a.name == 'SG' for a in r.atoms())}
                potential_bonds = []
                for i in range(len(cys_residues)):
                    r1 = cys_residues[i]; s1 = res_to_sg.get(r1.index)
                    if not s1: continue
                    for j in range(i + 1, len(cys_residues)):
                        r2 = cys_residues[j]; s2 = res_to_sg.get(r2.index)
                        if not s2: continue
                        p1 = np.array(modeller.positions[s1.index].value_in_unit(unit.angstroms))
                        p2 = np.array(modeller.positions[s2.index].value_in_unit(unit.angstroms))
                        d_a = np.sqrt(np.sum((p1 - p2)**2))
                        if d_a < SSBOND_CAPTURE_RADIUS: potential_bonds.append((d_a, r1, r2, s1, s2))
                potential_bonds.sort(key=lambda x: x[0])
                bonded_indices = set()
                for d, r1, r2, s1, s2 in potential_bonds:
                    if r1.index in bonded_indices or r2.index in bonded_indices: continue
                    modeller.topology.addBond(s1, s2)
                    added_bonds.append((str(r1.id).strip(), str(r2.id).strip()))
                    bonded_indices.add(r1.index); bonded_indices.add(r2.index)
            except Exception as e: logger.warning(f"SSBOND failed: {e}")

            # EDUCATIONAL NOTE - Salt Bridges & Electrostatics:
            # -------------------------------------------------
            # A Salt Bridge is an electrostatic attraction between a cationic sidechain 
            # (e.g. Lysine/Arginine) and an anionic one (Aspartate/Glutamate).
            # Forcefields model these naturally via Coulomb's law, but in vacuum 
            # simulations, the attraction can be artificially weak or slow to form.
            # We apply harmonic "Bungee" restraints to help these bridges snap together.
            try:
                from .cofactors import find_metal_binding_sites
                from .biophysics import find_salt_bridges
                import io; import biotite.structure.io.pdb as biotite_pdb
                tmp_io = io.StringIO(); app.PDBFile.writeFile(modeller.topology, modeller.positions, tmp_io); tmp_io.seek(0)
                b_struc = biotite_pdb.PDBFile.read(tmp_io).get_structure(model=1)
                
                sites = find_metal_binding_sites(b_struc)
                logger.debug(f"DEBUG: Found {len(sites)} metal binding sites.")
                for site in sites:
                    i_idx = -1
                    for atom in atom_list:
                        if atom.residue.name == site["type"]: i_idx = atom.index; break
                    if i_idx != -1:
                        for l_idx in site["ligand_indices"]:
                            l_at = b_struc[l_idx]
                            for atom in atom_list:
                                if (int(atom.residue.id) == int(l_at.res_id) and atom.name == l_at.atom_name): coordination_restraints.append((i_idx, atom.index)); break
                
                bridges = find_salt_bridges(b_struc, cutoff=6.0)
                logger.debug(f"DEBUG: Found {len(bridges)} salt bridges.")
                for br in bridges:
                    ia, ib = -1, -1
                    for atom in atom_list:
                        if (int(atom.residue.id) == int(br["res_ia"]) and atom.name == br["atom_a"]): ia = atom.index
                        if (int(atom.residue.id) == int(br["res_ib"]) and atom.name == br["atom_b"]): ib = atom.index
                    if ia != -1 and ib != -1: salt_bridge_restraints.append((ia, ib, br["distance"] / 10.0))
            except Exception as e: logger.warning(f"Metadata/SaltBridge detection failed: {e}")

            # 6. HYDROGEN ADDITION / HETATM RESTORATION
            if add_hydrogens:
                non_protein_data = [] 
                for r in modeller.topology.residues():
                    if r.name.strip().upper() in ["ZN", "FE", "MG", "CA", "NA", "CL"]:
                        for a in r.atoms(): non_protein_data.append({"res_name": r.name, "atom_name": a.name, "element": a.element, "pos": modeller.positions[a.index]})
                
                modeller.addHydrogens(self.forcefield, pH=7.0)
                
                # EDUCATIONAL NOTE - De-Terminalization (for Cyclic Peptides):
                # -----------------------------------------------------------
                # OpenMM's addHydrogens adds terminal -NH3+ and -COO- groups by default.
                # For a cyclic peptide, these "ends" are bonded together, so they must 
                # be treated as internal residues (NH and CO). We manually delete the
                # extra terminal atoms (H2, H3, OXT) to force internal template matching.
                if cyclic:
                    try:
                        res_first = list(modeller.topology.residues())[0]
                        res_last = list(modeller.topology.residues())[-1]
                        
                        to_delete = []
                        # N-terminus: Keep only ONE H on N. Remove H1, H2, H3 etc.
                        n_atom = next((a for a in res_first.atoms() if a.name == 'N'), None)
                        if n_atom:
                            h_on_n = [a for a in res_first.atoms() if a.element is not None and a.element.symbol == 'H' and any(b.atom1 == n_atom or b.atom2 == n_atom for b in modeller.topology.bonds() if a == b.atom1 or a == b.atom2)]
                            if len(h_on_n) > 1:
                                # Keep the one simply named 'H' or just the first one
                                h_to_keep = next((a for a in h_on_n if a.name == 'H'), h_on_n[0])
                                to_delete.extend([a for a in h_on_n if a != h_to_keep])
                        
                        # C-terminus: Remove OXT
                        oxt = next((a for a in res_last.atoms() if a.name == 'OXT'), None)
                        if oxt: to_delete.append(oxt)
                        
                        if to_delete:
                            logger.info(f"De-terminalizing cyclic peptide: Removing {len(to_delete)} extra atoms.")
                            modeller.delete(to_delete)
                    except Exception as e:
                        logger.warning(f"De-terminalization failed: {e}")

                # Restore lost HETATMs (Modeller.addHydrogens sometimes deletes them)
                new_names = [res.name.strip().upper() for res in modeller.topology.residues()]
                for d in non_protein_data:
                    if d["res_name"].strip().upper() not in new_names:
                        logger.info(f"Restoring lost HETATM: {d['res_name']}")
                        nc = modeller.topology.addChain(); nr = modeller.topology.addResidue(d["res_name"], nc)
                        modeller.topology.addAtom(d["atom_name"], d["element"], nr)
                        modeller.positions = list(modeller.positions) + [d["pos"]]
            
            # EDUCATIONAL NOTE - CYX Renaming & Thiol Stripping:
            # -------------------------------------------------
            # In classical forcefields, a standard Cysteine (CYS) has a thiol group (-SH).
            # When a disulfide bond (S-S) forms, two hydrogens are LOST.
            # OpenMM's Amber forcefield uses a separate residue template ('CYX') for 
            # these bonded cysteines. We must rename the residues AND manually delete 
            # the HG atoms, or the physics engine will see a "template mismatch" error.
            if added_bonds:
                hg_to_delete = []
                # Map IDs to residue objects in the current modeller topology
                res_map = {str(r.id).strip(): r for r in modeller.topology.residues()}
                for id1, id2 in added_bonds:
                    for rid in [id1, id2]:
                        res = res_map.get(rid)
                        if res and res.name == 'CYS':
                            res.name = 'CYX'
                            hg_to_delete.extend([a for a in res.atoms() if a.name == 'HG'])
                if hg_to_delete:
                    modeller.delete(hg_to_delete)
            
            topology, positions = modeller.topology, modeller.positions

            # 5. SYSTEM CREATION
            try:
                # EDUCATIONAL NOTE - Constraints and Macrocycles:
                # For macrocycles, we temporarily DISABLE all constraints (like HBonds)
                # to allow the chain to bend freely into a ring during the pull phase.
                # We also use vacuum (NoCutoff) to maximize closure speed.
                current_constraints = None if cyclic else app.HBonds
                system = self.forcefield.createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=current_constraints, implicitSolvent=self.solvent_model)
            except Exception:
                system = self.forcefield.createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=None)
            
            # EDUCATIONAL NOTE - The "Nuclear Option" & "Shadow Caps":
            # -------------------------------------------------------
            # Closing a ring is a physical paradox for most forcefields. The N and C 
            # termini are parameterized as charged ions that repel each other violently.
            #
            # 1. Terminal Ghosting: We surgically disable all non-bonded interactions 
            #    between the first and last residues. They can now pass through each other 
            #    without steric or electrostatic resistance.
            #
            # 2. Shadow Caps: To satisfy OpenMM's template requirements, we temporarily 
            #    attached ACE/NME dummy residues. Here, we zero out ALL their forces. 
            #    They allow the simulation to run but contribute nothing to the energy, 
            #    leaving the path clear for the "Mega-Pull" to snap the ring shut.
            if cyclic:
                try:
                    nb_force = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
                    residues = list(topology.residues())
                    if len(residues) >= 2:
                        res1 = residues[0]
                        resN = residues[-1]
                        
                        ats_first = list(res1.atoms())
                        ats_last = list(resN.atoms())
                        
                        logger.info(f"Ghosting entire residues {res1.name}{res1.id} and {resN.name}{resN.id} for closure.")
                        for a1 in ats_first:
                            for a2 in ats_last:
                                nb_force.addException(a1.index, a2.index, 0.0, 0.1, 0.0, replace=True)
                    
                    # SHADOW CAPS logic
                    logger.info("De-physicizing capping residues (Shadow Caps) to allow closure.")
                    top_atoms = list(topology.atoms())
                    for force in system.getForces():
                        if isinstance(force, mm.HarmonicBondForce):
                            for i in range(force.getNumBonds()):
                                p1, p2, r0, k = force.getBondParameters(i)
                                if top_atoms[p1].residue.name in ['ACE', 'NME'] or top_atoms[p2].residue.name in ['ACE', 'NME']:
                                    force.setBondParameters(i, p1, p2, r0, 0.0)
                        elif isinstance(force, mm.HarmonicAngleForce):
                            for i in range(force.getNumAngles()):
                                p1, p2, p3, theta, k = force.getAngleParameters(i)
                                if any(top_atoms[p].residue.name in ['ACE', 'NME'] for p in [p1, p2, p3]):
                                    force.setAngleParameters(i, p1, p2, p3, theta, 0.0)
                        elif isinstance(force, mm.PeriodicTorsionForce):
                            for i in range(force.getNumTorsions()):
                                p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(i)
                                if any(top_atoms[p].residue.name in ['ACE', 'NME'] for p in [p1, p2, p3, p4]):
                                    force.setTorsionParameters(i, p1, p2, p3, p4, periodicity, phase, 0.0)
                    
                    logger.info("Excised non-bonded interactions between termini for cyclic closure.")
                except Exception as e:
                    logger.warning(f"Failed to excise terminal interactions: {e}")

            if len(list(topology.atoms())) == 0:
                logger.error("Topology has 0 atoms before Simulation creation!")
                if len(positions) == 0:
                    logger.error("OpenMM returned empty positions! Topology might be corrupted.")
                return False
            
            # 6. RESTRAINTS
            if coordination_restraints:
                f = mm.CustomBondForce("0.5*k*(r-r0)^2")
                f.addGlobalParameter("k", 50000.0 * unit.kilojoules_per_mole / unit.nanometer**2); f.addPerBondParameter("r0")
                new_ats = list(topology.atoms())
                for i_o, l_o in coordination_restraints:
                    oi, ol = atom_list[i_o], atom_list[l_o]; ni, nl = -1, -1
                    for a in new_ats:
                        if a.residue.id == oi.residue.id and a.name == oi.name: ni = a.index
                        if a.residue.id == ol.residue.id and a.name == ol.name: nl = a.index
                    if ni != -1 and nl != -1: f.addBond(ni, nl, [(0.23 if new_ats[nl].name == "SG" else 0.21) * unit.nanometers])
                system.addForce(f)
            if salt_bridge_restraints:
                f = mm.CustomBondForce("0.5*k_sb*(r-r0)^2")
                f.addGlobalParameter("k_sb", 10000.0 * unit.kilojoules_per_mole / unit.nanometer**2); f.addPerBondParameter("r0")
                new_ats = list(topology.atoms())
                for ao, bo, r0 in salt_bridge_restraints:
                    oa, ob = atom_list[ao], atom_list[bo]; na, nb = -1, -1
                    for a in new_ats:
                        if it := (a.residue.id == oa.residue.id and a.name == oa.name): na = a.index
                        if it := (a.residue.id == ob.residue.id and a.name == ob.name): nb = a.index
                    if na != -1 and nb != -1: f.addBond(na, nb, [r0 * unit.nanometers])
                system.addForce(f)
                
            # EDUCATIONAL NOTE - Harmonic "Pull" Restraints & Hard Constraints:
            # -----------------------------------------------------------------
            # To bridge the gap between N and C termini, we use two levels of force:
            # 1. Harmonic Pull: A massive "spring" (10.0M kJ/mol/nm²) that treats the 
            #    termini like two magnets. It provides a global gradient that pulls 
            #    the structure toward closure.
            # 2. Hard Constraint: A specialized OpenMM constraint that FIXES the 
            #    distance at exactly 1.33 Angstroms. While the pull force gets us close, 
            #    the constraint ensures the final "weld" satisfies the perfect geometry 
            #    required by downstream NMR tools.
            if cyclic or added_bonds:
                pull_force = mm.CustomBondForce("0.5*k_pull*(r-r0)^2")
                pull_force.addGlobalParameter("k_pull", 10000000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
                pull_force.addPerBondParameter("r0")
                
                # 1. Head-to-Tail Pull (for cyclic peptides)
                if cyclic:
                    # Find N of first NON-CAP residue and C of last NON-CAP residue
                    n_idx, c_idx = -1, -1
                    residues = list(topology.residues())
                    real_residues = [r for r in residues if r.name not in ['ACE', 'NME']]
                    if real_residues:
                        r_first, r_last = real_residues[0], real_residues[-1]
                        for a in r_first.atoms():
                            if a.name == 'N': n_idx = a.index; break
                        for a in r_last.atoms():
                            if a.name == 'C': c_idx = a.index; break
                    
                    if n_idx != -1 and c_idx != -1:
                        # Target peptide bond length is ~1.33 Angstroms (0.133 nm)
                        pull_force.addBond(n_idx, c_idx, [0.133 * unit.nanometers])
                        # HARD CONSTRAINT: Force the distance to be exactly 1.33 A
                        system.addConstraint(n_idx, c_idx, 0.133 * unit.nanometers)
                        logger.info(f"Added hard cyclic constraint: {n_idx} -- {c_idx}")
                
                # 2. Sidechain-to-Sidechain Pull (for disulfides)
                if added_bonds:
                    for r1, r2 in added_bonds:
                        s1, s2 = -1, -1
                        for res in topology.residues():
                            if res.index == r1.index:
                                for a in res.atoms():
                                    if a.name == 'SG': s1 = a.index; break
                            if res.index == r2.index:
                                for a in res.atoms():
                                    if a.name == 'SG': s2 = a.index; break
                        if s1 != -1 and s2 != -1:
                            pull_force.addBond(s1, s2, [0.203 * unit.nanometers])
                
                if pull_force.getNumBonds() > 0:
                    system.addForce(pull_force)

            # 7. SIMULATION
            integrator = mm.LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtoseconds)
            platform = None; props = {}
            for name in ['CUDA', 'Metal', 'OpenCL']:
                try:
                    platform = mm.Platform.getPlatformByName(name)
                    if name in ['CUDA', 'OpenCL']: props = {'Precision': 'mixed'}
                    logger.info(f"Using OpenMM Platform: {name}")
                    break
                except Exception: continue
            
            if platform:
                try: simulation = app.Simulation(topology, system, integrator, platform, props)
                except Exception: platform = None
            if not platform: simulation = app.Simulation(topology, system, integrator)

            simulation.context.setPositions(positions)
            
            logger.info(f"Minimizing (Tolerance={tolerance} kJ/mol, MaxIter={max_iterations})...")
            if cyclic or added_bonds:
                # Macrocycles and disulfides need help closing. 
                # We use unlimited iterations (0) to ensure the "pull" forces fully converge.
                cyc_iter = 0 
                logger.info("Macrocycle/Disulfide Optimization: Running unlimited iterations for closure.")
                simulation.minimizeEnergy(maxIterations=cyc_iter, tolerance=(tolerance*0.1)*unit.kilojoule/(unit.mole*unit.nanometer))
                
                # Second pass with perturbation to break linear deadlocks
                if cyclic:
                    pos = simulation.context.getState(getPositions=True).getPositions()
                    import numpy as np
                    new_pos = pos + np.random.normal(0, 0.05, (len(pos), 3)) * unit.nanometers
                    simulation.context.setPositions(new_pos)
                    simulation.minimizeEnergy(maxIterations=cyc_iter, tolerance=(tolerance*0.1)*unit.kilojoule/(unit.mole*unit.nanometer))
            else:
                simulation.minimizeEnergy(maxIterations=max_iterations, tolerance=tolerance*unit.kilojoule/(unit.mole*unit.nanometer))
            
            # EDUCATIONAL NOTE - Thermal Equilibration (MD):
            # ----------------------------------------------
            # Minimization only finds a "Static Minimum" (0 Kelvin). 
            # Real proteins are dynamic. Running MD steps (Langevin Dynamics) 
            # allows the structure to "breathe" at 300K, resolving subtle 
            # clashes and satisfying entropy-driven structural preferences.
            if equilibration_steps > 0: 
                simulation.step(equilibration_steps)
                
            state = simulation.context.getState(getPositions=True)
            
            # EDUCATIONAL NOTE - Serialization:
            # ---------------------------------
            # (Note: This also handles Macrocycle Cleanup)
            # --------------------------------------------
            # After physics completes, we must "tidy up" our synthetic hack. 
            # We prune the "Shadow Caps" (ACE/NME) and any extra terminal hydration 
            # protons (H1, H2, H3, OXT) that modeller added. We rename the remaining 
            # amide proton to 'H' to satisfy canonical PDB naming. Finally, we 
            # project the original residue names and IDs back onto the physics-optimized 
            # coordinates, bridging the gap between molecular physics and structural metadata.
            with open(output_path, 'w') as f: 
                pos = state.getPositions()
                final_topology = simulation.topology
                final_positions = pos

                # Macrocycle Cleanup: Delete terminal atoms that are no longer needed
                if cyclic:
                    try:
                        logger.info("Cleaning up terminal atoms for cyclic peptide output...")
                        mod_modeller = app.Modeller(final_topology, final_positions)
                        residues = list(mod_modeller.topology.residues())
                        if residues:
                            # 1. Prune ACE/NME dummy residues (if present)
                            to_prune_caps = [a for r in residues if r.name in ['ACE', 'NME'] for a in r.atoms()]
                            if to_prune_caps:
                                mod_modeller.delete(to_prune_caps)
                                final_topology = mod_modeller.topology
                                final_positions = mod_modeller.positions
                                residues = list(final_topology.residues())

                            if residues:
                                res1, resN = residues[0], residues[-1]
                                to_prune = []
                                
                                # N-terminus: Keep ONLY the backbone H.
                                # Delete H1, H2, H3 and ensure one is named 'H'
                                n1 = next((a for a in res1.atoms() if a.name == 'N'), None)
                                if n1:
                                    h_on_n1 = [a for a in res1.atoms() if a.element is not None and a.element.symbol == 'H' and any(b.atom1 == n1 or b.atom2 == n1 for b in final_topology.bonds() if a == b.atom1 or a == b.atom2)]
                                    if len(h_on_n1) > 0:
                                        # Keep the one closest to standard 'H' name or just the first
                                        h_to_keep = next((a for a in h_on_n1 if a.name == 'H'), h_on_n1[0])
                                        to_prune.extend([a for a in h_on_n1 if a != h_to_keep])
                                        h_to_keep.name = 'H' # Canonical backbone name
                                
                                # C-terminus: Remove OXT
                                oxt = next((a for a in resN.atoms() if a.name == 'OXT'), None)
                                if oxt: to_prune.append(oxt)
                                
                                if to_prune:
                                    mod_modeller.delete(to_prune)
                                    final_topology = mod_modeller.topology
                                    final_positions = mod_modeller.positions
                                
                    except Exception as e:
                        logger.warning(f"Macrocycle cleanup failed: {e}")

                if len(final_positions) == 0:
                    logger.error("OpenMM returned empty positions! Topology might be corrupted.")
                    return False
                    
                # Restore original residue names AND IDS
                # Robust matching: match by original ID (resSeq) as preserved by OpenMM
                for res in final_topology.residues():
                    res_key = (str(res.id).strip(), res.chain.id)
                    if res_key in original_metadata:
                        res.name = original_metadata[res_key]["name"]
                        res.id = original_metadata[res_key]["id"]

                if added_bonds:
                    for s, (id1, id2) in enumerate(added_bonds, 1):
                        try:
                            f.write(f"SSBOND{s:4d} CYS A {int(id1):4d}    CYS A {int(id2):4d}                          \n")
                        except: pass
                
                # Export PDB with manual CONECT records to ensure visual 'bridges'
                import io
                pdb_buffer = io.StringIO()
                app.PDBFile.writeFile(final_topology, final_positions, pdb_buffer)
                pdb_lines = pdb_buffer.getvalue().split('\n')
                
                # Force CONECT for disulfides and coordination
                extra_conects = []
                # 1. Disulfides (from Topology)
                for bond in final_topology.bonds():
                    a1, a2 = bond.atom1, bond.atom2
                    if a1.name == 'SG' and a2.name == 'SG':
                        extra_conects.append((a1.index + 1, a2.index + 1))
                
                # 2. Metal Coordination (from restraints)
                for idx1, idx2 in coordination_restraints:
                    extra_conects.append((idx1 + 1, idx2 + 1))
                
                # Filter PDB and append CONECTs
                final_lines = []
                for line in pdb_lines:
                    if line.startswith('END') or line.startswith('CONECT'): continue
                    if line.strip(): final_lines.append(line)
                
                for i1, i2 in extra_conects:
                    final_lines.append(f"CONECT{i1:5d}{i2:5d}")
                    final_lines.append(f"CONECT{i2:5d}{i1:5d}")
                
                final_lines.append("END")
                f.write("\n".join(final_lines) + "\n")
            return True
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return False
