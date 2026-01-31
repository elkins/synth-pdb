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
        try:
            pdb = app.PDBFile(input_path)
            topology, positions = pdb.topology, pdb.positions
            
            # EDUCATIONAL NOTE - Topology Bridging (welding the ring):
            # --------------------------------------------------------
            # Standard PDB readers treat chains as linear (Res 1 -> Res N).
            # To simulate a cyclic peptide, we must explicitly tell the physics 
            # engine to create a covalent bond between the first and last residue.
            # Without this, the atoms might be close in space, but they won't 
            # "feel" each other's bond forces, and the ring will fly apart!
            if cyclic:
                residues = list(topology.residues())
                res_first, res_last = residues[0], residues[-1]
                c_atom, n_atom = None, None
                for a in res_last.atoms():
                    if a.name == 'C': c_atom = a; break
                for a in res_first.atoms():
                    if a.name == 'N': n_atom = a; break
                if c_atom and n_atom:
                    logger.info(f"Cyclizing peptide: Adding bond between {res_last.name}{res_last.id} C and {res_first.name}{res_first.id} N")
                    topology.addBond(c_atom, n_atom)
                else: logger.warning("Could not find N/C atoms for cyclization.")

            atom_list = list(topology.atoms())
            coordination_restraints = []
            salt_bridge_restraints = []

            # 1. MODELLER SETUP
            modeller = app.Modeller(topology, positions)
            
            # 2. HEURISTIC BACKBONE BONDING (Backbone Stitching)
            # -------------------------------------------------
            # EDUCATIONAL NOTE:
            # During automated construction (especially with PTMs or caps), 
            # standard PDB topology builders may fail to recognize the connectivity.
            # We use a distance-based heuristic (< 1.6 A) to "stitch" the chain.
            # This is a common technique in bio-simulation to fix topology "breaks".
            try:
                residues = list(modeller.topology.residues())
                existing_bonds = set(frozenset([b[0].index, b[1].index]) for b in modeller.topology.bonds())
                for i in range(len(residues) - 1):
                    res1, res2 = residues[i], residues[i+1]
                    c_s, n_s = None, None
                    for a in res1.atoms(): 
                        if a.name == 'C': c_s = a; break
                    for a in res2.atoms(): 
                        if a.name == 'N': n_s = a; break
                    if c_s and n_s and frozenset([c_s.index, n_s.index]) not in existing_bonds:
                        p1 = np.array(modeller.positions[c_s.index].value_in_unit(unit.angstroms))
                        p2 = np.array(modeller.positions[n_s.index].value_in_unit(unit.angstroms))
                        dist = np.sqrt(np.sum((p1 - p2)**2))
                        if dist < 1.6: modeller.topology.addBond(c_s, n_s)
            except Exception as e: logger.warning(f"Backbone stitching failed: {e}")

            # 3. IMPORTANT: PTM and Tautomer Reversion
            # OpenMM's standard forcefields (amber14-all) do not include templates for:
            # 1. Phosphorylated residues (SEP, TPO, PTR)
            # 2. Explicit histidine tautomers (HIE, HID, HIP) as INPUT names (it prefers HIS + addHydrogens)
            #
            # To prevent "No template found" errors, we revert these to standard residues
            # and let OpenMM's addHydrogens(pH=7.0) re-assign protonation states physically.
            # For PTMs, we unfortunately lose the phosphate group during minimization (limit of standard FF),
            # but this is better than crashing.
            ptm_map = {
                'SEP': 'SER', 'TPO': 'THR', 'PTR': 'TYR',
                'HIE': 'HIS', 'HID': 'HIS', 'HIP': 'HIS'
            }
            
            # We reuse the Modeller to rename non-standard residues and delete extra atoms (Phosphates)
            # BEFORE hydrogen stripping.
            atoms_to_delete = []
            ptm_atom_names = ["P", "O1P", "O2P", "O3P"] 
            for res in modeller.topology.residues():
                if res.name in ptm_map:
                    original = res.name; res.name = ptm_map[res.name]
                    if original in ['SEP', 'TPO', 'PTR']:
                        for atom in res.atoms():
                            if atom.name in ptm_atom_names: atoms_to_delete.append(atom)
            if atoms_to_delete: modeller.delete(atoms_to_delete)
            
            # 4. HYDROGEN STRIPPING (Conditional)
            # For linear peptides, we strip H to let modeller re-add them at pH 7.0.
            # For cyclic peptides, we prefer to keep them if they exist to avoid terminal issues.
            added_bonds = []
            if not cyclic and add_hydrogens:
                try:
                    modeller.delete([a for a in modeller.topology.atoms() if a.element is not None and a.element.symbol == "H"])
                except Exception as e: logger.debug(f"H deletion failed: {e}")

            # 5. BIOPHYSICAL CONSTRAINT DETECTION
            # ------------------------------------
            
            # SSBOND Detection (Internal OpenMM Topology)
            try:
                cys_residues = [r for r in modeller.topology.residues() if r.name == 'CYS']
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
                    modeller.topology.addBond(s1, s2); added_bonds.append((r1, r2))
                    bonded_indices.add(r1.index); bonded_indices.add(r2.index)
            except Exception as e: logger.warning(f"SSBOND failed: {e}")

            # Meta/Salt Bridge Detection (Requires Biotite helper)
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
                                if (atom.residue.id == str(l_at.res_id) and atom.name == l_at.atom_name): coordination_restraints.append((i_idx, atom.index)); break
                
                bridges = find_salt_bridges(b_struc, cutoff=6.0)
                logger.debug(f"DEBUG: Found {len(bridges)} salt bridges.")
                for br in bridges:
                    ia, ib = -1, -1
                    for atom in atom_list:
                        if (atom.residue.id == str(br["res_ia"]) and atom.name == br["atom_a"]): ia = atom.index
                        if (atom.residue.id == str(br["res_ib"]) and atom.name == br["atom_b"]): ib = atom.index
                    if ia != -1 and ib != -1: salt_bridge_restraints.append((ia, ib, br["distance"] / 10.0))
            except Exception as e: logger.warning(f"Metadata/SaltBridge detection failed: {e}")

            # 6. HYDROGEN ADDITION / HETATM RESTORATION (Linear only)
            if not cyclic and add_hydrogens:
                non_protein_data = [] 
                for r in modeller.topology.residues():
                    if r.name.strip().upper() in ["ZN", "FE", "MG", "CA", "NA", "CL"]:
                        for a in r.atoms(): non_protein_data.append({"res_name": r.name, "atom_name": a.name, "element": a.element, "pos": modeller.positions[a.index]})
                
                modeller.addHydrogens(self.forcefield, pH=7.0)
                
                # Restore lost HETATMs (Modeller.addHydrogens sometimes deletes them)
                new_names = [res.name.strip().upper() for res in modeller.topology.residues()]
                for d in non_protein_data:
                    if d["res_name"].strip().upper() not in new_names:
                        logger.info(f"Restoring lost HETATM: {d['res_name']}")
                        nc = modeller.topology.addChain(); nr = modeller.topology.addResidue(d["res_name"], nc)
                        modeller.topology.addAtom(d["atom_name"], d["element"], nr)
                        modeller.positions = list(modeller.positions) + [d["pos"]]
            
            # 7. CYX RENAME (Always for disulfides)
            # ------------------------------------
            # OpenMM's ForceField requires disulfide-bonded cysteines to be named 'CYX'
            # to match the covalent-bonded template (instead of standard 'CYS').
            # We also MUST delete the thiol hydrogen (HG) for the template to match.
            if added_bonds:
                hg_to_delete = []
                for r1, r2 in added_bonds:
                    for res in [r1, r2]:
                        if res.name == 'CYS':
                            res.name = 'CYX'
                            hg_to_delete.extend([a for a in res.atoms() if a.name == 'HG'])
                if hg_to_delete:
                    modeller.delete(hg_to_delete)
            
            topology, positions = modeller.topology, modeller.positions
            if cyclic: logger.info("Using pre-existing hydrogens for cyclic peptide.")

            # 5. SYSTEM CREATION
            try:
                system = self.forcefield.createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds, implicitSolvent=self.solvent_model)
            except Exception:
                system = self.forcefield.createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
            
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

            # 7. SIMULATION
            integrator = mm.LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtoseconds)
            # OPTIMIZATION: Hardware Acceleration
            # OpenMM defaults to 'CPU' or 'Reference' if not guided. We explicitly request GPU platforms.
            platform = None; props = {}
            for name in ['CUDA', 'Metal', 'OpenCL']:
                try:
                    platform = mm.Platform.getPlatformByName(name)
                    # Optimization properties for CUDA/OpenCL
                    if name in ['CUDA', 'OpenCL']: props = {'Precision': 'mixed'}
                    logger.info(f"Using OpenMM Platform: {name}")
                    break
                except Exception: continue
            
            if platform:
                try: simulation = app.Simulation(topology, system, integrator, platform, props)
                except Exception: platform = None
            if not platform: simulation = app.Simulation(topology, system, integrator)

            simulation.context.setPositions(positions)
            
            # IMPORTANT NOTE:
            # Energy minimization alone isn't enough for structural stability.
            # We must also equilibrate (simulation.step) to let the structure settle
            # into a local minimum that respects the custom coordination restraints.
            logger.info(f"Minimizing (Tolerance={tolerance} kJ/mol, MaxIter={max_iterations})...")
            simulation.minimizeEnergy(maxIterations=max_iterations, tolerance=tolerance*unit.kilojoule/(unit.mole*unit.nanometer))
            
            if equilibration_steps > 0: 
                logger.info(f"Running {equilibration_steps} steps of equilibration...")
                simulation.step(equilibration_steps)
                
            state = simulation.context.getState(getPositions=True)
            
            # EDUCATIONAL NOTE - Serialization:
            # We write back to PDB to pass the improved coordinates and connectivity
            # (including newly added SSBOND or cyclic bonds) to the next stage.
            with open(output_path, 'w') as f: 
                pos = state.getPositions()
                if len(pos) == 0:
                    logger.error("OpenMM returned empty positions! Topology might be corrupted.")
                    return False
                if added_bonds:
                    for s, (r1, r2) in enumerate(added_bonds, 1):
                        f.write(f"SSBOND{s:4d} CYS A {int(r1.id):4d}    CYS A {int(r2.id):4d}                          \n")
                app.PDBFile.writeFile(simulation.topology, pos, f)
            return True
        except Exception as e: logger.error(f"Simulation failed: {e}"); return False
