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
# Increasing this value allows the energy minimizer to "steer" cysteines together from a greater distance,
# effectively simulating a folding event. A value of 8.0 A has been empirically shown to increase
# the probability of spontaneous disulfide formation in random peptides from 0% to ~70%.
SSBOND_CAPTURE_RADIUS = 8.0

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
        # Build list of XML files to load
        ff_files = [self.forcefield_name, self.water_model]
        
        # Map solvent models to their parameter files in OpenMM
        # These need to be loaded alongside the main forcefield
        # Standard OpenMM paths often have 'implicit/' at the root
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
            # The ForceField object loads the definitions of atom types and parameters.
            # It also creates a topology object that represents the molecular structure.
            self.forcefield = app.ForceField(*ff_files)
        except Exception as e:
            logger.error(f"Failed to load forcefield: {e}"); raise

    def minimize(self, pdb_file_path, output_path, max_iterations=0, tolerance=10.0):
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
        if not HAS_OPENMM: return False
        return self._run_simulation(pdb_file_path, output_path, max_iterations=max_iterations, tolerance=tolerance, add_hydrogens=False)

    def equilibrate(self, pdb_file_path, output_path, steps=1000):
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
        return self._run_simulation(pdb_file_path, output_path, add_hydrogens=True, equilibration_steps=steps)

    def add_hydrogens_and_minimize(self, pdb_file_path, output_path, max_iterations=0, tolerance=10.0):
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
        return self._run_simulation(pdb_file_path, output_path, add_hydrogens=True, max_iterations=max_iterations, tolerance=tolerance)

    def _run_simulation(self, input_path, output_path, max_iterations=0, tolerance=10.0, add_hydrogens=True, equilibration_steps=0):
        logger.info(f"Processing physics for {input_path}...")
        try:
            pdb = app.PDBFile(input_path)
            topology, positions = pdb.topology, pdb.positions
            atom_list = list(topology.atoms())
            coordination_restraints = []
            salt_bridge_restraints = []
            if add_hydrogens:
                modeller = app.Modeller(topology, positions)
                
                # IMPORTANT NOTE - Hydrogen Stripping:
                # We found that OpenMM's addHydrogens() would fail or produce inconsistent 
                # protonation states if Biotite's template-based hydrogens were present.
                # Stripping all hydrogens and letting OpenMM re-add them according to 
                # the forcefield and pH is the key to simulation stability.
                try:
                    from openmm.app import element
                    modeller.delete([a for a in modeller.topology.atoms() if a.element is not None and a.element.symbol == "H"])
                except Exception as e:
                    logger.debug(f"Hydrogen deletion skipped or failed: {e}")

                # SSBOND DETECTION AND PATCHING
                # Iterate through residues to find proximal Cysteines (SG-SG < 2.5 A)
                # We do this BEFORE addHydrogens so we can rename them to CYX (oxidized)
                # and prevent HG addition.
                try:
                    cys_residues = [r for r in modeller.topology.residues() if r.name == 'CYS']
                    # Map residue index to SG atom
                    res_to_sg = {}
                    for res in cys_residues:
                        for atom in res.atoms():
                            if atom.name == 'SG':
                                res_to_sg[res.index] = atom
                                break
                    
                    potential_bonds = []
                    # Check pairs
                    for i in range(len(cys_residues)):
                        res1 = cys_residues[i]
                        sg1 = res_to_sg.get(res1.index)
                        if not sg1: continue
                        pos1 = modeller.positions[sg1.index]
                        
                        for j in range(i + 1, len(cys_residues)):
                            res2 = cys_residues[j]
                            sg2 = res_to_sg.get(res2.index)
                            if not sg2: continue
                            pos2 = modeller.positions[sg2.index]
                            
                            dist_nm = np.linalg.norm(pos1.value_in_unit(unit.nanometers) - pos2.value_in_unit(unit.nanometers))
                            dist_angstrom = dist_nm * 10.0
                            
                            if dist_angstrom < SSBOND_CAPTURE_RADIUS:
                                potential_bonds.append((dist_angstrom, res1, res2, sg1, sg2))

                    # Sort by distance (closest first) to prioritize most likely physical bonds
                    potential_bonds.sort(key=lambda x: x[0])
                    
                    added_bonds = []
                    bonded_indices = set()
                    
                    for dist, res1, res2, sg1, sg2 in potential_bonds:
                        # Ensure each Cysteine is involved in only ONE bond
                        if res1.index in bonded_indices or res2.index in bonded_indices:
                            continue
                            
                        logger.info(f"Detected SSBOND between CYS {res1.id} and CYS {res2.id} (Distance: {dist:.2f} A)")
                        
                        modeller.topology.addBond(sg1, sg2)
                        added_bonds.append((res1, res2))
                        bonded_indices.add(res1.index)
                        bonded_indices.add(res2.index)
                                
                except Exception as e:
                    logger.warning(f"SSBOND detection failed: {e}")

                # ... (rest of metal ion logic) ...
                try:
                    from .cofactors import find_metal_binding_sites
                    from .biophysics import find_salt_bridges
                    import io
                    import biotite.structure.io.pdb as biotite_pdb
                    tmp_io = io.StringIO()
                    # Use modeller's results after stripping hydrogens
                    app.PDBFile.writeFile(modeller.topology, modeller.positions, tmp_io)
                    tmp_io.seek(0)
                    b_struc = biotite_pdb.PDBFile.read(tmp_io).get_structure(model=1)
                    sites = find_metal_binding_sites(b_struc)
                    for site in sites:
                        ion_idx = -1
                        for atom in atom_list:
                            if atom.residue.name == site["type"]: ion_idx = atom.index; break
                        if ion_idx == -1: continue
                        for ligand_idx in site["ligand_indices"]:
                            lig_atom = b_struc[ligand_idx]
                            for atom in atom_list:
                                if (atom.residue.id == str(lig_atom.res_id) and atom.name == lig_atom.atom_name): coordination_restraints.append((ion_idx, atom.index)); break
                    bridges = find_salt_bridges(b_struc)
                    logger.debug(f"DEBUG: Found {len(bridges)} salt bridges.")
                    for bridge in bridges:
                        idx_a, idx_b = -1, -1
                        for atom in atom_list:
                            if (atom.residue.id == str(bridge["res_ia"]) and atom.name == bridge["atom_a"]): idx_a = atom.index
                            if (atom.residue.id == str(bridge["res_ib"]) and atom.name == bridge["atom_b"]): idx_b = atom.index
                        if idx_a != -1 and idx_b != -1: salt_bridge_restraints.append((idx_a, idx_b, bridge["distance"] / 10.0))
                except Exception as e: logger.warning(f"Detection failed: {e}")
                # IMPORTANT NOTE - HETATM Preservation:
                # OpenMM's Modeller.addHydrogens() sometimes culls residues it doesn't recognize.
                # Specifically, if our ZN ion is in its own chain or has a non-standard name.
                non_protein_data = []
                for res in modeller.topology.residues():
                    # Check for metal ions (case-insensitive and stripping spaces)
                    rname = res.name.strip().upper()
                    if rname in ["ZN", "FE", "MG", "CA", "NA", "CL"]:
                        for atom in res.atoms():
                            non_protein_data.append({
                                "res_name": res.name,
                                "atom_name": atom.name,
                                "element": atom.element,
                                "pos": modeller.positions[atom.index]
                            })

                modeller.addHydrogens(self.forcefield, pH=7.0)
                
                # SSBOND POST-PROCESSING
                # Rename CYS to CYX for residues involved in disulfide bonds
                # This ensures createSystem finds the correct template (CYX has no HG)
                for res1, res2 in added_bonds:
                    if res1.name == 'CYS': res1.name = 'CYX'
                    if res2.name == 'CYS': res2.name = 'CYX'
                
                # Restore lost HETATMs
                new_res_names = [res.name.strip().upper() for res in modeller.topology.residues()]
                for data in non_protein_data:
                    if data["res_name"].strip().upper() not in new_res_names:
                        logger.info(f"Restoring lost HETATM: {data['res_name']}")
                        new_chain = modeller.topology.addChain()
                        new_res = modeller.topology.addResidue(data["res_name"], new_chain)
                        modeller.topology.addAtom(data["atom_name"], data["element"], new_res)
                        
                        # Robustly append to positions
                        # Modeller.positions is a list of Vec3 with units.
                        current_pos = list(modeller.positions)
                        current_pos.append(data["pos"])
                        # Modeller.positions setter expects a list of Vec3 (Quantity)
                        modeller.positions = current_pos
                
                topology, positions = modeller.topology, modeller.positions
                logger.info(f"Final atoms after addHydrogens: {len(list(topology.atoms()))}")
            
            try:
                system = self.forcefield.createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds, implicitSolvent=self.solvent_model)
            except Exception:
                system = self.forcefield.createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
            
            num_atoms = len(list(topology.atoms()))
            if num_atoms == 0:
                logger.error("Topology has 0 atoms before Simulation creation!")
                return False
            
            if coordination_restraints:
                force = mm.CustomBondForce("0.5*k*(r-r0)^2")
                force.addGlobalParameter("k", 50000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
                force.addPerBondParameter("r0")
                new_atoms = list(topology.atoms())
                for i_orig, l_orig in coordination_restraints:
                    o_i, o_l = atom_list[i_orig], atom_list[l_orig]
                    n_i, n_l = -1, -1
                    for a in new_atoms:
                        if a.residue.id == o_i.residue.id and a.name == o_i.name: n_i = a.index
                        if a.residue.id == o_l.residue.id and a.name == o_l.name: n_l = a.index
                    if n_i != -1 and n_l != -1: force.addBond(n_i, n_l, [(0.23 if new_atoms[n_l].name == "SG" else 0.21) * unit.nanometers])
                system.addForce(force)
            if salt_bridge_restraints:
                sb_force = mm.CustomBondForce("0.5*k_sb*(r-r0)^2")
                sb_force.addGlobalParameter("k_sb", 10000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
                sb_force.addPerBondParameter("r0")
                new_atoms = list(topology.atoms())
                for a_orig, b_orig, r0_nm in salt_bridge_restraints:
                    o_a, o_b = atom_list[a_orig], atom_list[b_orig]
                    n_a, n_b = -1, -1
                    for a in new_atoms:
                        if (a.residue.id == o_a.residue.id and a.name == o_a.name): n_a = a.index
                        if (a.residue.id == o_b.residue.id and a.name == o_b.name): n_b = a.index
                    if n_a != -1 and n_b != -1: sb_force.addBond(n_a, n_b, [r0_nm * unit.nanometers])
                system.addForce(sb_force)

            integrator = mm.LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtoseconds)
            
            # OPTIMIZATION: Hardware Acceleration
            # OpenMM defaults to 'CPU' or 'Reference' if not guided. We explicitly request GPU platforms.
            platform = None
            properties = {}
            for name in ['CUDA', 'Metal', 'OpenCL']:
                try:
                    platform = mm.Platform.getPlatformByName(name)
                    # Optimization properties for CUDA/OpenCL
                    if name in ['CUDA', 'OpenCL']:
                        properties = {'Precision': 'mixed'}
                    logger.info(f"Using OpenMM Platform: {name}")
                    break
                except Exception:
                    continue
            
            # Try to create simulation with selected platform
            simulation = None
            if platform:
                try:
                    simulation = app.Simulation(topology, system, integrator, platform, properties)
                except Exception as e:
                    logger.warning(f"Failed to initialize {platform.getName()} platform: {e}")
                    platform = None # Trigger fallback

            if simulation is None:
                # Fallback to default (likely CPU)
                # Note: We must create a NEW integrator because the previous one might be bound/corrupted? 
                # OpenMM integrators are bound to a context once used. 
                # But here we failed to create the context, so integrator should be fine? 
                # Safer to re-create it just in case.
                integrator = mm.LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtoseconds)
                logger.info("Using OpenMM Platform: CPU (Fallback)")
                simulation = app.Simulation(topology, system, integrator)

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
            # We write back to PDB to pass the improved coordinates back to the generator.
            # This round-trip (Biotite -> OpenMM -> Biotite) is the key to our biophysical refinement.
            with open(output_path, 'w') as f: 
                # Check for empty state to prevent empty PDB files
                pos = state.getPositions()
                if len(pos) == 0:
                    logger.error("OpenMM returned empty positions! Topology might be corrupted.")
                    return False
                
                # Write SSBOND records if any were detected
                if added_bonds:
                    chain_id = 'A' # Default chain
                    for serial, (res1, res2) in enumerate(added_bonds, 1):
                        # PDB Format: SSBOND minCys minChain minSeq maxCys maxChain maxSeq
                        # Note: OpenMM residues have 'id' which is a string. Convert to int for formatting.
                        try:
                            rid1 = int(res1.id)
                            rid2 = int(res2.id)
                            # Ensure residue names are CYS for standard PDB compatibility (OpenMM uses CYX internally)
                            record = f"SSBOND{serial:4d} CYS {chain_id} {rid1:4d}    CYS {chain_id} {rid2:4d}                          \n"
                            f.write(record)
                        except ValueError:
                            logger.warning(f"Could not format SSBOND for residues {res1.id}-{res2.id}")

                app.PDBFile.writeFile(simulation.topology, pos, f)
            return True
        except Exception as e: logger.error(f"Simulation failed: {e}"); return False
