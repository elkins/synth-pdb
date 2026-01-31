"""
3D Molecular Viewer for synth-pdb.

Opens generated PDB structures in browser-based 3D viewer using 3Dmol.js.
Based on pdbstat's molecular viewer implementation.
"""

import logging
import tempfile
import webbrowser
from pathlib import Path
import io
import traceback
import numpy as np
import biotite.structure as struc

import biotite.structure.io.pdb as pdb
import biotite.structure.hbond as hbond



logger = logging.getLogger(__name__)


def view_structure_in_browser(
    pdb_content: str,
    filename: str = "structure.pdb",
    style: str = "cartoon",
    color: str = "spectrum",
    restraints: list = None,
    highlights: list = None,
    show_hbonds: bool = True,

) -> None:
    """
    Open 3D structure viewer in browser.
    
    Args:
        pdb_content: PDB file contents as string
        filename: Name to display in viewer title
        style: Initial representation style (cartoon/stick/sphere/line)
        color: Initial color scheme (spectrum/chain/ss/white)
        restraints: Optional list of restraint dicts to visualize
        
    Example:
        >>> pdb = generate_pdb_content(length=20)
        >>> view_structure_in_browser(pdb, "my_peptide.pdb")
        
    Raises:
        Exception: If viewer fails to open
    """
    try:
        logger.info(f"Opening 3D viewer for {filename}")
        
        # Generate HTML with embedded 3Dmol.js viewer
        html = _create_3dmol_html(pdb_content, filename, style, color, restraints, highlights, show_hbonds)
        
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            temp_path = f.name
        
        # Open in default browser
        webbrowser.open(f"file://{temp_path}")
        
        logger.info(f"3D viewer opened in browser: {temp_path}")
        
    except Exception as e:
        logger.error(f"Failed to open 3D viewer: {e}")
        raise


def _create_3dmol_html(
    pdb_data: str, 
    filename: str, 
    style: str, 
    color: str, 
    restraints: list = None, 
    highlights: list = None,
    show_hbonds: bool = False
) -> str:
    """
    Generate HTML with embedded 3Dmol.js viewer.
    
    EDUCATIONAL NOTE - Why Browser-Based Visualization:
    Browser-based viewers are ideal for quick structure inspection because:
    1. No installation required (works on any system with a browser)
    2. Interactive (rotate, zoom, change styles)
    3. Lightweight (uses 3Dmol.js JavaScript library)
    4. Shareable (can save HTML file and share with others)
    
    EDUCATIONAL NOTE - 3Dmol.js:
    3Dmol.js is a JavaScript library for molecular visualization that:
    - Runs entirely in the browser (no server needed)
    - Supports PDB, SDF, MOL2, and other formats
    - Provides interactive controls (rotate, zoom, style changes)
    - Uses WebGL for hardware-accelerated 3D graphics
    
    Args:
        pdb_data: PDB file contents
        filename: Name of PDB file for display
        style: Representation style (cartoon/stick/sphere/line)
        color: Color scheme (spectrum/chain/ss/white)
        color: Color scheme (spectrum/chain/ss/white)
        restraints: Optional list of restraint dicts to visualize as cylinders
        highlights: Optional list of dicts {'start', 'end', 'color', 'label'} for secondary structure
        show_hbonds: Whether to calculate and show backbone H-bonds

        
    Returns:
        Complete HTML document as string
    """
    # Identify max residue for backbone bridging
    res_ids = []
    for line in pdb_data.splitlines():
        if line.startswith("ATOM"):
            try:
                rid = int(line[22:26].strip())
                res_ids.append(rid)
            except: continue
    max_res = max(res_ids) if res_ids else 1

    # Escape PDB data for JavaScript
    # We escape backslashes first, then backticks (used in template literals),
    # and finally $ (to avoid interpolation issues with ${...})
    pdb_escaped = pdb_data.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    
    # Calculate H-bonds if requested
    hbond_cmds = ""
    if show_hbonds:
        hbonds = _find_hbonds(pdb_data)
        if hbonds:
             hbond_cmds += "/* Detected Backbone H-Bonds */\n"
             for hb in hbonds:
                 # Add dashed yellow cylinder
                 # We use start_resi and end_resi
                 # Note: JS indices for arrays are 0-based but 3Dmol resi selection is 1-based (PDB numbering)
                 
                 # 3Dmol selection needs explicit atom names for start/end
                 start_sel = f"{{chain:'A', resi:{hb['start_resi']}, atom:'{hb['start_atom']}'}}"
                 end_sel = f"{{chain:'A', resi:{hb['end_resi']}, atom:'{hb['end_atom']}'}}"
                 
                 # JS logic to find atoms and draw line
                 # We wrap in a block to reuse variable names safely
                 hbond_cmds += f"""
                 {{
                     let sel1 = {start_sel};
                     let sel2 = {end_sel};
                     let atoms1 = viewer.selectedAtoms(sel1);
                     let atoms2 = viewer.selectedAtoms(sel2);
                     if(atoms1.length > 0 && atoms2.length > 0) {{
                         viewer.addLine({{
                             start: {{x: atoms1[0].x, y: atoms1[0].y, z: atoms1[0].z}},
                             end:   {{x: atoms2[0].x, y: atoms2[0].y, z: atoms2[0].z}},
                             color: 'magenta', /* High contrast vs white background */
                             dashed: true,
                             linewidth: 10,  /* Doubled width for better visibility */
                             opacity: 1.0    /* Full opacity */
                         }});
                     }}
                 }}
                 """

    # Generate SSBOND Visualization Logic
    # We parse the PDB again (lightweight) to find SSBOND records
    ssbond_cmds = ""
    ssbonds = _find_ssbonds(pdb_data)
    if ssbonds:
        ssbond_cmds += "/* Detected Disulfide Bonds */\n"
        for ss in ssbonds:
             # Add thick yellow cylinder between SG atoms
             
             # 3Dmol selection
             start_sel = f"{{chain:'{ss['c1']}', resi:{ss['r1']}, atom:'SG'}}"
             end_sel = f"{{chain:'{ss['c2']}', resi:{ss['r2']}, atom:'SG'}}"
             
             ssbond_cmds += f"""
             {{
                 let sel1 = {start_sel};
                 let sel2 = {end_sel};
                 let atoms1 = viewer.selectedAtoms(sel1);
                 let atoms2 = viewer.selectedAtoms(sel2);
                 if(atoms1.length > 0 && atoms2.length > 0) {{
                     viewer.addCylinder({{
                         start: {{x: atoms1[0].x, y: atoms1[0].y, z: atoms1[0].z}},
                         end:   {{x: atoms2[0].x, y: atoms2[0].y, z: atoms2[0].z}},
                         radius: 0.15,
                         color: 'yellow', /* Conventional disulfide color */
                         fromCap: 1, toCap: 1,
                         opacity: 1.0
                     }});
                     
                     /* Force stick visibility for these residues so the bond doesn't float in space */
                     viewer.addStyle({{chain: sel1.chain, resi: sel1.resi}}, {{stick:{{radius:0.2}}}});
                     viewer.addStyle({{chain: sel2.chain, resi: sel2.resi}}, {{stick:{{radius:0.2}}}});
                 }}
             }}
             """

    # Generate CONECT Visualization Logic
    # This bridges the visual gap for cyclic peptides and other extra bonds
    conect_cmds = ""
    conects = _find_conects(pdb_data)
    if conects:
        conect_cmds += "/* Detected CONECT Records (Cyclic/Extra Bonds) */\n"
        
        # Bridge the gap: style terminal backbones as sticks so they meet the ribbon
        # Ribbon ends at CA. We need path: [Ribbon End CA] --stick-- [N or C] --thick cylinder-- [...]
        # Style entire first and last residues as sticks to be robust
        conect_cmds += f"viewer.addStyle({{chain:'A', resi:1}}, {{stick:{{radius:0.18, color:'cyan'}}}});\n"
        conect_cmds += f"viewer.addStyle({{chain:'A', resi:{max_res}}}, {{stick:{{radius:0.18, color:'cyan'}}}});\n"
        
        # Explicitly ensure CA-C and N-CA path is visible as cyan sticks for contrast
        conect_cmds += f"viewer.addStyle({{chain:'A', resi:1, atom:['N','CA']}}, {{stick:{{radius:0.25, color:'cyan'}}}});\n"
        conect_cmds += f"viewer.addStyle({{chain:'A', resi:{max_res}, atom:['CA','C']}}, {{stick:{{radius:0.25, color:'cyan'}}}});\n"

        for s1, s2 in conects:
             conect_cmds += f"""
             {{
                 let atoms1 = viewer.selectedAtoms({{serial:{s1}}});
                 let atoms2 = viewer.selectedAtoms({{serial:{s2}}});
                 if(atoms1.length > 0 && atoms2.length > 0) {{
                     viewer.addCylinder({{
                         start: {{x: atoms1[0].x, y: atoms1[0].y, z: atoms1[0].z}},
                         end:   {{x: atoms2[0].x, y: atoms2[0].y, z: atoms2[0].z}},
                         radius: 0.25, /* Thicker for definitive closure */
                         color: 'cyan', 
                         fromCap: 1, toCap: 1,
                         opacity: 1.0
                     }});
                 }}
             }}
             """

    # Generate Highlights JS
    highlight_cmds = ""
    if highlights:
         for h in highlights:
             start = h['start']
             end = h['end']
             h_color = h.get('color', 'purple')
             h_style = h.get('style', 'stick')
             h_label = h.get('label', '')
             
             # Create array of residues [3, 4, 5, 6]
             res_array = str(list(range(start, end + 1)))
             
             # Add style override
             if h_style == 'stick':
                 highlight_cmds += f"viewer.addStyle({{chain:'A', resi:{res_array}}}, {{stick:{{colorscheme:'{h_color}', radius:0.2}}}});\n"
             elif h_style == 'cartoon':
                 highlight_cmds += f"viewer.addStyle({{chain:'A', resi:{res_array}}}, {{cartoon:{{color:'{h_color}'}}}});\n"
             
             # Add label at center residue
             center_res = (start + end) // 2
             if h_label:
                 highlight_cmds += f"viewer.addLabel('{h_label}', {{fontSize:12, fontColor:'white', backgroundColor:'black', backgroundOpacity:0.7}}, {{chain:'A', resi:{center_res}, atom:'CA'}});\n"

    # Generate PTM labels logic
    # We essentially grep the PDB for SEP/TPO/PTR and add labels
    ptm_cmds = ""
    # Python-side coarse detection for optimization, but JS will do the heavy lifting
    if "SEP" in pdb_escaped or "TPO" in pdb_escaped or "PTR" in pdb_escaped:
        ptm_cmds += """
        /* PTM Visualization Logic with Fallback for Minimized Structures */
        /* If P atom exists (un-minimized), put label on P. */
        /* If P atom missing (minimized/stripped), put label on sidechain oxygen/carbon. */
        
        /* SEP */
        viewer.addLabel("SEP", {fontSize:10, fontColor:'black', backgroundColor:'orange'}, {resn:'SEP', atom:'P'});
        viewer.addLabel("SEP", {fontSize:10, fontColor:'black', backgroundColor:'orange'}, {resn:'SEP', atom:'OG', not:{atom:'P'}}); /* Fallback */
        
        /* TPO */
        viewer.addLabel("TPO", {fontSize:10, fontColor:'black', backgroundColor:'orange'}, {resn:'TPO', atom:'P'});
        viewer.addLabel("TPO", {fontSize:10, fontColor:'black', backgroundColor:'orange'}, {resn:'TPO', atom:'OG1', not:{atom:'P'}}); /* Fallback */
        
        /* PTR */
        viewer.addLabel("PTR", {fontSize:10, fontColor:'black', backgroundColor:'orange'}, {resn:'PTR', atom:'P'});
        viewer.addLabel("PTR", {fontSize:10, fontColor:'black', backgroundColor:'orange'}, {resn:'PTR', atom:'OH', not:{atom:'P'}}); /* Fallback */

        /* Ensure PTMs serve as spheres too */
        viewer.addStyle({resn:['SEP','TPO','PTR']}, {stick:{radius:0.2}});
        
        /* Orange Sphere on P if present */
        viewer.addStyle({resn:['SEP','TPO','PTR'], atom:'P'}, {sphere:{radius:1.0, color:'orange'}});
        
        /* Orange Sphere on Oxygen if P missing (to simulate modified state) */
        /* Note: This might overlap with normal oxygen, but creates the visual cue requested by user */
        /* SEP -> OG */
        viewer.addStyle({resn:'SEP', atom:'OG'}, {sphere:{radius:0.8, color:'orange', opacity:0.7}});
        /* TPO -> OG1 */
        viewer.addStyle({resn:'TPO', atom:'OG1'}, {sphere:{radius:0.8, color:'orange', opacity:0.7}});
        /* PTR -> OH */
        viewer.addStyle({resn:'PTR', atom:'OH'}, {sphere:{radius:0.8, color:'orange', opacity:0.7}});
        """

    # Serialize restraints to JSON-like logic in JS
    js_restraints = ""

    if restraints:
        js_restraints = "let restraints = [\n"
        for r in restraints:
            # Handle key variations
            c1 = r.get('chain_1', 'A')
            s1 = r.get('seq_1', r.get('residue_index_1'))
            a1 = r.get('atom_1', r.get('atom_name_1'))
            c2 = r.get('chain_2', 'A')
            s2 = r.get('seq_2', r.get('residue_index_2'))
            a2 = r.get('atom_2', r.get('atom_name_2'))
            dist = r.get('dist', r.get('actual_distance', 5.0))
            
            if s1 and s2:
                js_restraints += f"    {{ c1:'{c1}', s1:{s1}, a1:'{a1}', c2:'{c2}', s2:{s2}, a2:'{a2}', d:{dist} }},\n"
        js_restraints += "];\n"
        
        js_restraints += """
            // Add cylinders for restraints
            // EDUCATIONAL NOTE:
            // NMR Short-Range Restraints (NOEs) roughly depend on 1/r^6.
            // This means we only see protons that are very close (< 5-6 Angstroms).
            // We visualize these as red bonds to show which atoms are "talking" to each other via the magnetic field.
            try {
                if (typeof restraints !== 'undefined') {
                    if (restraints.length === 0) {
                        alert("Warning: 0 restraints found. No lines will be drawn.");
                    } else {
                        console.log("Visualizing " + restraints.length + " restraints.");
                        let drawnCount = 0;
                        for(let i=0; i<restraints.length; i++) {
                            let r = restraints[i];
                            
                            // Robustness: Trim strings and try fallback if chain is missing
                            let c1 = r.c1 ? r.c1.trim() : '';
                            let c2 = r.c2 ? r.c2.trim() : '';
                            let a1 = r.a1.trim();
                            let a2 = r.a2.trim();

                            // Try Strict Selection first
                            let sel1 = {chain: c1, resi: r.s1, atom: a1};
                            let sel2 = {chain: c2, resi: r.s2, atom: a2};
                            
                            let atoms1 = viewer.selectedAtoms(sel1);
                            let atoms2 = viewer.selectedAtoms(sel2);

                            // Fallback: Try without chain if strictly failed
                            if (atoms1.length === 0) {
                                atoms1 = viewer.selectedAtoms({resi: r.s1, atom: a1});
                            }
                            if (atoms2.length === 0) {
                                atoms2 = viewer.selectedAtoms({resi: r.s2, atom: a2});
                            }
                            
                            if(atoms1 && atoms1.length > 0 && atoms2 && atoms2.length > 0) {
                                let atom1 = atoms1[0];
                                let atom2 = atoms2[0];
                                
                                viewer.addCylinder({
                                    start: {x: atom1.x, y: atom1.y, z: atom1.z},
                                    end:   {x: atom2.x, y: atom2.y, z: atom2.z},
                                    radius: 0.05,
                                    color: 'red',
                                    fromCap: 1, toCap: 1,
                                    opacity: 0.4
                                });
                                drawnCount++;
                            }
                        }
                        if (drawnCount === 0) {
                            alert("Warning: Restraints exist (" + restraints.length + ") but no matching atoms found in structure.\\nCheck chain/resi/atom names.");
                        }
                    }
                }
            } catch (e) {
                console.error("Error visualizing restraints:", e);
            }
        """
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>3D Viewer - {filename}</title>
    <!-- Try to load 3Dmol.js from CDN -->
    <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
    <!-- Fallback CDN in case pitt.edu is down/blocked -->
    <script>
        if (typeof $3Dmol === 'undefined') {{
            document.write('<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"><\\/script>');
        }}
    </script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: #f5f5f5;
        }}
        #header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        #header h1 {{
            margin: 0 0 10px 0;
            font-size: 28px;
            font-weight: 600;
        }}
        #header p {{
            margin: 0;
            opacity: 0.9;
            font-size: 14px;
        }}
        #controls {{
            background: white;
            padding: 20px;
            border-bottom: 1px solid #e0e0e0;
            display: flex;
            gap: 30px;
            align-items: center;
            flex-wrap: wrap;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        .control-group {{
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        label {{
            font-weight: 600;
            color: #333;
            font-size: 14px;
        }}
        button {{
            padding: 10px 18px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s;
            box-shadow: 0 2px 4px rgba(102, 126, 234, 0.2);
        }}
        button:hover {{
            background: #5568d3;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
        }}
        button.active {{
            background: #10b981;
            box-shadow: 0 2px 4px rgba(16, 185, 129, 0.2);
        }}
        button.active:hover {{
            background: #059669;
        }}
        #viewer {{
            width: 100%;
            height: calc(100vh - 200px);
            position: relative;
            background: white;
        }}
        #instructions {{
            background: #f9fafb;
            padding: 12px;
            text-align: center;
            color: #6b7280;
            font-size: 13px;
            border-top: 1px solid #e0e0e0;
        }}
        .emoji {{
            font-size: 16px;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>🧬 synth-pdb 3D Molecular Viewer</h1>
        <p>{filename}</p>
    </div>

    <div id="controls">
        <div class="control-group">
            <label>Style:</label>
            <button id="btn-cartoon" onclick="setStyle('cartoon')">Cartoon</button>
            <button id="btn-stick" onclick="setStyle('stick')">Stick</button>
            <button id="btn-sphere" onclick="setStyle('sphere')">Sphere</button>
            <button id="btn-line" onclick="setStyle('line')">Line</button>
        </div>

        <div class="control-group">
            <label>Color:</label>
            <button id="color-spectrum" onclick="setColor('spectrum')">Spectrum</button>
            <button id="color-chain" onclick="setColor('chain')">Chain</button>
            <button id="color-ss" onclick="setColor('ss')">Secondary Structure</button>
            <button id="color-white" onclick="setColor('white')">White</button>
        </div>

        <div class="control-group">
            <label>Options:</label>
            <button id="btn-ghost" onclick="toggleGhost()">👻 Ghost Mode</button>
            <button id="btn-restraints" onclick="toggleRestraints()">🔴 Restraints</button>
        </div>

        <div class="control-group">
            <button onclick="resetView()">🔄 Reset View</button>
            <button onclick="toggleSpin()">🔄 Toggle Spin</button>
        </div>
    </div>

    <div id="viewer">
        <div id="loading-msg" style="color: #6b7280; font-style: italic;">
            🚀 Initializing 3D Engine...
        </div>
        <div id="error-overlay">
            <h3 style="margin-top:0">⚠️ Viewer Error</h3>
            <p id="error-text">Unexpected error</p>
            <p style="font-size: 12px; margin-bottom:0">Check your internet connection or try a different browser.</p>
        </div>
    </div>

    <div id="instructions">
        <span class="emoji">🖱️</span> Left-click + drag to rotate | Scroll to zoom | Right-click + drag to pan
    </div>

    <script>
        let viewer;
        let currentStyle = '{style}';
        let currentColor = '{color}';
        let spinning = false;
        let ghostMode = false;
        let showRestraints = true;
        let allShapes = []; // Store shape objects to toggle them

        // Initialize viewer when page loads
        window.addEventListener('load', function() {{
            const loadingMsg = document.getElementById('loading-msg');
            const errorOverlay = document.getElementById('error-overlay');
            const errorText = document.getElementById('error-text');

            try {{
                if (typeof $3Dmol === 'undefined') {{
                    throw new Error("3Dmol.js library failed to load. Please check your internet connection.");
                }}

                let element = document.getElementById('viewer');
                let config = {{ backgroundColor: 'white' }};
                viewer = $3Dmol.createViewer(element, config);

                // Load PDB data
                let pdbData = `{pdb_escaped}`;
                
                if (!pdbData || pdbData.trim().length === 0) {{
                    throw new Error("No PDB data provided to viewer.");
                }}

                // IMPORTANT: keepH: true is required to visualize NMR restraints involving protons!
                viewer.addModel(pdbData, "pdb", {{keepH: true}});

                // Set initial style and render
                applyStyle();
                
                // Draw initial restraints
                drawRestraints();
                
                viewer.zoomTo();
                viewer.render();
                
                // Hide loading message if we got this far
                if (loadingMsg) loadingMsg.style.display = 'none';

                // Highlight active buttons
                updateActiveButtons();
                console.log("3D Viewer initialized successfully.");

            }} catch (err) {{
                console.error("3D Viewer Initialization Failed:", err);
                if (loadingMsg) loadingMsg.style.display = 'none';
                if (errorOverlay) {{
                    errorOverlay.style.display = 'block';
                    errorText.innerText = err.message;
                }}
            }}
        }});
        
        // Define Restraints Data
        {js_restraints}

        function drawRestraints() {{
            // Remove previously tracked restraint shapes only
            if (typeof allShapes !== 'undefined' && allShapes.length > 0) {{
                for(let i=0; i<allShapes.length; i++) {{
                    viewer.removeShape(allShapes[i]);
                }}
            }}
            allShapes = [];

            if (!showRestraints) return;

            // Restraint drawing logic moved here
            // EDUCATIONAL NOTE:
            // NMR Short-Range Restraints (NOEs) roughly depend on 1/r^6.
            // This means we only see protons that are very close (< 5-6 Angstroms).
            try {{
                if (typeof restraints !== 'undefined') {{
                    if (restraints.length === 0) {{
                        // checking purely for debug, usually we just don't draw
                    }} else {{
                        console.log("Visualizing " + restraints.length + " restraints.");
                        for(let i=0; i<restraints.length; i++) {{
                            let r = restraints[i];
                            let c1 = r.c1 ? r.c1.trim() : '';
                            let c2 = r.c2 ? r.c2.trim() : '';
                            let a1 = r.a1.trim();
                            let a2 = r.a2.trim();

                            let sel1 = {{chain: c1, resi: r.s1, atom: a1}};
                            let sel2 = {{chain: c2, resi: r.s2, atom: a2}};
                            
                            let atoms1 = viewer.selectedAtoms(sel1);
                            let atoms2 = viewer.selectedAtoms(sel2);

                            if (atoms1.length === 0) atoms1 = viewer.selectedAtoms({{resi: r.s1, atom: a1}});
                            if (atoms2.length === 0) atoms2 = viewer.selectedAtoms({{resi: r.s2, atom: a2}});
                            
                            if(atoms1 && atoms1.length > 0 && atoms2 && atoms2.length > 0) {{
                                let atom1 = atoms1[0];
                                let atom2 = atoms2[0];
                                
                                let shape = viewer.addCylinder({{
                                    start: {{x: atom1.x, y: atom1.y, z: atom1.z}},
                                    end:   {{x: atom2.x, y: atom2.y, z: atom2.z}},
                                    radius: 0.1, // Slightly thicker for visibility
                                    color: 'red',
                                    fromCap: 1, toCap: 1,
                                    opacity: 0.6
                                }});
                                allShapes.push(shape);
                            }}
                        }}
                    }}
                }}
            }} catch (e) {{
                console.error("Error visualizing restraints:", e);
            }}
        }}

        function applyStyle() {{
            viewer.removeAllShapes(); // Clear all shapes (H-bonds, arrows, etc)
            viewer.removeAllLabels(); // Clear all labels
            allShapes = []; // Reset tracked restraints
            
            viewer.setStyle({{}}, {{}}); // Clear style

            let styleObj = {{}};
            let opacityVal = ghostMode ? 0.4 : 1.0; // Ghost mode opacity

            if (currentStyle === 'cartoon') {{
                styleObj.cartoon = {{ color: currentColor, opacity: opacityVal }};
            }} else if (currentStyle === 'stick') {{
                styleObj.stick = {{ colorscheme: currentColor, opacity: opacityVal, radius: 0.2 }}; // thinner sticks
            }} else if (currentStyle === 'sphere') {{
                styleObj.sphere = {{ colorscheme: currentColor, opacity: opacityVal }};
            }} else if (currentStyle === 'line') {{
                styleObj.line = {{ colorscheme: currentColor, opacity: opacityVal }};
            }}

            // Apply main style to everything first
            viewer.setStyle({{}}, styleObj);

            // ALWAYS render HETATMs (like Zinc) as spheres so they don't disappear in cartoon mode
            // We use multiple selectors (element and resn) to be as robust as possible
            // Note: We escape braces for Python f-string
            viewer.addStyle({{element: 'Zn'}}, {{sphere: {{radius: 1.2, color: 'silver'}}}});
            viewer.addStyle({{resn: 'ZN'}}, {{sphere: {{radius: 1.2, color: 'silver'}}}});
            viewer.addStyle({{hetatom: true}}, {{sphere: {{radius: 1.2, color: 'silver'}}}});

            // Apply Custom Highlights (Beta Turns, PTMs, H-bonds)
            // These are injected from Python
            {highlight_cmds}
            {ptm_cmds}
            {hbond_cmds}
            {ssbond_cmds}
            {conect_cmds}
            
            drawRestraints(); // Re-draw restraints (if enabled)
            
            viewer.render();
        }}


        function setStyle(style) {{
            currentStyle = style;
            applyStyle();
            updateActiveButtons();
        }}

        function setColor(color) {{
            currentColor = color;
            applyStyle();
            updateActiveButtons();
        }}

        function toggleGhost() {{
            ghostMode = !ghostMode;
            applyStyle();
            updateActiveButtons();
        }}

        function toggleRestraints() {{
            showRestraints = !showRestraints;
            drawRestraints();
            viewer.render();
            updateActiveButtons();
        }}

        function resetView() {{
            viewer.zoomTo();
            viewer.render();
        }}

        function toggleSpin() {{
            if (spinning) {{
                viewer.spin(false);
                spinning = false;
            }} else {{
                viewer.spin(true);
                spinning = true;
            }}
        }}

        function updateActiveButtons() {{
            // Remove all active classes
            document.querySelectorAll('button').forEach(btn => btn.classList.remove('active'));

            // Add active class to current style and color buttons
            let styleBtn = document.getElementById('btn-' + currentStyle);
            if (styleBtn) styleBtn.classList.add('active');

            let colorBtn = document.getElementById('color-' + currentColor);
            if (colorBtn) colorBtn.classList.add('active');
            
            // Toggle buttons state
            if (ghostMode) document.getElementById('btn-ghost').classList.add('active');
            if (showRestraints) document.getElementById('btn-restraints').classList.add('active');
        }}
    </script>
</body>
</html>
"""
    return html


def _find_ssbonds(pdb_content: str) -> list:
    """
    Parse SSBOND records from PDB header.
    Returns list of dicts: {'c1', 'r1', 'c2', 'r2'}
    """
    ssbonds = []
    try:
        for line in pdb_content.splitlines():
            if line.startswith("SSBOND"):
                # SSBOND   1 CYS A    6    CYS A   11
                # 16:    Chain ID 1
                # 18-21: Residue number 1
                # 30:    Chain ID 2
                # 32-35: Residue number 2
                
                try:
                    c1 = line[15]
                    r1 = int(line[17:22].strip()) # 18-21 in spec, but generous slicing
                    c2 = line[29]
                    r2 = int(line[31:36].strip()) # 32-35 in spec
                    
                    ssbonds.append({
                        'c1': c1, 'r1': r1,
                        'c2': c2, 'r2': r2
                    })
                except (ValueError, IndexError):
                    continue
        return ssbonds
    except Exception as e:
        logger.warning(f"Could not parse SSBOND records: {e}")
        return []


def _find_hbonds(pdb_content: str) -> list:
    """
    Find backbone hydrogen bonds (O to N) using simple geometric criteria.
    Returns list of dicts: {'start_resi', 'start_atom', 'end_resi', 'end_atom'}
    """
    try:
        pdb_file = pdb.PDBFile.read(io.StringIO(pdb_content))
        structure = pdb_file.get_structure(model=1)
        
        # Use Biotite's hbond analysis
        # Only backbone-backbone (triplet returned: [donor_idx, hydrogen_idx, acceptor_idx])

        # Note: synth-pdb often generates explicit hydrogens. If missing, biotite might fail or need mask.
        # We assume explicit hydrogens here or rely on acceptor-donor distance.
        
        # Actually, biotite.structure.hbond works well if H usually exists. 
        # But synth-pdb might output files without H if generated without minimization/prep?
        # Let's assume hydrogens are present for now (since we usually have them).
        
        # Try finding hydrogens first to see if we can use strict hbond
        has_hydrogens = np.any(structure.element == "H")
        
        triplets = []
        if has_hydrogens:
            try:
                triplets = hbond(structure, selection1="atom_name N", selection2="atom_name O")
            except Exception:
                pass # Fallback
        
        hbonds = []
        
        if len(triplets) > 0:
            # Strict mode worked
            for donor_idx, h_idx, acceptor_idx in triplets:
                donor = structure[donor_idx]
                acceptor = structure[acceptor_idx]
                hbonds.append({
                    'start_resi': acceptor.res_id,
                    'start_atom': acceptor.atom_name, # O
                    'end_resi': donor.res_id,
                    'end_atom': donor.atom_name # N
                })
        else:
            # Fallback: Geometric distance check (O...N < 3.5 A)
            # This is "good enough" for visualization
            ns = structure[structure.atom_name == "N"]
            os_atoms = structure[structure.atom_name == "O"]
            
            if len(ns) > 0 and len(os_atoms) > 0:
                n_coords = ns.coord
                o_coords = os_atoms.coord
                
                # Brute force distance matrix (N_n x N_o)
                # n_coords[:, None, :] is (N_n, 1, 3)
                # o_coords[None, :, :] is (1, N_o, 3)
                # Broadcasting gives (N_n, N_o, 3) displacement vectors
                diff = n_coords[:, np.newaxis, :] - o_coords[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=2)
                
                # Find pairs < 4.0 Angstrom (relaxed for visualization)
                # Returns tuple of arrays (row_indices, col_indices)
                n_indices, o_indices = np.where(dists < 4.0)
                
                for i, j in zip(n_indices, o_indices):
                    n_atom = ns[i]
                    o_atom = os_atoms[j]
                    
                    # Check sequence separation
                    seq_sep = abs(n_atom.res_id - o_atom.res_id)
                    if seq_sep >= 3:
                         hbonds.append({
                            'start_resi': o_atom.res_id,
                            'start_atom': o_atom.atom_name,
                            'end_resi': n_atom.res_id,
                            'end_atom': n_atom.atom_name
                        })

        return hbonds
        
    except Exception as e:
        logger.warning(f"Could not calculate H-bonds: {e}\n{traceback.format_exc()}")
        return []


def _find_conects(pdb_content: str) -> list:
    """
    Parse CONECT records from PDB content.
    Returns list of tuples: (serial1, serial2)
    """
    conects = []
    try:
        for line in pdb_content.splitlines():
            if line.startswith("CONECT"):
                # CONECT serial1 serial2 ...
                try:
                    parts = line.split()
                    if len(parts) >= 3:
                        s1 = int(parts[1])
                        # A CONECT record can have multiple bonds for the first atom
                        for s2_str in parts[2:]:
                            s2 = int(s2_str)
                            # Avoid duplicates (1, 80) and (80, 1) in our drawing list
                            # (PDB files often list both directions)
                            if (s1, s2) not in conects and (s2, s1) not in conects:
                                conects.append((s1, s2))
                except (ValueError, IndexError):
                    continue
        return conects
    except Exception as e:
        logger.warning(f"Could not parse CONECT records: {e}")
        return []

