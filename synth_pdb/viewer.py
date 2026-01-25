"""
3D Molecular Viewer for synth-pdb.

Opens generated PDB structures in browser-based 3D viewer using 3Dmol.js.
Based on pdbstat's molecular viewer implementation.
"""

import logging
import tempfile
import webbrowser
from pathlib import Path

logger = logging.getLogger(__name__)


def view_structure_in_browser(
    pdb_content: str,
    filename: str = "structure.pdb",
    style: str = "cartoon",
    color: str = "spectrum",
    restraints: list = None,
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
        html = _create_3dmol_html(pdb_content, filename, style, color, restraints)
        
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
    pdb_data: str, filename: str, style: str, color: str, restraints: list = None
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
        restraints: Optional list of restraint dicts to visualize as cylinders
        
    Returns:
        Complete HTML document as string
    """
    # Escape PDB data for JavaScript (prevent injection attacks)
    pdb_escaped = pdb_data.replace("\\", "\\\\").replace("`", "\\`")
    
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
    <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
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

    <div id="viewer"></div>

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
            let element = document.getElementById('viewer');
            let config = {{ backgroundColor: 'white' }};
            viewer = $3Dmol.createViewer(element, config);

            // Load PDB data
            let pdbData = `{pdb_escaped}`;
            // IMPORTANT: keepH: true is required to visualize NMR restraints involving protons!
            viewer.addModel(pdbData, "pdb", {{keepH: true}});

            // Set initial style and render
            applyStyle();
            
            // Draw initial restraints
            drawRestraints();
            
            viewer.zoomTo();
            viewer.render();
            
            // Highlight active buttons
            updateActiveButtons();
        }});
        
        // Define Restraints Data
        {js_restraints}

        function drawRestraints() {{
            // clear existing shapes first
            viewer.removeAllShapes();
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

            viewer.setStyle({{}}, styleObj);
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
