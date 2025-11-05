# =========================================
# LCI visualization · compact swimlane (Graphviz) + interactive web (Cytoscape)
# =========================================
from __future__ import annotations
from typing import List, Tuple, Dict
import re, json, logging
from pathlib import Path

from perovops.io.schemas import LCIModel

# ---- Logger setup to avoid NameError ----
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger(__name__)

try:
    import graphviz
except ImportError:
    graphviz = None

# ==== Palette and styles (inspired by ColorBrewer pastel/qualitative sets) ====
PALETTE = {
    # Swimlane backgrounds (pastel)
    "lane_material": "#ccebc5",   # pastel green
    "lane_energy":    "#b3cde3",   # pastel blue
    "lane_mfg":       "#f2f2f2",   # light gray
    "lane_eol":       "#fbb4ae",   # pastel red
    # Node fill colors
    "node_material":  "#e6f4ea",
    "node_energy_el": "#e7f1fb",
    "node_energy_th": "#fff3e0",
    "node_process":   "#ffffff",
    "node_eol":       "#ffe5e5",
    # Edge colors (material green, electric blue, thermal orange, product gray, emission red)
    "edge_material":  "#2e7d32",
    "edge_energy_el": "#1f77b4",
    "edge_energy_th": "#fb8c00",
    "edge_product":   "#5f6368",
    "edge_eol":       "#d62728",
}

# ---- Basic helpers ----
_SAN_RE = re.compile(r"[^0-9A-Za-z_]")
def _san(s: str) -> str:
    """Sanitize into a DOT-safe ID (remove colons/spaces, etc.)."""
    return _SAN_RE.sub("_", s)

def _fmt_qty(v: float, unit: str) -> str:
    u = (unit or "").lower()
    if u in ("kwh", "mj"):
        kv = v if u == "kwh" else (v / 3.6)
        return f"{kv:.3g} kWh"
    if u in ("kg", "g", "mg"):
        g = v * (1000.0 if u == "kg" else (1.0 if u == "g" else 0.001))
        return f"{g:.3g} g"
    if u in ("l", "ml", "mL", "ul", "uL"):
        ml = v * (1000.0 if u.lower() == "l" else (1.0 if u.lower() == "ml" else 0.001))
        return f"{ml:.3g} mL"
    if u in ("m^3", "m3"):
        return f"{v:.3g} m³"
    return f"{v:.3g} {unit}"

def _energy_kwh(step) -> float:
    total = 0.0
    for e in step.energy:
        v = float(e.quantity.value); u = e.quantity.unit.lower()
        total += v if u == "kwh" else (v/3.6 if u == "mj" else 0.0)
    return total


def export_lci_json(lci: LCIModel, output_path: str) -> str:
    """Serialize the raw ``LCIModel`` content to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(lci.dict(), fh, ensure_ascii=False, indent=2, default=str)
    logger.info("[raw-json] LCI saved: %s", path)
    return str(path)


def export_graphviz_dot(lci: LCIModel, output_path_noext: str) -> str:
    """Write a minimal DOT graph for quick inspection / debugging."""
    base = Path(output_path_noext)
    base.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "digraph LCI {",
        "  rankdir=LR;",
        "  node [shape=box, fontsize=10];",
    ]

    for step in lci.steps:
        energy = _energy_kwh(step)
        label = f"{step.id}\\n{step.name}\\n[{step.category}]\\nE={energy:.2f} kWh"
        lines.append(f'  "{step.id}" [label="{label}"];')

    for edge in lci.edges:
        lines.append(f'  "{edge.source}" -> "{edge.target}" [label="{edge.flow_type}"];')

    lines.append("}")

    dot_path = base.with_suffix('.dot')
    dot_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("[dot] Graphviz saved: %s", dot_path)
    return str(dot_path)

# ===============================
# 1) Graphviz swimlane compact export
# ===============================
def export_graphviz_swimlane_compact(
    lci,
    output_path_noext: str,
    format: str = "svg",
    rankdir: str = "LR",
    nodesep: float = 0.25,
    ranksep: float = 0.35,
    fontname: str = "Arial",
    show_sum_energy: bool = False,   # Process nodes show only the name; set True to append Σ energy
):
    """Render a compact swimlane diagram with the requested visual style.

    - Swimlanes: Raw materials / Energy / Manufacturing / Emissions & EoL
    - Nodes: process = rounded rectangle; material = light green box; energy = hexagon (electric blue or thermal orange); emission = light red box
    - Edges: material (green solid), electric energy (blue dashed), thermal energy (orange dashed), product (gray), emission (red)
    - Labels: keep process labels minimal; place values on edge ``xlabel`` to avoid layout issues with ``ortho``.
    """
    if graphviz is None:
        raise RuntimeError("graphviz package not installed; please `pip install graphviz` and install Graphviz binary")

    dot = graphviz.Digraph(
        name=_san(lci.name),
        comment=f"LCI Compact Graph for {lci.name}",
        format=format,
    )
    # Global graph attributes
    dot.attr(
        rankdir=rankdir, splines="ortho", concentrate="true",
        nodesep=str(nodesep), ranksep=str(ranksep), compound="true"
    )
    dot.attr("node", fontname=fontname, fontsize="9", style="rounded,filled", margin="0.03,0.02")
    dot.attr("edge", fontname=fontname, fontsize="8", arrowsize="0.7")

    # Swimlane clusters
    with dot.subgraph(name="cluster_materials") as lane_m:
        lane_m.attr(label="Raw materials", fontsize="11", style="rounded,filled", color="#d5e8d4", bgcolor=PALETTE["lane_material"])
        lane_m.attr("node", shape="box", fillcolor=PALETTE["node_material"])
    with dot.subgraph(name="cluster_energy") as lane_e:
        lane_e.attr(label="Energy", fontsize="11", style="rounded,filled", color="#c5d9f1", bgcolor=PALETTE["lane_energy"])
        lane_e.attr("node", shape="hexagon", fillcolor=PALETTE["node_energy_el"])
    with dot.subgraph(name="cluster_proc") as lane_p:
        lane_p.attr(label="Manufacturing", fontsize="12", style="rounded,filled", color="#e0e0e0", bgcolor=PALETTE["lane_mfg"])
        lane_p.attr("node", shape="box", fillcolor=PALETTE["node_process"])
    with dot.subgraph(name="cluster_eol") as lane_x:
        lane_x.attr(label="Emissions & EoL", fontsize="11", style="rounded,filled", color="#f5c6c6", bgcolor=PALETTE["lane_eol"])
        lane_x.attr("node", shape="box", fillcolor=PALETTE["node_eol"])

    # Reuse energy nodes (grouped by type and region)
    energy_key_to_nodeid: Dict[str, str] = {}
    def _energy_node_id(e) -> str:
        key = f"{e.type}_{e.region}"
        if key not in energy_key_to_nodeid:
            nid = _san(f"energy_{key}")
            # Differentiate fill color for electric vs thermal nodes
            fill = PALETTE["node_energy_el"] if e.type == "electricity" else PALETTE["node_energy_th"]
            label = f"<<B>{e.type.title()}</B>>"  # Show only "Electricity" / "Thermal"
            lane_e.node(nid, label=label, tooltip=f"{e.type}, {e.region}", fillcolor=fill)
            energy_key_to_nodeid[key] = nid
        return energy_key_to_nodeid[key]

    # Process nodes plus adjacent material/energy/emission nodes
    for step in lci.steps:
        sid = _san(step.id)

        # --- Process node (minimal label) ---
        node_label = f"<<B>{step.name}</B>" + (f"<BR/><FONT POINT-SIZE='8'>ΣE { _energy_kwh(step):.2f} kWh</FONT>" if show_sum_energy else "") + ">"
        lane_p.node(sid, label=node_label, tooltip=f"{step.name}")

        # --- Material inputs (dedicated node per step) ---
        for i, m in enumerate(step.inputs):
            mid = _san(f"mat_{step.id}_{i}_{m.material}")
            mlabel = f"<<B>{m.material}</B>>"   # Put quantities on edges to keep node height compact
            lane_m.node(mid, label=mlabel, tooltip=f"{m.material} • {_fmt_qty(m.quantity.value, m.quantity.unit)}")
            dot.edge(
                mid, sid,
                color=PALETTE["edge_material"], style="solid", penwidth="1.3", arrowhead="vee",
                xlabel=_fmt_qty(m.quantity.value, m.quantity.unit)  # Use xlabel to avoid ortho+label warnings/compression
            )

        # --- Energy inputs (reuse electric/thermal nodes) ---
        for en in step.energy:
            enid = _energy_node_id(en)
            ecol = PALETTE["edge_energy_el"] if en.type == "electricity" else PALETTE["edge_energy_th"]
            dot.edge(
                enid, sid,
                color=ecol, style="dashed", penwidth="1.5", arrowhead="vee",
                xlabel=_fmt_qty(en.quantity.value, en.quantity.unit)
            )

        # --- Emissions/products (right side) ---
        for j, o in enumerate(step.outputs or []):
            oid = _san(f"out_{step.id}_{j}_{o.material}")
            olabel = f"<<B>{o.material}</B>>"
            lane_x.node(oid, label=olabel, tooltip=f"{o.material} • {_fmt_qty(o.quantity.value, o.quantity.unit)}")
            dot.edge(
                sid, oid,
                color=PALETTE["edge_eol"], style="solid", penwidth="1.2", arrowhead="vee",
                xlabel=_fmt_qty(o.quantity.value, o.quantity.unit)
            )

    # --- Process sequence (product) ---
    for e in lci.edges:
        if e.flow_type == "product":
            dot.edge(
                _san(e.source), _san(e.target),
                color=PALETTE["edge_product"], penwidth="1.5", arrowhead="normal",
                xlabel=(e.label or "")
            )

    # Render graph
    out_file = dot.render(filename=output_path_noext, cleanup=True)
    logger.info(f"[swimlane-compact] Graphviz exported: {out_file}")
    return out_file

# ===============================
# 2) Interactive HTML (compact preset grid)
# ===============================
def _topo_ranks(lci) -> Dict[str, int]:
    """Derive topological ranks from product edges for the compact preset layout."""
    succ, indeg = {}, {}
    ids = [s.id for s in lci.steps]
    for s in ids:
        succ[s] = []
        indeg[s] = 0
    for e in lci.edges:
        if e.flow_type == "product":
            succ[e.source].append(e.target)
            indeg[e.target] = indeg.get(e.target, 0) + 1
    # Kahn
    from collections import deque
    q = deque([n for n in ids if indeg.get(n, 0) == 0])
    rank = {n: 0 for n in q}
    while q:
        u = q.popleft()
        for v in succ.get(u, []):
            rank[v] = max(rank.get(v, 0), rank[u] + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return rank

def export_cytoscape_compact_html(
    lci,
    output_html: str,
    col_gap: int = 260,   # Column spacing
    row_gap: int = 120,   # Row spacing
) -> bool:
    """Single-file HTML with a compact preset layout (materials/energy column = 0, process column = topo rank, emissions column = max + 1)."""
    ranks = _topo_ranks(lci)
    max_rank = (max(ranks.values()) if ranks else 0)
    # elements
    nodes, edges = [], []

    # Process nodes
    row = 0
    for s in lci.steps:
        r = ranks.get(s.id, 0)
        nodes.append({
            "data": {"id": s.id, "label": s.name, "type": "process", "category": s.category},
            "position": {"x": (r+1)*col_gap, "y": (row+1)*row_gap}
        })
        row += 1

    # Material/energy/emission nodes and edges
    mat_i, en_i, out_i = 0, 0, 0
    for s in lci.steps:
        # Materials: column = 0
        for m in s.inputs:
            mid = f"m_{mat_i}"; mat_i += 1
            nodes.append({
                "data": {"id": mid, "label": f"{m.material}", "type": "material"},
                "position": {"x": 0, "y": (mat_i)*row_gap*0.8}
            })
            edges.append({"data": {"id": f"em_{mid}_{s.id}", "source": mid, "target": s.id,
                                   "label": _fmt_qty(m.quantity.value, m.quantity.unit), "etype": "material"}})
        # Energy: column = 0, adjacent to materials
        for en in s.energy:
            eid = f"e_{en_i}"; en_i += 1
            en_label = "Electricity" if en.type == "electricity" else "Thermal"
            nodes.append({
                "data": {"id": eid, "label": en_label, "type": "energy", "energyType": en.type},
                "position": {"x": 0, "y": (mat_i+en_i)*row_gap*0.8}
            })
            edges.append({"data": {"id": f"ee_{eid}_{s.id}", "source": eid, "target": s.id,
                                   "label": _fmt_qty(en.quantity.value, en.quantity.unit), "etype": "energy",
                                   "energyType": en.type}})
        # Emissions: column = max_rank + 2
        for o in (s.outputs or []):
            oid = f"o_{out_i}"; out_i += 1
            nodes.append({
                "data": {"id": oid, "label": o.material, "type": "emission"},
                "position": {"x": (max_rank+2)*col_gap, "y": (out_i)*row_gap*0.9}
            })
            edges.append({"data": {"id": f"eo_{s.id}_{oid}", "source": s.id, "target": oid,
                                   "label": _fmt_qty(o.quantity.value, o.quantity.unit), "etype": "emission"}})

    # product edges
    for e in lci.edges:
        if e.flow_type == "product":
            edges.append({"data": {"id": f"p_{e.source}_{e.target}", "source": e.source, "target": e.target,
                                   "label": e.label or "", "etype": "product"}})

    # Color mapping
    cat_colors = {
        "sputter": "#CCE5FF", "screen_print": "#CCFFCC", "anneal": "#FFCCCC",
        "lamination": "#FFFFCC", "cleaning": "#E6E6E6", "PECVD": "#E6CCFF",
        "evaporation": "#FFE6CC", "ALD": "#E6CCFF", "other": "#FFFFFF",
    }

    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>LCI — {lci.name}</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  body {{ margin:0; font-family:Arial, sans-serif; }}
  #cy {{ width:100vw; height:100vh; background:#fafafa; }}
  .legend {{ position:fixed; right:10px; top:10px; background:#fff; border:1px solid #ddd; padding:6px; font-size:12px; }}
</style>
</head>
<body>
<div id="cy"></div>
<div class="legend">Process/Material/Energy/Emission colors per lane.</div>
<script src="https://unpkg.com/cytoscape@3.26.0/dist/cytoscape.min.js"></script>
<script>
const elements = {json.dumps({"nodes": nodes, "edges": edges})};
const catColor = {json.dumps(cat_colors)};
const PALETTE = {json.dumps(PALETTE)};

const cy = cytoscape({{
  container: document.getElementById('cy'),
  elements: elements,
  style: [
    {{ selector: 'node[type = "process"]',
      style: {{
        'shape':'round-rectangle','background-color': ele => catColor[ele.data('category')] || '#F8F9FB',
        'label':'data(label)','font-size':10,'text-wrap':'wrap','text-max-width':160,'padding':'4px','border-width':0
      }} }},
    {{ selector: 'node[type = "material"]',
      style: {{ 'shape':'round-rectangle','background-color': PALETTE.node_material,
        'label':'data(label)','font-size':9,'padding':'3px','border-width':0 }} }},
    {{ selector: 'node[type = "energy"]',
      style: {{ 'shape':'hexagon','background-color': ele => ele.data('energyType') === 'thermal' ? PALETTE.node_energy_th : PALETTE.node_energy_el,
        'label':'data(label)','font-size':9,'padding':'3px','border-width':0 }} }},
    {{ selector: 'node[type = "emission"]',
      style: {{ 'shape':'round-rectangle','background-color': PALETTE.node_eol,
        'label':'data(label)','font-size':9,'padding':'3px','border-width':0 }} }},
    {{ selector: 'edge[etype = "product"]',
      style: {{ 'width':2,'line-color':PALETTE.edge_product,'target-arrow-shape':'triangle','target-arrow-color':PALETTE.edge_product,'curve-style':'bezier','label':'data(label)','font-size':8 }} }},
    {{ selector: 'edge[etype = "material"]',
      style: {{ 'width':1.4,'line-color':PALETTE.edge_material,'target-arrow-shape':'vee','target-arrow-color':PALETTE.edge_material,'curve-style':'bezier','label':'data(label)','font-size':8 }} }},
    {{ selector: 'edge[etype = "energy"]',
      style: {{ 'width':1.6,'line-color': ele => ele.data('energyType') === 'thermal' ? PALETTE.edge_energy_th : PALETTE.edge_energy_el,
        'line-style':'dashed','target-arrow-shape':'vee','target-arrow-color': ele => ele.data('energyType') === 'thermal' ? PALETTE.edge_energy_th : PALETTE.edge_energy_el,
        'curve-style':'bezier','label':'data(label)','font-size':8 }} }},
    {{ selector: 'edge[etype = "emission"]',
      style: {{ 'width':1.4,'line-color':PALETTE.edge_eol,'target-arrow-shape':'vee','target-arrow-color':PALETTE.edge_eol,'curve-style':'bezier','label':'data(label)','font-size':8 }} }}
  ],
  layout: {{ name:'preset', fit:true, padding:20 }}
}});
</script>
</body></html>"""
    Path(output_html).parent.mkdir(parents=True, exist_ok=True)
    Path(output_html).write_text(html, encoding="utf-8")
    logger.info(f"[interactive-compact] HTML exported: {output_html}")
    return True


def export_graphviz_compact(*args, **kwargs):
    """Backward-compatible wrapper for ``export_graphviz_swimlane_compact``."""
    return export_graphviz_swimlane_compact(*args, **kwargs)


def export_cytoscape_html(*args, **kwargs):
    """Backward-compatible wrapper for ``export_cytoscape_compact_html``."""
    return export_cytoscape_compact_html(*args, **kwargs)
