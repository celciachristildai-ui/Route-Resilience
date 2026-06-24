"""
Phase IV: Interactive Dashboard
Streamlit app tying together Phases I-III with live visualization,
disaster simulation, and criticality exploration.
"""

import streamlit as st
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import sys
import os
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.phase1_segmentation.model import RoadSegmentor
from src.phase2_graph.graph_builder import GraphBuilder
from src.phase3_analysis.analyzer import NetworkAnalyzer


# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ROUTE RESILIENCE // Network Operations",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styling — Operations Console ─────────────────────────────────────────────
# Design language: a network-operations console for crisis response, not a
# product dashboard. Dark instrument panel, monospace telemetry type for data,
# a condensed grotesk for command labels, and exactly one alarm color reserved
# for severity — everything else stays cold and quiet so red actually means something.

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    :root {
        --void:      #0a0d10;
        --panel:     #11161b;
        --panel-2:   #161d23;
        --line:      #232c33;
        --line-lit:  #2f3b44;
        --ink:       #d8dee2;
        --ink-dim:   #748089;
        --ink-bright:#f2f5f6;
        --signal:    #ff9d2e;
        --signal-dim:#7a5226;
        --ok:        #4a9b8e;
        --crit:      #e44b4b;
        --mono: 'JetBrains Mono', monospace;
        --grot: 'Space Grotesk', sans-serif;
    }

    .stApp { background-color: var(--void); }
    body, p, span, div, label { color: var(--ink); }

    /* kill default streamlit chrome smell */
    #MainMenu, footer, header[data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 1.6rem; max-width: 1400px; }

    h1, h2, h3, h4 {
        font-family: var(--grot);
        color: var(--ink-bright);
        letter-spacing: -0.01em;
        font-weight: 600;
    }

    /* ── Console header bar ───────────────────────────────────── */
    .console-bar {
        display: flex; align-items: stretch; justify-content: space-between;
        background: var(--panel); border: 1px solid var(--line);
        border-radius: 3px; padding: 0; margin-bottom: 1.4rem;
        font-family: var(--mono); overflow: hidden;
    }
    .console-id {
        padding: 0.9rem 1.3rem; border-right: 1px solid var(--line);
        display: flex; flex-direction: column; gap: 2px; min-width: 230px;
    }
    .console-id .tag { font-size: 0.68rem; letter-spacing: 0.14em; color: var(--ink-dim); text-transform: uppercase; }
    .console-id .name { font-family: var(--grot); font-size: 1.25rem; font-weight: 700; color: var(--ink-bright); letter-spacing: -0.01em; }
    .console-readouts { display: flex; flex: 1; }
    .readout {
        flex: 1; padding: 0.9rem 1.3rem; border-right: 1px solid var(--line);
        display: flex; flex-direction: column; gap: 2px; justify-content: center;
    }
    .readout:last-child { border-right: none; }
    .readout .rlabel { font-size: 0.65rem; letter-spacing: 0.12em; color: var(--ink-dim); text-transform: uppercase; }
    .readout .rvalue { font-size: 1.15rem; font-weight: 600; color: var(--ink-bright); font-family: var(--mono); }
    .readout .rvalue.signal { color: var(--signal); }
    .readout .rvalue.ok { color: var(--ok); }
    .readout .rvalue.crit { color: var(--crit); }
    .pulse-dot {
        display: inline-block; width: 7px; height: 7px; border-radius: 50%;
        background: var(--ok); margin-right: 6px; box-shadow: 0 0 6px var(--ok);
        animation: pulse 2s infinite;
    }
    .pulse-dot.idle { background: var(--ink-dim); box-shadow: none; animation: none; }
    @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.35; } }

    /* ── Section eyebrow tags ─────────────────────────────────── */
    .phase-tag {
        display: inline-flex; align-items: center; gap: 6px;
        background: var(--panel-2); color: var(--ink-dim); border: 1px solid var(--line);
        padding: 3px 10px; border-radius: 2px; font-family: var(--mono);
        font-size: 0.68rem; letter-spacing: 0.12em; text-transform: uppercase;
        margin-bottom: 0.7rem;
    }
    .phase-tag::before { content: '◆'; color: var(--signal); font-size: 0.6rem; }

    /* ── Cards / metric surfaces ──────────────────────────────── */
    .op-card {
        background: var(--panel); border: 1px solid var(--line); border-radius: 3px;
        padding: 1rem 1.2rem; margin-bottom: 0.6rem;
    }
    div[data-testid="stMetric"] {
        background: var(--panel); border: 1px solid var(--line); border-radius: 3px;
        padding: 0.85rem 1rem;
    }
    div[data-testid="stMetric"] label { font-family: var(--mono); font-size: 0.68rem !important;
        letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-dim) !important; }
    div[data-testid="stMetricValue"] { font-family: var(--mono); color: var(--ink-bright); }

    /* ── Severity classes ─────────────────────────────────────── */
    .severity-low { color: var(--ok); font-weight: 700; font-family: var(--mono); }
    .severity-medium { color: var(--signal); font-weight: 700; font-family: var(--mono); }
    .severity-high { color: #e08a3a; font-weight: 700; font-family: var(--mono); }
    .severity-critical { color: var(--crit); font-weight: 700; font-family: var(--mono); }

    /* ── Sidebar ───────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background-color: var(--panel); border-right: 1px solid var(--line);
    }
    section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label {
        font-family: var(--mono); font-size: 0.82rem;
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        font-family: var(--mono); font-size: 0.78rem; letter-spacing: 0.1em;
        text-transform: uppercase; color: var(--ink-dim); border-top: 1px solid var(--line);
        padding-top: 1rem; margin-top: 0.4rem;
    }
    /* primary action button */
    .stButton button[kind="primary"] {
        background: var(--signal); border: none; color: #1a1304; font-family: var(--mono);
        font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; border-radius: 2px;
    }
    .stButton button[kind="primary"]:hover { background: #ffb554; }
    .stButton button[kind="secondary"] {
        background: var(--panel-2); border: 1px solid var(--line-lit); color: var(--ink);
        font-family: var(--mono); border-radius: 2px;
    }

    /* ── Tabs ──────────────────────────────────────────────────── */
    button[data-baseweb="tab"] {
        font-family: var(--mono); font-size: 0.78rem; letter-spacing: 0.05em;
        text-transform: uppercase; color: var(--ink-dim);
    }
    button[data-baseweb="tab"][aria-selected="true"] { color: var(--signal) !important; }
    div[data-baseweb="tab-highlight"] { background-color: var(--signal) !important; }
    div[data-baseweb="tab-border"] { background-color: var(--line) !important; }

    /* ── Dataframes / tables ──────────────────────────────────── */
    div[data-testid="stDataFrame"] { font-family: var(--mono); }

    /* ── Alerts ────────────────────────────────────────────────── */
    div[data-testid="stAlert"] { font-family: var(--mono); font-size: 0.85rem; border-radius: 3px; }

    /* ── Captions / dividers ───────────────────────────────────── */
    .stCaption, [data-testid="stCaptionContainer"] { font-family: var(--mono); color: var(--ink-dim) !important; }
    hr { border-color: var(--line) !important; }

    /* images get a console frame */
    div[data-testid="stImage"] img { border: 1px solid var(--line); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = dict(
    plot_bgcolor='#0d1216',
    paper_bgcolor='#11161b',
    font=dict(family='JetBrains Mono, monospace', color='#d8dee2', size=11),
)




# ─── Config Loading ────────────────────────────────────────────────────────────

@st.cache_resource
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'pipeline.yaml')
    with open(config_path) as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_segmentor(seg_config):
    return RoadSegmentor(seg_config, device='cpu')


def get_graph_builder(graph_config):
    return GraphBuilder(graph_config)


def get_analyzer(analysis_config):
    return NetworkAnalyzer(analysis_config)


# ─── Session State ─────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        'mask': None, 'prob_map': None, 'graph': None, 'skeleton': None,
        'report': None, 'stress_results': None, 'disaster_sim': None,
        'pipeline_ran': False, 'image_size': 512,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()
config = load_config()


# ─── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div style="font-family:var(--mono); font-size:0.95rem; font-weight:700; '
        'color:var(--ink-bright); letter-spacing:0.02em;">ROUTE RESILIENCE</div>'
        '<div style="font-family:var(--mono); font-size:0.68rem; color:var(--ink-dim); '
        'letter-spacing:0.08em; margin-top:2px; margin-bottom:0.6rem;">NETWORK OPS CONSOLE</div>',
        unsafe_allow_html=True
    )

    st.subheader("Input feed")
    input_mode = st.radio("Source", ["Synthetic demo network", "Upload satellite image"],
                           label_visibility="collapsed")

    uploaded_file = None
    if input_mode == "Upload satellite image":
        uploaded_file = st.file_uploader("Satellite tile (.png / .jpg / .tif)",
                                          type=['png', 'jpg', 'jpeg', 'tif', 'tiff'])

    image_size = st.select_slider("Tile resolution", options=[256, 512, 768], value=512)

    st.subheader("Healing thresholds")
    min_branch = st.slider("Discard branches shorter than (px)", 5, 40, 15)
    max_gap = st.slider("Bridge gaps up to (px)", 10, 120, 50)
    snap_tol = st.slider("Snap endpoints within (px)", 2, 30, 10)

    st.subheader("Criticality scope")
    top_k = st.slider("Track top-N critical elements", 5, 50, 20)
    ablation_k = st.slider("Stress-test depth (edges removed)", 3, 25, 10)

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    run_btn = st.button("▶  RUN FULL PIPELINE", type="primary", use_container_width=True)


# ─── Header — console status bar ───────────────────────────────────────────────

_ran = st.session_state.get('pipeline_ran', False)
_res = st.session_state.get('report').resilience_score if _ran and st.session_state.get('report') else None
_stats = st.session_state.get('graph_stats') if _ran else None

if not _ran:
    status_html = (
        '<span class="pulse-dot idle"></span>STANDBY'
    )
    res_html = '<span class="rvalue">—</span>'
    nodes_html = '<span class="rvalue">—</span>'
    comp_html = '<span class="rvalue">—</span>'
else:
    status_html = '<span class="pulse-dot"></span>LIVE'
    res_class = 'ok' if _res >= 0.7 else ('signal' if _res >= 0.4 else 'crit')
    res_html = f'<span class="rvalue {res_class}">{_res:.3f}</span>'
    nodes_html = f'<span class="rvalue">{_stats["num_nodes"]} / {_stats["num_edges"]}</span>'
    comp_class = 'ok' if _stats['num_components'] == 1 else 'signal'
    comp_html = f'<span class="rvalue {comp_class}">{_stats["num_components"]}</span>'

st.markdown(f"""
<div class="console-bar">
    <div class="console-id">
        <div class="tag">🛰️ Phases I–IV</div>
        <div class="name">ROUTE RESILIENCE</div>
    </div>
    <div class="console-readouts">
        <div class="readout"><div class="rlabel">Status</div><div class="rvalue">{status_html}</div></div>
        <div class="readout"><div class="rlabel">Resilience score</div>{res_html}</div>
        <div class="readout"><div class="rlabel">Nodes / Edges</div>{nodes_html}</div>
        <div class="readout"><div class="rlabel">Components</div>{comp_html}</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Swin-UNet occlusion-robust segmentation → skeleton healing with MST gap-bridging "
    "→ betweenness-centrality criticality ranking → live disaster stress testing."
)

tabs = st.tabs(["📡  SEGMENTATION", "🕸️  GRAPH + HEALING", "📊  CRITICALITY",
                 "🌊  DISASTER SIM", "📋  REPORT"])


# ─── Pipeline Execution ─────────────────────────────────────────────────────────

if run_btn:
    with st.spinner("Running Phase I — segmentation..."):
        seg_config = config['segmentation']
        seg_config['input_size'] = [image_size, image_size]
        segmentor = load_segmentor(seg_config)

        if uploaded_file is not None:
            from PIL import Image
            img = Image.open(uploaded_file).convert('RGB').resize((image_size, image_size))
            img_arr = np.array(img)
            mask, prob_map = segmentor.segment(img_arr)
        else:
            mask, prob_map = segmentor.generate_synthetic_demo(size=image_size)

        st.session_state.mask = mask
        st.session_state.prob_map = prob_map
        st.session_state.image_size = image_size

    with st.spinner("Running Phase II — skeletonization & topological healing..."):
        graph_config = dict(config['graph'])
        graph_config['min_branch_length'] = min_branch
        graph_config['max_gap_fill'] = max_gap
        graph_config['snap_tolerance'] = snap_tol

        builder = get_graph_builder(graph_config)
        G, skeleton = builder.build(mask)
        st.session_state.graph = G
        st.session_state.skeleton = skeleton
        st.session_state.graph_stats = builder.graph_stats(G)

    with st.spinner("Running Phase III — criticality analysis..."):
        analysis_config = dict(config['analysis'])
        analysis_config['top_k_critical'] = top_k
        analysis_config['ablation_k'] = ablation_k

        analyzer = get_analyzer(analysis_config)
        report = analyzer.analyze(G)
        stress_results = analyzer.stress_test(G, report, k=ablation_k)

        st.session_state.report = report
        st.session_state.stress_results = stress_results
        st.session_state.analyzer = analyzer

    st.session_state.pipeline_ran = True
    st.success("Pipeline complete — explore results in the tabs below.")


# ─── Helper: Plotly graph rendering ────────────────────────────────────────────

def render_graph_plotly(G, highlight_nodes=None, highlight_edges=None,
                          node_colors=None, title="Road Network Graph",
                          epicenter=None, radius=None, destroyed_edges=None):
    edge_x, edge_y = [], []
    for u, v in G.edges():
        y0, x0 = G.nodes[u].get('y', 0), G.nodes[u].get('x', 0)
        y1, x1 = G.nodes[v].get('y', 0), G.nodes[v].get('x', 0)
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(width=1.1, color='#2f3b44'),
        hoverinfo='none', name='network'
    ))

    if destroyed_edges:
        dx, dy = [], []
        for u, v in destroyed_edges:
            if u in G.nodes and v in G.nodes:
                y0, x0 = G.nodes[u].get('y', 0), G.nodes[u].get('x', 0)
                y1, x1 = G.nodes[v].get('y', 0), G.nodes[v].get('x', 0)
                dx += [x0, x1, None]
                dy += [y0, y1, None]
        fig.add_trace(go.Scatter(
            x=dx, y=dy, mode='lines',
            line=dict(width=3, color='#e44b4b'),
            hoverinfo='none', name='destroyed'
        ))

    if highlight_edges:
        hx, hy = [], []
        for u, v in highlight_edges:
            if G.has_edge(u, v):
                y0, x0 = G.nodes[u].get('y', 0), G.nodes[u].get('x', 0)
                y1, x1 = G.nodes[v].get('y', 0), G.nodes[v].get('x', 0)
                hx += [x0, x1, None]
                hy += [y0, y1, None]
        fig.add_trace(go.Scatter(
            x=hx, y=hy, mode='lines',
            line=dict(width=3, color='#ff9d2e'),
            hoverinfo='none', name='critical edges'
        ))

    node_x = [G.nodes[n].get('x', 0) for n in G.nodes()]
    node_y = [G.nodes[n].get('y', 0) for n in G.nodes()]

    if node_colors is not None:
        colors = [node_colors.get(n, 0) for n in G.nodes()]
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y, mode='markers',
            marker=dict(size=5.5, color=colors, colorscale=[
                            [0.0, '#1c3a3a'], [0.35, '#2f6f6a'],
                            [0.65, '#e8a23d'], [1.0, '#ff5050']],
                        showscale=True,
                        colorbar=dict(title=dict(text="criticality", font=dict(color='#748089', size=10)),
                                      tickfont=dict(color='#748089', size=9),
                                      outlinewidth=0, bgcolor='rgba(0,0,0,0)')),
            text=[f"deg={G.degree(n)}" for n in G.nodes()],
            hoverinfo='text', name='nodes'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y, mode='markers',
            marker=dict(size=4, color='#4a9b8e'),
            hoverinfo='skip', name='nodes'
        ))

    if epicenter is not None:
        ey, ex = epicenter
        theta = np.linspace(0, 2 * np.pi, 100)
        fig.add_trace(go.Scatter(
            x=ex + radius * np.cos(theta), y=ey + radius * np.sin(theta),
            mode='lines', line=dict(color='#e44b4b', width=2, dash='dash'),
            name='impact radius'
        ))
        fig.add_trace(go.Scatter(
            x=[ex], y=[ey], mode='markers',
            marker=dict(size=14, color='#e44b4b', symbol='x'),
            name='epicenter'
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(family='JetBrains Mono, monospace', size=13, color='#d8dee2')),
        showlegend=True,
        legend=dict(font=dict(color='#748089', size=10), bgcolor='rgba(0,0,0,0)'),
        height=560,
        plot_bgcolor='#0d1216', paper_bgcolor='#11161b',
        yaxis=dict(autorange='reversed', scaleanchor='x', showgrid=False, zeroline=False,
                   showticklabels=False),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ─── TAB 1: Segmentation ───────────────────────────────────────────────────────

with tabs[0]:
    st.markdown('<div class="phase-tag">Phase I</div>', unsafe_allow_html=True)
    st.subheader("Occlusion-Robust Segmentation")
    st.write(
        "Swin-UNet transformer segments road pixels from satellite imagery. "
        "Shifted-window self-attention gives the model global context, so it can "
        "infer road continuity through tree-canopy shadows, vehicle occlusion, and cloud cover."
    )

    if st.session_state.mask is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                '<div style="font-family:var(--mono); font-size:0.75rem; color:var(--ink-dim); '
                'letter-spacing:0.05em; text-transform:uppercase; margin-bottom:6px;">'
                'ROAD MASK — binary</div>', unsafe_allow_html=True)
            inv_mask = 255 - (st.session_state.mask * 255)
            st.image(inv_mask, clamp=True, use_container_width=True)
        with col2:
            st.markdown(
                '<div style="font-family:var(--mono); font-size:0.75rem; color:var(--ink-dim); '
                'letter-spacing:0.05em; text-transform:uppercase; margin-bottom:6px;">'
                'CONFIDENCE FIELD — per-pixel probability</div>', unsafe_allow_html=True)
            st.image(1.0 - st.session_state.prob_map, clamp=True, use_container_width=True)

        coverage = st.session_state.mask.mean() * 100
        m1, m2, m3 = st.columns(3)
        m1.metric("Road pixel coverage", f"{coverage:.1f}%")
        m2.metric("Tile resolution", f"{st.session_state.image_size}×{st.session_state.image_size}")
        m3.metric("Mean confidence", f"{st.session_state.prob_map.mean():.3f}")
    else:
        st.info("Awaiting input — run the pipeline from the console to generate a segmentation mask.")


# ─── TAB 2: Graph & Healing ─────────────────────────────────────────────────────

with tabs[1]:
    st.markdown('<div class="phase-tag">Phase II</div>', unsafe_allow_html=True)
    st.subheader("Skeletonization & Topological Healing")
    st.write(
        "The binary mask is reduced to a 1-pixel-wide medial-axis skeleton, then traced "
        "into a graph of junctions and road segments. Disconnected fragments caused by "
        "occlusion are bridged using a minimum-spanning-tree heuristic over inter-component "
        "distances, weighted via Union-Find to avoid redundant bridges."
    )

    if st.session_state.graph is not None:
        G = st.session_state.graph
        stats = st.session_state.graph_stats

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Nodes", stats['num_nodes'])
        c2.metric("Edges", stats['num_edges'])
        c3.metric("Components", stats['num_components'])
        c4.metric("Synthetic bridges", stats['synthetic_bridges'])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Extracted Skeleton**")
            st.image(st.session_state.skeleton * 255, clamp=True, use_container_width=True)
        with col2:
            st.markdown("**Road Graph (healed)**")
            fig = render_graph_plotly(G, title="Healed Road Network")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Topology metrics**")
        st.json({
            'total_road_length_px': round(stats['total_road_length_px'], 1),
            'largest_component_size': stats['largest_component'],
            'avg_node_degree': round(stats['avg_degree'], 2),
            'max_node_degree': stats['max_degree'],
        })
    else:
        st.info("Awaiting input — run the pipeline from the console to build the road graph.")


# ─── TAB 3: Criticality Analysis ───────────────────────────────────────────────

with tabs[2]:
    st.markdown('<div class="phase-tag">Phase III</div>', unsafe_allow_html=True)
    st.subheader("Graph-Theoretic Criticality Analysis")
    st.write(
        "Betweenness centrality identifies road segments and intersections that lie on "
        "the most shortest-paths across the network — the chokepoints whose loss would "
        "fragment the city. Composite scores blend betweenness, closeness, and degree centrality."
    )

    if st.session_state.report is not None:
        G = st.session_state.graph
        report = st.session_state.report

        c1, c2, c3 = st.columns(3)
        c1.metric("Resilience score", f"{report.resilience_score:.3f}",
                  help="0 = fragile / disconnected, 1 = fully redundant network")
        c2.metric("Critical nodes tracked", len(report.top_critical_nodes))
        c3.metric("Critical edges tracked", len(report.top_critical_edges))

        st.markdown("**Network with criticality heatmap**")
        fig = render_graph_plotly(
            G, node_colors=report.node_centrality,
            highlight_edges=[e for e, _ in report.top_critical_edges[:10]],
            title="Betweenness Centrality Heatmap (top-10 critical edges highlighted)"
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top critical nodes**")
            node_table = [
                {'rank': i + 1, 'node': str(n), 'score': round(s, 4)}
                for i, (n, s) in enumerate(report.top_critical_nodes[:10])
            ]
            st.dataframe(node_table, use_container_width=True, hide_index=True)
        with col2:
            st.markdown("**Top critical edges**")
            edge_table = [
                {'rank': i + 1, 'edge': f"{u}↔{v}", 'score': round(s, 4)}
                for i, ((u, v), s) in enumerate(report.top_critical_edges[:10])
            ]
            st.dataframe(edge_table, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown(
            '<div class="phase-tag">Stress test</div>'
            '<div style="font-family:var(--mono); font-size:0.92rem; color:var(--ink-bright); '
            'margin-bottom:0.4rem;">Sequential ablation — top critical edges removed in rank order</div>',
            unsafe_allow_html=True
        )
        st.caption(
            "Each step removes the next most critical edge and re-measures network "
            "reachability — this is the curve that tells you how many failures the "
            "network can absorb before it fragments."
        )

        if st.session_state.stress_results:
            stress = st.session_state.stress_results
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=[r['step'] for r in stress], y=[r['reachability'] for r in stress],
                mode='lines+markers', name='Reachability',
                line=dict(color='#ff9d2e', width=2.2),
                marker=dict(size=6, color='#ff9d2e')
            ))
            fig2.add_trace(go.Scatter(
                x=[r['step'] for r in stress], y=[r['lcc_fraction'] for r in stress],
                mode='lines+markers', name='Largest component fraction',
                line=dict(color='#4a9b8e', width=2, dash='dot'),
                marker=dict(size=6, color='#4a9b8e')
            ))
            fig2.update_layout(
                title=dict(text="NETWORK DEGRADATION UNDER SEQUENTIAL EDGE REMOVAL",
                           font=dict(family='JetBrains Mono, monospace', size=12, color='#d8dee2')),
                xaxis_title="Edges removed (ranked by criticality)",
                yaxis_title="Metric value", height=400,
                plot_bgcolor='#0d1216', paper_bgcolor='#11161b',
                font=dict(family='JetBrains Mono, monospace', color='#748089', size=11),
                legend=dict(font=dict(color='#748089', size=10), bgcolor='rgba(0,0,0,0)'),
                xaxis=dict(gridcolor='#1c2329', zerolinecolor='#232c33'),
                yaxis=dict(gridcolor='#1c2329', zerolinecolor='#232c33'),
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(stress, use_container_width=True, hide_index=True)
    else:
        st.info("Awaiting input — run the pipeline from the console to compute criticality scores.")


# ─── TAB 4: Disaster Simulator ──────────────────────────────────────────────────

with tabs[3]:
    st.markdown('<div class="phase-tag">Phase IV</div>', unsafe_allow_html=True)
    st.subheader("Interactive Disaster Simulation")
    st.write(
        "Simulate a localized failure event and observe network fragmentation, "
        "surviving evacuation routes, and emergent bottlenecks in real time."
    )

    if st.session_state.graph is not None:
        G = st.session_state.graph
        nodes = list(G.nodes())

        col1, col2, col3 = st.columns(3)
        with col1:
            event_type = st.selectbox(
                "Disaster type",
                ["flood", "earthquake", "bridge_failure", "random"],
                help="Flood: radial decay from epicenter. Earthquake: stochastic. "
                     "Bridge failure: sharp localized cut. Random: uniform probability."
            )
        with col2:
            radius = st.slider("Impact radius (px)", 20, 300, 80)
        with col3:
            run_disaster = st.button("💥 Trigger Event", use_container_width=True)

        if run_disaster:
            analyzer = st.session_state.get('analyzer') or get_analyzer(config['analysis'])
            sim = analyzer.run_disaster(G, event_type=event_type, radius=radius)
            st.session_state.disaster_sim = sim

        if st.session_state.disaster_sim is not None:
            sim = st.session_state.disaster_sim
            ab = sim.ablation_result

            sev_class = f"severity-{ab.severity}"
            st.markdown(
                f"### Impact severity: <span class='{sev_class}'>{ab.severity.upper()}</span>",
                unsafe_allow_html=True
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Edges destroyed", len(sim.affected_edges))
            c2.metric("Nodes isolated", ab.isolated_nodes)
            c3.metric("Reachability drop",
                      f"{(ab.original_reachability - ab.new_reachability)*100:.1f}%")
            c4.metric("Impact score", f"{ab.impact_score:.3f}")

            fig = render_graph_plotly(
                G, destroyed_edges=sim.affected_edges,
                epicenter=sim.epicenter, radius=sim.radius,
                title=f"{event_type.replace('_', ' ').upper()} — IMPACT ZONE"
            )
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Surviving evacuation routes**")
                if sim.evacuation_routes:
                    for i, route in enumerate(sim.evacuation_routes):
                        st.write(f"Route {i+1}: {route['hops']} hops, "
                                 f"{route['length']:.1f}px total length")
                else:
                    st.warning("No evacuation routes found — epicenter area fully isolated.")
            with col2:
                st.markdown("**Emergent bottleneck nodes**")
                if sim.bottleneck_nodes:
                    for n in sim.bottleneck_nodes:
                        st.write(f"Node {n} — degree {G.degree(n) if n in G else 'removed'}")
                else:
                    st.write("No significant bottlenecks detected.")
    else:
        st.info("Awaiting input — run the pipeline from the console before simulating disasters.")


# ─── TAB 5: Report ──────────────────────────────────────────────────────────────

with tabs[4]:
    st.markdown('<div class="phase-tag">Summary</div>', unsafe_allow_html=True)
    st.subheader("Pipeline Report")

    if st.session_state.pipeline_ran:
        stats = st.session_state.graph_stats
        report = st.session_state.report

        st.markdown("#### Phase I — Segmentation")
        st.write(f"- Resolution: {st.session_state.image_size}×{st.session_state.image_size}")
        st.write(f"- Road coverage: {st.session_state.mask.mean()*100:.1f}%")

        st.markdown("#### Phase II — Graph Construction")
        st.write(f"- Nodes: {stats['num_nodes']}, Edges: {stats['num_edges']}")
        st.write(f"- Connected components: {stats['num_components']}")
        st.write(f"- Synthetic bridges added during healing: {stats['synthetic_bridges']}")

        st.markdown("#### Phase III — Criticality")
        st.write(f"- Overall resilience score: **{report.resilience_score:.3f}**")
        st.write(f"- Vulnerability zones identified: {len(report.vulnerability_zones)}")

        if st.session_state.disaster_sim:
            sim = st.session_state.disaster_sim
            st.markdown("#### Phase IV — Last Disaster Simulation")
            st.write(f"- Event: {sim.event_type}, severity: **{sim.ablation_result.severity}**")
            st.write(f"- Evacuation routes found: {len(sim.evacuation_routes)}")

        st.divider()
        st.download_button(
            "📥 Download criticality report (JSON)",
            data=str({
                'resilience_score': report.resilience_score,
                'graph_stats': stats,
                'top_critical_nodes': [[str(n), s] for n, s in report.top_critical_nodes[:10]],
            }),
            file_name="route_resilience_report.json",
            mime="application/json",
        )
    else:
        st.info("Awaiting input — run the pipeline from the console to generate a report.")

st.divider()
st.caption("Route Resilience — occlusion-robust extraction & graph-theoretic criticality analysis pipeline.")
