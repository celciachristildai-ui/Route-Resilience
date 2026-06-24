# Route Resilience

Occlusion-robust road extraction and graph-theoretic criticality analysis for urban satellite imagery.

```
Phase I   → Occlusion-Robust Segmentation (Swin-UNet Transformer)
Phase II  → Graph Skeletonization & Topological Healing (MST + Disjoint Sets)
Phase III → Network Analysis & Stress Testing (Betweenness Centrality + Ablation)
Phase IV  → Interactive Dashboard (Streamlit + Plotly)
```

## What each phase does

**Phase I — Segmentation.** A Swin-UNet (shifted-window transformer encoder/decoder
with skip connections) segments road pixels from satellite tiles. The windowed
self-attention gives every patch context from across the whole tile, which is what
lets the model infer a road continues under a tree canopy, a parked truck, or a
cloud shadow instead of just stopping at the occlusion boundary.

**Phase II — Skeletonization & healing.** The binary mask is thinned to a 1px-wide
medial-axis skeleton, then traced into a graph: junctions and endpoints become
nodes, the pixel runs between them become weighted edges. Occlusion in Phase I
reliably produces disconnected fragments, so a healing pass snaps near-coincident
endpoints and bridges remaining components using a minimum-spanning-tree heuristic
over inter-component nearest-neighbor distances (Union-Find guarantees no redundant
bridges and no cycles introduced purely by the healer).

**Phase III — Criticality analysis.** Betweenness centrality (node + edge) finds
the chokepoints — intersections and segments that sit on the most shortest-paths
across the city. A sequential ablation stress test removes the top-K critical
edges one at a time and tracks reachability and largest-component-fraction decay.
A disaster simulator (`flood`, `earthquake`, `bridge_failure`, `random`) damages a
local region and reports surviving evacuation routes and emergent bottlenecks.

**Phase IV — Dashboard.** Streamlit app wiring all three phases together with
live Plotly visualizations: segmentation masks, the healed graph, a centrality
heatmap, ablation degradation curves, and an interactive disaster trigger.

## Setup

```bash
pip install -r requirements.txt
```

If `torch` fails to install from the default index in your environment, install
the CPU wheel explicitly:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Run the full pipeline (CLI)

```bash
# Synthetic demo network (no image needed, good for a first smoke test)
python scripts/run_pipeline.py --demo --output outputs/

# Real satellite image
python scripts/run_pipeline.py --image path/to/satellite.tif --output outputs/

# With a disaster scenario appended to the run
python scripts/run_pipeline.py --demo --output outputs/ --disaster flood
```

Outputs land in `outputs/`:
- `masks/road_mask.png`, `masks/skeleton.png`
- `heatmaps/probability_map.png`
- `graphs/road_graph.gexf` (open in Gephi, or `nx.read_gexf` in Python)
- `criticality_report.json`

## Deploying (Streamlit Community Cloud — free, recommended)

1. Push this folder to a **public GitHub repo**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, pick your repo, set:
   - **Main file path**: `src/phase4_dashboard/app.py`
   - **Branch**: `main` (or whichever you pushed)
4. Click **Deploy**. First build takes 3-5 minutes (installing torch CPU + the rest).
5. You'll get a public URL like `https://yourname-route-resilience.streamlit.app` — this is
   what you submit / put in your README / share with judges.

The repo already includes everything Streamlit Cloud needs:
- `requirements.txt` — pinned to the **CPU-only PyTorch wheel** via
  `--extra-index-url https://download.pytorch.org/whl/cpu`, which keeps the install to
  ~200MB instead of ~2.5GB (the default PyPI torch pulls full CUDA libraries you'll
  never use on a CPU-only cloud instance — without this pin, the build can fail on
  free-tier disk/memory limits). Uses `opencv-python-headless` specifically (not
  `opencv-python`) so no system GL/GLib libraries are needed at all — no `packages.txt`
  required.
- `runtime.txt` — pins Python 3.11.
- `.streamlit/config.toml` — locks in the dark console theme and disables
  Streamlit's usage-stats ping.

If you instead want to deploy on **Render**, **Railway**, or your own server: the same
`requirements.txt` works as-is, but you must set the start command to bind to the
platform's `$PORT` env var, e.g.:
```bash
streamlit run src/phase4_dashboard/app.py --server.port $PORT --server.address 0.0.0.0
```

## Run locally

```bash
streamlit run src/phase4_dashboard/app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).

In the sidebar:
1. Pick **synthetic demo** (instant, no checkpoint needed) or **upload an image**.
2. Tune healing parameters (min branch length, max bridge gap, snap tolerance).
3. Tune the centrality/ablation depth.
4. Hit **Run Full Pipeline**.
5. Walk through the tabs: Segmentation → Graph & Healing → Criticality →
   Disaster Simulator → Report.

## Using a trained checkpoint

The segmentation model is fully wired but ships untrained (random weights) so the
demo runs out of the box. To use real predictions, train a Swin-UNet checkpoint
against a labeled road dataset (e.g. SpaceNet, DeepGlobe, Massachusetts Roads) and
point either entry point at it:

```bash
python scripts/run_pipeline.py --image satellite.tif --checkpoint models/swin_unet.pt
```

In the dashboard, drop the checkpoint path into `configs/pipeline.yaml` under
`segmentation.checkpoint` and it'll be picked up by `RoadSegmentor.load_checkpoint`.

## Project layout

```
route_resilience/
├── configs/pipeline.yaml          # all phase hyperparameters
├── src/
│   ├── phase1_segmentation/model.py   # SwinUNet + RoadSegmentor inference wrapper
│   ├── phase2_graph/graph_builder.py  # RoadSkeleton, SkeletonToGraph, TopologicalHealer
│   ├── phase3_analysis/analyzer.py    # CentralityAnalyzer, AblationEngine, DisasterSimulator
│   └── phase4_dashboard/app.py        # Streamlit dashboard
├── scripts/run_pipeline.py        # CLI runner, image → report
└── outputs/                       # generated masks, graphs, reports
```

## Notes on the synthetic demo

When no image is supplied, `RoadSegmentor.generate_synthetic_demo()` draws a
plausible grid-plus-diagonal road network and stamps random occlusion blobs onto
it — enough irregularity that the healing and criticality phases have real gaps
to bridge and real chokepoints to find, without needing a dataset on hand.
