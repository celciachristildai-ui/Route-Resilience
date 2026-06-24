"""
scripts/run_pipeline.py
End-to-end CLI runner: image -> mask -> graph -> criticality report.

Usage:
    python scripts/run_pipeline.py --image path/to/satellite.tif --output outputs/
    python scripts/run_pipeline.py --demo --output outputs/        # synthetic demo
"""

import argparse
import os
import sys
import json
import yaml
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.phase1_segmentation.model import RoadSegmentor
from src.phase2_graph.graph_builder import GraphBuilder
from src.phase3_analysis.analyzer import NetworkAnalyzer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_image(path: str) -> np.ndarray:
    """Load an image (including .tif) as RGB numpy array."""
    try:
        import rasterio
        with rasterio.open(path) as src:
            arr = src.read()
            if arr.shape[0] >= 3:
                arr = arr[:3]
            else:
                arr = np.repeat(arr, 3, axis=0)
            arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                arr = (255 * (arr - arr.min()) / max(1, arr.max() - arr.min())).astype(np.uint8)
            return arr
    except Exception:
        from PIL import Image
        return np.array(Image.open(path).convert('RGB'))


def _json_safe(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    return obj


def save_outputs(output_dir: str, mask, prob_map, skeleton, G, report, stress_results):
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'graphs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'heatmaps'), exist_ok=True)

    from PIL import Image
    Image.fromarray((mask * 255).astype(np.uint8)).save(
        os.path.join(output_dir, 'masks', 'road_mask.png'))
    Image.fromarray((prob_map * 255).astype(np.uint8)).save(
        os.path.join(output_dir, 'heatmaps', 'probability_map.png'))
    Image.fromarray((skeleton * 255).astype(np.uint8)).save(
        os.path.join(output_dir, 'masks', 'skeleton.png'))

    import networkx as nx
    G_export = nx.Graph()
    for n, d in G.nodes(data=True):
        G_export.add_node(str(n), **{k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
                                       for k, v in d.items()})
    for u, v, d in G.edges(data=True):
        d_clean = {k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v)
                   for k, v in d.items() if k != 'pixel_path'}
        G_export.add_edge(str(u), str(v), **d_clean)
    nx.write_gexf(G_export, os.path.join(output_dir, 'graphs', 'road_graph.gexf'))

    report_dict = {
        'resilience_score': report.resilience_score,
        'summary': report.summary(),
        'top_critical_nodes': [[str(n), round(s, 4)] for n, s in report.top_critical_nodes],
        'top_critical_edges': [[f"{u}->{v}", round(s, 4)] for (u, v), s in report.top_critical_edges],
        'vulnerability_zones': [[round(y, 1), round(x, 1), round(s, 4)]
                                  for y, x, s in report.vulnerability_zones],
        'stress_test': stress_results,
    }
    with open(os.path.join(output_dir, 'criticality_report.json'), 'w') as f:
        json.dump(_json_safe(report_dict), f, indent=2)

    print(f"\n[Output] All artifacts saved to: {output_dir}")
    print(f"  - masks/road_mask.png")
    print(f"  - masks/skeleton.png")
    print(f"  - heatmaps/probability_map.png")
    print(f"  - graphs/road_graph.gexf")
    print(f"  - criticality_report.json")


def main():
    parser = argparse.ArgumentParser(description="Route Resilience full pipeline runner")
    parser.add_argument('--image', type=str, default=None, help="Path to satellite image")
    parser.add_argument('--demo', action='store_true', help="Use synthetic demo network")
    parser.add_argument('--output', type=str, default='outputs/', help="Output directory")
    parser.add_argument('--config', type=str, default='configs/pipeline.yaml')
    parser.add_argument('--checkpoint', type=str, default=None, help="Model checkpoint path")
    parser.add_argument('--disaster', type=str, default=None,
                        choices=['flood', 'earthquake', 'bridge_failure', 'random'],
                        help="Optionally run a disaster simulation")
    args = parser.parse_args()

    if not args.image and not args.demo:
        print("No --image provided; falling back to --demo synthetic network.")
        args.demo = True

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), '..', config_path)
    config = load_config(config_path)

    print("=" * 70)
    print(" ROUTE RESILIENCE PIPELINE")
    print("=" * 70)

    # Phase I
    print("\n--- PHASE I: SEGMENTATION ---")
    segmentor = RoadSegmentor(config['segmentation'], device='cpu')
    if args.checkpoint:
        segmentor.load_checkpoint(args.checkpoint)

    if args.demo:
        print("[Phase I] Generating synthetic demo road network...")
        mask, prob_map = segmentor.generate_synthetic_demo(
            size=config['segmentation']['input_size'][0])
    else:
        print(f"[Phase I] Loading image: {args.image}")
        image = load_image(args.image)
        size = config['segmentation']['input_size'][0]
        if image.shape[0] != size or image.shape[1] != size:
            from PIL import Image as PILImage
            image = np.array(PILImage.fromarray(image).resize((size, size)))
        mask, prob_map = segmentor.segment(image)

    print(f"[Phase I] Road pixel coverage: {mask.mean()*100:.2f}%")

    # Phase II
    print("\n--- PHASE II: GRAPH SKELETONIZATION & HEALING ---")
    builder = GraphBuilder(config['graph'])
    G, skeleton = builder.build(mask)
    graph_stats = builder.graph_stats(G)
    print(f"[Phase II] Final graph stats: {graph_stats}")

    # Phase III
    print("\n--- PHASE III: CRITICALITY ANALYSIS ---")
    analyzer = NetworkAnalyzer(config['analysis'])
    report = analyzer.analyze(G)
    stress_results = analyzer.stress_test(G, report)

    print(f"\n[Phase III] Resilience score: {report.resilience_score:.4f}")
    print(f"[Phase III] Top 5 critical nodes:")
    for n, s in report.top_critical_nodes[:5]:
        print(f"    {n} -> {s:.4f}")

    if args.disaster:
        print(f"\n--- BONUS: DISASTER SIMULATION ({args.disaster}) ---")
        sim = analyzer.run_disaster(G, event_type=args.disaster)
        print(f"[Disaster] Severity: {sim.ablation_result.severity}")
        print(f"[Disaster] Edges destroyed: {len(sim.affected_edges)}")
        print(f"[Disaster] Evacuation routes found: {len(sim.evacuation_routes)}")

    # Save
    print("\n--- SAVING OUTPUTS ---")
    output_dir = args.output
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(__file__), '..', output_dir)
    save_outputs(output_dir, mask, prob_map, skeleton, G, report, stress_results)

    print("\n" + "=" * 70)
    print(" PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
