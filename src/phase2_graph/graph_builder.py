"""
Phase II: Graph Skeletonization & Topological Healing
Converts binary road mask → topologically consistent road graph.
Uses MST-based gap bridging and Union-Find for component healing.
"""

import numpy as np
import networkx as nx
from skimage.morphology import skeletonize, remove_small_objects, binary_dilation, disk
from skimage.measure import label, regionprops
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
from typing import Tuple, List, Dict, Optional, Set
import warnings
warnings.filterwarnings('ignore')


# ─── Union-Find (Disjoint Set Union) ──────────────────────────────────────────

class UnionFind:
    """Path-compressed Union-Find for component merging."""
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)

    def component_size(self, x: int) -> int:
        return self.size[self.find(x)]


# ─── Skeleton Extraction ──────────────────────────────────────────────────────

class RoadSkeleton:
    """Extract pixel-level skeleton from binary road mask."""

    @staticmethod
    def clean_mask(mask: np.ndarray, min_area: int = 100) -> np.ndarray:
        """Remove noise and small disconnected fragments."""
        cleaned = remove_small_objects(mask.astype(bool), min_size=min_area)
        # Close small gaps
        cleaned = binary_dilation(cleaned, disk(2))
        return cleaned.astype(np.uint8)

    @staticmethod
    def extract_skeleton(mask: np.ndarray) -> np.ndarray:
        """Medial axis skeletonization preserving topology."""
        cleaned = RoadSkeleton.clean_mask(mask)
        skeleton = skeletonize(cleaned > 0)
        return skeleton.astype(np.uint8)

    @staticmethod
    def get_junction_points(skeleton: np.ndarray) -> np.ndarray:
        """Find skeleton junction pixels (degree ≥ 3)."""
        kernel_coords = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        junctions = []
        ys, xs = np.where(skeleton > 0)
        for y, x in zip(ys, xs):
            count = sum(
                1 for dy, dx in kernel_coords
                if 0 <= y+dy < skeleton.shape[0] and
                   0 <= x+dx < skeleton.shape[1] and
                   skeleton[y+dy, x+dx] > 0
            )
            if count >= 3:
                junctions.append((int(y), int(x)))
        return np.array(junctions) if junctions else np.empty((0, 2), dtype=int)

    @staticmethod
    def get_endpoints(skeleton: np.ndarray) -> np.ndarray:
        """Find skeleton endpoints (degree == 1)."""
        kernel_coords = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        endpoints = []
        ys, xs = np.where(skeleton > 0)
        for y, x in zip(ys, xs):
            count = sum(
                1 for dy, dx in kernel_coords
                if 0 <= y+dy < skeleton.shape[0] and
                   0 <= x+dx < skeleton.shape[1] and
                   skeleton[y+dy, x+dx] > 0
            )
            if count == 1:
                endpoints.append((int(y), int(x)))
        return np.array(endpoints) if endpoints else np.empty((0, 2), dtype=int)


# ─── Graph Construction ───────────────────────────────────────────────────────

class SkeletonToGraph:
    """Convert skeleton pixels into a proper NetworkX road graph."""

    def __init__(self, min_branch_length: int = 15):
        self.min_branch_length = min_branch_length

    def build(self, skeleton: np.ndarray) -> nx.Graph:
        """
        Build graph from skeleton via connected component tracing.
        Nodes = junctions + endpoints; Edges = road segments.
        """
        G = nx.Graph()
        junctions = set(tuple(int(v) for v in p) for p in RoadSkeleton.get_junction_points(skeleton))
        endpoints = set(tuple(int(v) for v in p) for p in RoadSkeleton.get_endpoints(skeleton))
        key_points = junctions | endpoints

        if not key_points:
            # Fallback: every skeleton pixel becomes a node
            ys, xs = np.where(skeleton > 0)
            for y, x in zip(ys, xs):
                G.add_node((y, x), y=y, x=x, node_type='pixel')
            return G

        # BFS trace from each key point
        visited_edges: Set = set()
        for start in key_points:
            y, x = start
            G.add_node(start, y=y, x=x,
                       node_type='junction' if start in junctions else 'endpoint')
            for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                ny, nx_ = y + dy, x + dx
                if (0 <= ny < skeleton.shape[0] and 0 <= nx_ < skeleton.shape[1]
                        and skeleton[ny, nx_] > 0):
                    path = self._trace_branch(skeleton, start, (ny, nx_), key_points)
                    if path:
                        end = path[-1]
                        if end not in G:
                            G.add_node(end, y=end[0], x=end[1],
                                       node_type='junction' if end in junctions
                                       else ('endpoint' if end in endpoints else 'deadend'))
                        edge_key = (min(start, end), max(start, end))
                        if edge_key not in visited_edges:
                            visited_edges.add(edge_key)
                            length = self._path_length(path)
                            if length >= self.min_branch_length:
                                G.add_edge(start, end,
                                           length=length,
                                           pixel_path=path,
                                           weight=length)

        return G

    def _trace_branch(self, skeleton, start, first_step, key_points):
        """Follow skeleton pixels until hitting another key point."""
        path = [start, first_step]
        prev, cur = start, first_step
        max_steps = skeleton.shape[0] * skeleton.shape[1]
        for _ in range(max_steps):
            if cur in key_points and cur != start:
                return path
            neighbors = []
            y, x = cur
            for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                ny, nx_ = y + dy, x + dx
                nbr = (ny, nx_)
                if (0 <= ny < skeleton.shape[0] and 0 <= nx_ < skeleton.shape[1]
                        and skeleton[ny, nx_] > 0 and nbr != prev):
                    neighbors.append(nbr)
            if not neighbors:
                return path   # dead end
            next_pt = neighbors[0]
            path.append(next_pt)
            prev, cur = cur, next_pt
        return path

    @staticmethod
    def _path_length(path: List[Tuple]) -> float:
        length = 0.0
        for i in range(1, len(path)):
            dy = path[i][0] - path[i-1][0]
            dx = path[i][1] - path[i-1][1]
            length += math.sqrt(dy**2 + dx**2)
        return length


# ─── Topological Healing ──────────────────────────────────────────────────────

import math

class TopologicalHealer:
    """
    Bridge disconnected road components using:
    1. MST-based nearest-neighbor gap filling
    2. Occlusion-aware endpoint snapping
    """

    def __init__(self, max_gap: float = 50.0, snap_tolerance: float = 10.0):
        self.max_gap = max_gap
        self.snap_tolerance = snap_tolerance

    def heal(self, G: nx.Graph, mask: np.ndarray) -> nx.Graph:
        """Full healing pipeline."""
        G = self._snap_endpoints(G)
        G = self._bridge_components_mst(G, mask)
        G = self._remove_isolated_nodes(G)
        return G

    def _snap_endpoints(self, G: nx.Graph) -> nx.Graph:
        """Snap near-coincident endpoints to merge micro-gaps."""
        endpoints = [n for n, d in G.degree() if d == 1]
        if len(endpoints) < 2:
            return G
        coords = np.array([
            (G.nodes[n].get('y', n[0] if isinstance(n, tuple) else 0),
             G.nodes[n].get('x', n[1] if isinstance(n, tuple) else 0))
            for n in endpoints
        ])
        tree = cKDTree(coords)
        pairs = tree.query_pairs(self.snap_tolerance)
        for i, j in pairs:
            u, v = endpoints[i], endpoints[j]
            if u in G and v in G and u != v:
                dist = math.hypot(
                    G.nodes[u]['y'] - G.nodes[v]['y'],
                    G.nodes[u]['x'] - G.nodes[v]['x']
                )
                if not G.has_edge(u, v):
                    G.add_edge(u, v, length=dist, weight=dist,
                               bridge=True, synthetic=True)
        return G

    def _bridge_components_mst(self, G: nx.Graph, mask: np.ndarray) -> nx.Graph:
        """
        Find disconnected components and bridge them using MST on
        inter-component distances. Only bridges gaps ≤ max_gap.
        """
        components = list(nx.connected_components(G))
        if len(components) <= 1:
            return G

        # Build list of (component_id, node, y, x) for all nodes
        node_info = []
        comp_map = {}
        for ci, comp in enumerate(components):
            for n in comp:
                y = G.nodes[n].get('y', n[0] if isinstance(n, tuple) else 0)
                x = G.nodes[n].get('x', n[1] if isinstance(n, tuple) else 0)
                node_info.append((ci, n, y, x))
                comp_map[n] = ci

        # For each component pair, find minimum-distance node pair
        coords = np.array([(info[2], info[3]) for info in node_info])
        tree = cKDTree(coords)

        uf = UnionFind(len(components))
        bridge_edges = []

        for idx, (ci, n, y, x) in enumerate(node_info):
            dists, idxs = tree.query([y, x], k=min(20, len(node_info)))
            for dist, j in zip(dists, idxs):
                if dist > self.max_gap:
                    break
                cj = node_info[j][1]
                other_ci = comp_map[cj]
                if other_ci != ci and dist > 0:
                    bridge_edges.append((dist, n, node_info[j][1], dist))

        # Sort by distance and greedily bridge via Kruskal-like approach
        bridge_edges.sort(key=lambda e: e[0])
        for dist, u, v, length in bridge_edges:
            cu, cv = comp_map[u], comp_map[v]
            if not uf.connected(cu, cv):
                uf.union(cu, cv)
                G.add_edge(u, v, length=length, weight=length,
                           bridge=True, synthetic=True, gap_distance=dist)

        return G

    def _remove_isolated_nodes(self, G: nx.Graph) -> nx.Graph:
        """Remove degree-0 nodes (noise)."""
        isolated = [n for n in G.nodes() if G.degree(n) == 0]
        G.remove_nodes_from(isolated)
        return G


# ─── Main Pipeline ────────────────────────────────────────────────────────────

class GraphBuilder:
    """Orchestrates the full mask → healed graph pipeline."""

    def __init__(self, config: dict):
        self.config = config
        self.skeleton_extractor = RoadSkeleton()
        self.graph_builder = SkeletonToGraph(
            min_branch_length=config.get('min_branch_length', 15)
        )
        self.healer = TopologicalHealer(
            max_gap=config.get('max_gap_fill', 50),
            snap_tolerance=config.get('snap_tolerance', 10)
        )

    def build(self, mask: np.ndarray) -> Tuple[nx.Graph, np.ndarray]:
        """
        Full pipeline: binary mask → healed road graph.
        Returns: (graph, skeleton_image)
        """
        print("[Phase II] Extracting skeleton...")
        skeleton = self.skeleton_extractor.extract_skeleton(mask)

        print("[Phase II] Building initial graph...")
        G = self.graph_builder.build(skeleton)
        n_init = G.number_of_nodes()
        e_init = G.number_of_edges()
        print(f"           Initial graph: {n_init} nodes, {e_init} edges")

        print("[Phase II] Healing topology...")
        G = self.healer.heal(G, mask)

        # Add metadata
        components = list(nx.connected_components(G))
        n_comp = len(components)
        print(f"           Healed graph: {G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges, {n_comp} components")

        # Annotate component IDs
        for ci, comp in enumerate(components):
            for n in comp:
                G.nodes[n]['component'] = ci

        return G, skeleton

    def graph_stats(self, G: nx.Graph) -> Dict:
        """Compute basic topological statistics."""
        components = list(nx.connected_components(G))
        total_length = sum(d.get('length', 1.0) for _, _, d in G.edges(data=True))
        degrees = [G.degree(n) for n in G.nodes()]
        return {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'num_components': len(components),
            'largest_component': max(len(c) for c in components) if components else 0,
            'total_road_length_px': total_length,
            'avg_degree': np.mean(degrees) if degrees else 0,
            'max_degree': max(degrees) if degrees else 0,
            'synthetic_bridges': sum(1 for _, _, d in G.edges(data=True) if d.get('synthetic')),
        }
