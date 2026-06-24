"""
Phase III: Network Criticality Analysis & Disaster Stress Testing
Betweenness centrality, edge ablation, resilience scoring, and failure simulation.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class CriticalityReport:
    """Results of criticality analysis for the road network."""
    node_centrality: Dict        # node -> betweenness score
    edge_centrality: Dict        # edge -> betweenness score
    top_critical_nodes: List     # ranked list of critical nodes
    top_critical_edges: List     # ranked list of critical edges
    resilience_score: float      # 0–1 network resilience
    connectivity_matrix: np.ndarray
    vulnerability_zones: List[Tuple[float, float, float]]  # (y, x, severity)

    def summary(self) -> Dict:
        return {
            'resilience_score': round(self.resilience_score, 4),
            'critical_nodes': len(self.top_critical_nodes),
            'critical_edges': len(self.top_critical_edges),
            'vulnerability_zones': len(self.vulnerability_zones),
        }


@dataclass
class AblationResult:
    """Result of removing a set of edges/nodes from the network."""
    removed: List                    # removed elements
    original_components: int
    new_components: int
    original_reachability: float
    new_reachability: float
    isolated_nodes: int
    largest_component_fraction: float
    severity: str                    # 'low', 'medium', 'high', 'critical'

    @property
    def impact_score(self) -> float:
        """0–1 impact score."""
        connectivity_loss = max(0, self.original_reachability - self.new_reachability)
        component_penalty = min(1, (self.new_components - self.original_components) / 10)
        isolation_penalty = min(1, self.isolated_nodes / max(1, self.isolated_nodes + 10))
        return min(1.0, (connectivity_loss * 0.5 + component_penalty * 0.3 + isolation_penalty * 0.2))


@dataclass
class DisasterSimulation:
    """Result of a disaster event simulation."""
    event_type: str              # 'flood', 'earthquake', 'bridge_failure', 'random'
    epicenter: Tuple[float, float]
    radius: float
    affected_edges: List
    affected_nodes: List
    ablation_result: AblationResult
    evacuation_routes: List      # surviving paths from epicenter to network boundary
    bottleneck_nodes: List


# ─── Centrality Analysis ──────────────────────────────────────────────────────

class CentralityAnalyzer:
    """
    Compute betweenness, closeness, and eigenvector centrality
    for road network criticality ranking.
    """

    def __init__(self, weight: str = 'length'):
        self.weight = weight

    def compute_betweenness(self, G: nx.Graph, k: Optional[int] = None) -> Dict:
        """
        Edge + node betweenness centrality.
        k: number of pivot nodes for approximate computation (None = exact).
        """
        if G.number_of_nodes() < 2:
            return {}, {}

        # Use approximate for large graphs
        if k is None and G.number_of_nodes() > 500:
            k = min(200, G.number_of_nodes())

        node_bc = nx.betweenness_centrality(
            G, k=k, weight=self.weight, normalized=True)
        edge_bc = nx.edge_betweenness_centrality(
            G, k=k, weight=self.weight, normalized=True)

        return node_bc, edge_bc

    def compute_closeness(self, G: nx.Graph) -> Dict:
        """Closeness centrality per node."""
        try:
            return nx.closeness_centrality(G, distance=self.weight)
        except Exception:
            return {n: 0.0 for n in G.nodes()}

    def compute_degree_centrality(self, G: nx.Graph) -> Dict:
        return nx.degree_centrality(G)

    def composite_criticality(self, G: nx.Graph, k: int = None) -> Tuple[Dict, Dict]:
        """
        Combined criticality score: weighted sum of betweenness + degree + closeness.
        Returns: (node_scores, edge_scores)
        """
        node_bc, edge_bc = self.compute_betweenness(G, k)
        closeness = self.compute_closeness(G)
        degree = self.compute_degree_centrality(G)

        node_scores = {}
        for n in G.nodes():
            bc = node_bc.get(n, 0)
            cl = closeness.get(n, 0)
            dg = degree.get(n, 0)
            node_scores[n] = 0.5 * bc + 0.3 * cl + 0.2 * dg

        return node_scores, edge_bc

    def identify_vulnerability_zones(self, G: nx.Graph, node_scores: Dict,
                                     top_k: int = 20) -> List[Tuple]:
        """
        Cluster high-centrality nodes into vulnerability zones.
        Returns list of (y, x, severity_0_to_1).
        """
        sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        zones = []
        for node, score in sorted_nodes[:top_k]:
            y = G.nodes[node].get('y', node[0] if isinstance(node, tuple) else 0)
            x = G.nodes[node].get('x', node[1] if isinstance(node, tuple) else 0)
            zones.append((float(y), float(x), float(score)))
        return zones


# ─── Reachability Metrics ─────────────────────────────────────────────────────

class ReachabilityMetrics:
    """Compute network reachability and connectivity."""

    @staticmethod
    def pairwise_reachability(G: nx.Graph, sample_size: int = 100) -> float:
        """
        Fraction of node pairs that are mutually reachable.
        Sampled for large graphs.
        """
        nodes = list(G.nodes())
        if len(nodes) < 2:
            return 0.0

        if len(nodes) <= sample_size:
            sample_nodes = nodes
        else:
            idxs = np.random.choice(len(nodes), sample_size, replace=False)
            sample_nodes = [nodes[i] for i in idxs]

        reachable_pairs = 0
        total_pairs = 0
        components = {n: nx.node_connected_component(G, n) for n in sample_nodes}

        for i, u in enumerate(sample_nodes):
            for v in sample_nodes[i+1:]:
                total_pairs += 1
                if v in components.get(u, set()):
                    reachable_pairs += 1

        return reachable_pairs / max(1, total_pairs)

    @staticmethod
    def largest_component_fraction(G: nx.Graph) -> float:
        """Size of largest connected component / total nodes."""
        if G.number_of_nodes() == 0:
            return 0.0
        components = list(nx.connected_components(G))
        return max(len(c) for c in components) / G.number_of_nodes()

    @staticmethod
    def network_efficiency(G: nx.Graph, sample_size: int = 50) -> float:
        """
        Global network efficiency: mean of 1/shortest_path_length.
        """
        nodes = list(G.nodes())
        if len(nodes) < 2:
            return 0.0
        idxs = np.random.choice(len(nodes), min(sample_size, len(nodes)), replace=False)
        sample = [nodes[i] for i in idxs]
        total, count = 0.0, 0
        for u in sample:
            lengths = nx.single_source_shortest_path_length(G, u)
            for v, d in lengths.items():
                if v != u and d > 0:
                    total += 1.0 / d
                    count += 1
        return total / max(1, count)


# ─── Ablation Engine ──────────────────────────────────────────────────────────

class AblationEngine:
    """
    Stress test the road network by removing edges/nodes
    and measuring degradation.
    """

    def __init__(self):
        self.metrics = ReachabilityMetrics()

    def ablate_edges(self, G: nx.Graph, edges: List) -> AblationResult:
        """Remove edges and measure impact."""
        G2 = G.copy()
        G2.remove_edges_from([e for e in edges if G2.has_edge(*e)])

        return self._measure_impact(G, G2, removed=edges)

    def ablate_nodes(self, G: nx.Graph, nodes: List) -> AblationResult:
        """Remove nodes (and incident edges) and measure impact."""
        G2 = G.copy()
        G2.remove_nodes_from([n for n in nodes if n in G2])

        return self._measure_impact(G, G2, removed=nodes)

    def _measure_impact(self, G_orig: nx.Graph, G_new: nx.Graph,
                         removed: List) -> AblationResult:
        orig_comps = nx.number_connected_components(G_orig)
        new_comps = nx.number_connected_components(G_new)
        orig_reach = self.metrics.pairwise_reachability(G_orig)
        new_reach = self.metrics.pairwise_reachability(G_new)
        isolated = sum(1 for n in G_new.nodes() if G_new.degree(n) == 0)
        lcc_frac = self.metrics.largest_component_fraction(G_new)

        impact = (orig_reach - new_reach) / max(0.001, orig_reach)
        if impact < 0.1:
            severity = 'low'
        elif impact < 0.3:
            severity = 'medium'
        elif impact < 0.6:
            severity = 'high'
        else:
            severity = 'critical'

        return AblationResult(
            removed=removed,
            original_components=orig_comps,
            new_components=new_comps,
            original_reachability=orig_reach,
            new_reachability=new_reach,
            isolated_nodes=isolated,
            largest_component_fraction=lcc_frac,
            severity=severity,
        )

    def sequential_ablation(self, G: nx.Graph, ranked_edges: List,
                             k: int = 10) -> List[Dict]:
        """
        Iteratively remove top-k critical edges and track degradation.
        Returns list of metrics after each removal.
        """
        results = []
        G_cur = G.copy()
        base_reach = self.metrics.pairwise_reachability(G)

        for i, edge in enumerate(ranked_edges[:k]):
            if G_cur.has_edge(*edge):
                G_cur.remove_edge(*edge)
            reach = self.metrics.pairwise_reachability(G_cur)
            lcc = self.metrics.largest_component_fraction(G_cur)
            results.append({
                'step': i + 1,
                'removed_edge': edge,
                'reachability': round(reach, 4),
                'lcc_fraction': round(lcc, 4),
                'delta_reachability': round(base_reach - reach, 4),
                'components': nx.number_connected_components(G_cur),
            })

        return results


# ─── Disaster Simulator ───────────────────────────────────────────────────────

class DisasterSimulator:
    """
    Simulate urban disaster scenarios on the road network.
    Supports: flood, earthquake, bridge failure, random failures.
    """

    def __init__(self):
        self.ablation = AblationEngine()

    def simulate(self, G: nx.Graph, event_type: str = 'flood',
                 epicenter: Tuple = None, radius: float = 80.0,
                 failure_prob: float = 0.3) -> DisasterSimulation:
        """Run a disaster simulation and return full results."""
        nodes = list(G.nodes())
        if not nodes:
            raise ValueError("Empty graph")

        if epicenter is None:
            # Random epicenter among existing nodes
            n = np.random.choice(len(nodes))
            sample_node = nodes[n]
            epicenter = (
                G.nodes[sample_node].get('y', 0),
                G.nodes[sample_node].get('x', 0)
            )

        affected_nodes, affected_edges = self._identify_affected(
            G, epicenter, radius, event_type, failure_prob)

        ablation = self.ablation.ablate_edges(G, affected_edges)
        G_damaged = G.copy()
        G_damaged.remove_edges_from([e for e in affected_edges if G_damaged.has_edge(*e)])

        evacuation = self._find_evacuation_routes(G_damaged, epicenter, n_routes=3)
        bottlenecks = self._find_bottlenecks(G_damaged, epicenter)

        return DisasterSimulation(
            event_type=event_type,
            epicenter=epicenter,
            radius=radius,
            affected_edges=affected_edges,
            affected_nodes=affected_nodes,
            ablation_result=ablation,
            evacuation_routes=evacuation,
            bottleneck_nodes=bottlenecks,
        )

    def _identify_affected(self, G, epicenter, radius, event_type, prob):
        """Identify nodes and edges affected by disaster."""
        ey, ex = epicenter
        affected_nodes = []
        affected_edges = []

        for node in G.nodes():
            ny = G.nodes[node].get('y', node[0] if isinstance(node, tuple) else 0)
            nx_ = G.nodes[node].get('x', node[1] if isinstance(node, tuple) else 0)
            dist = ((ny - ey)**2 + (nx_ - ex)**2) ** 0.5

            if dist <= radius:
                # Damage probability varies by event type
                if event_type == 'flood':
                    p = max(0, 1 - dist / radius)   # higher at center
                elif event_type == 'earthquake':
                    p = prob * (1 + 0.5 * np.random.randn())  # stochastic
                elif event_type == 'bridge_failure':
                    p = 1.0 if dist < radius * 0.3 else 0.0   # localized
                else:  # random
                    p = prob

                if np.random.random() < max(0, min(1, p)):
                    affected_nodes.append(node)

        for u, v in G.edges():
            if u in affected_nodes or v in affected_nodes:
                affected_edges.append((u, v))

        return affected_nodes, affected_edges

    def _find_evacuation_routes(self, G, epicenter, n_routes=3):
        """Find surviving paths from epicenter area to network periphery."""
        if G.number_of_nodes() == 0:
            return []

        # Find node closest to epicenter
        ey, ex = epicenter
        nodes = list(G.nodes())
        dists = [
            ((G.nodes[n].get('y', 0) - ey)**2 + (G.nodes[n].get('x', 0) - ex)**2)**0.5
            for n in nodes
        ]
        source_idx = int(np.argmin(dists))
        source = nodes[source_idx]
        if source not in G:
            return []

        # Target: nodes on the network periphery (high degree from source via BFS)
        comp = nx.node_connected_component(G, source)
        if len(comp) < 2:
            return []

        comp_list = list(comp)
        # Sort by BFS distance to get furthest nodes
        try:
            bfs_lengths = nx.single_source_shortest_path_length(G, source)
            far_nodes = sorted(bfs_lengths.items(), key=lambda x: x[1], reverse=True)
            targets = [n for n, _ in far_nodes[:n_routes] if n != source]
        except Exception:
            return []

        routes = []
        for target in targets:
            try:
                path = nx.shortest_path(G, source, target, weight='length')
                path_length = sum(
                    G[path[i]][path[i+1]].get('length', 1)
                    for i in range(len(path)-1)
                )
                routes.append({'path': path, 'length': path_length,
                                'hops': len(path)})
            except nx.NetworkXNoPath:
                pass

        return routes

    def _find_bottlenecks(self, G, epicenter, top_k=5):
        """Identify bottleneck nodes near epicenter."""
        if G.number_of_nodes() < 2:
            return []
        ey, ex = epicenter
        node_bc = nx.betweenness_centrality(G, k=min(50, G.number_of_nodes()), normalized=True)
        # Weight by proximity to epicenter
        scored = []
        for n, bc in node_bc.items():
            ny = G.nodes[n].get('y', 0)
            nx_ = G.nodes[n].get('x', 0)
            dist = ((ny - ey)**2 + (nx_ - ex)**2)**0.5
            proximity = 1 / (1 + dist / 100)
            scored.append((n, bc * proximity))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [n for n, _ in scored[:top_k]]


# ─── Full Analysis Pipeline ───────────────────────────────────────────────────

class NetworkAnalyzer:
    """Orchestrates Phase III: full criticality + stress testing pipeline."""

    def __init__(self, config: dict):
        self.config = config
        self.centrality = CentralityAnalyzer(
            weight=config.get('centrality_weight', 'length'))
        self.ablation = AblationEngine()
        self.simulator = DisasterSimulator()
        self.metrics = ReachabilityMetrics()
        self.top_k = config.get('top_k_critical', 20)

    def analyze(self, G: nx.Graph) -> CriticalityReport:
        """Full criticality analysis."""
        print("[Phase III] Computing centrality...")
        node_scores, edge_scores = self.centrality.composite_criticality(
            G, k=min(100, G.number_of_nodes()))

        sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_edges = sorted(edge_scores.items(), key=lambda x: x[1], reverse=True)

        print("[Phase III] Computing resilience score...")
        reachability = self.metrics.pairwise_reachability(G)
        lcc = self.metrics.largest_component_fraction(G)
        n_comp = nx.number_connected_components(G)
        resilience = (reachability * 0.4 + lcc * 0.4 +
                      (1 / max(1, n_comp)) * 0.2)

        vuln_zones = self.centrality.identify_vulnerability_zones(
            G, node_scores, self.top_k)

        # Placeholder connectivity matrix (top-10 x top-10 subgraph density)
        top_nodes = [n for n, _ in sorted_nodes[:10]]
        n = len(top_nodes)
        conn_mat = np.zeros((n, n))
        for i, u in enumerate(top_nodes):
            for j, v in enumerate(top_nodes):
                if G.has_edge(u, v):
                    conn_mat[i, j] = G[u][v].get('length', 1)

        top_score = sorted_nodes[0][1] if sorted_nodes else 0.0
        print(f"           Resilience score: {resilience:.4f}")
        print(f"           Top critical node score: {top_score:.4f}")

        return CriticalityReport(
            node_centrality=node_scores,
            edge_centrality=edge_scores,
            top_critical_nodes=sorted_nodes[:self.top_k],
            top_critical_edges=sorted_edges[:self.top_k],
            resilience_score=float(resilience),
            connectivity_matrix=conn_mat,
            vulnerability_zones=vuln_zones,
        )

    def stress_test(self, G: nx.Graph, report: CriticalityReport,
                    k: int = None) -> List[Dict]:
        """Sequential ablation stress test on top critical edges."""
        k = k or self.config.get('ablation_k', 10)
        top_edges = [e for e, _ in report.top_critical_edges[:k]]
        print(f"[Phase III] Running sequential ablation on top-{k} edges...")
        return self.ablation.sequential_ablation(G, top_edges, k)

    def run_disaster(self, G: nx.Graph, event_type: str = 'flood',
                     epicenter=None, radius: float = 80.0) -> DisasterSimulation:
        """Run a named disaster scenario."""
        print(f"[Phase III] Simulating {event_type} disaster...")
        return self.simulator.simulate(G, event_type, epicenter, radius)
