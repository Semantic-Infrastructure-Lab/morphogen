"""
Graph and Network Analysis Domain

Provides graph data structures and algorithms for:
- Network analysis
- Shortest path algorithms
- Graph traversal
- Centrality measures
- Community detection
- Flow algorithms

Follows Morphogen's immutability pattern: all operations return new instances.

Version: v0.10.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional, Callable
from enum import Enum
import heapq
from collections import deque, defaultdict

from morphogen.core.operator import operator, OpCategory


class GraphType(Enum):
    """Graph type enumeration"""
    DIRECTED = "directed"
    UNDIRECTED = "undirected"


@dataclass
class Graph:
    """Graph data structure

    Represents a graph with nodes and edges. Supports both directed
    and undirected graphs with weighted or unweighted edges.

    Attributes:
        num_nodes: Number of nodes in the graph
        adjacency_list: Dict mapping node_id -> list of (neighbor_id, weight) tuples
        graph_type: DIRECTED or UNDIRECTED
        node_data: Optional dict mapping node_id -> arbitrary node data
    """
    num_nodes: int
    adjacency_list: Dict[int, List[Tuple[int, float]]]
    graph_type: GraphType = GraphType.UNDIRECTED
    node_data: Dict[int, any] = field(default_factory=dict)

    def copy(self) -> 'Graph':
        """Create a deep copy of the graph"""
        return Graph(
            num_nodes=self.num_nodes,
            adjacency_list={k: list(v) for k, v in self.adjacency_list.items()},
            graph_type=self.graph_type,
            node_data=dict(self.node_data)
        )


@dataclass
class GraphMetrics:
    """Results from graph analysis

    Attributes:
        node_metrics: Dict mapping node_id -> metric value
        edge_metrics: Dict mapping (node1, node2) -> metric value
        global_metrics: Dict of graph-level metrics
    """
    node_metrics: Dict[int, float] = field(default_factory=dict)
    edge_metrics: Dict[Tuple[int, int], float] = field(default_factory=dict)
    global_metrics: Dict[str, float] = field(default_factory=dict)

    def copy(self) -> 'GraphMetrics':
        """Create a deep copy"""
        return GraphMetrics(
            node_metrics=dict(self.node_metrics),
            edge_metrics=dict(self.edge_metrics),
            global_metrics=dict(self.global_metrics)
        )


class GraphOperations:
    """Graph and network analysis operations"""

    @staticmethod
    @operator(
        domain="graph",
        category=OpCategory.CONSTRUCT,
        signature="(num_nodes: int, graph_type: GraphType) -> Graph",
        deterministic=True,
        doc="Create an empty graph with specified number of nodes"
    )
    def create_empty(num_nodes: int, graph_type: GraphType = GraphType.UNDIRECTED) -> Graph:
        """Create an empty graph with specified number of nodes

        Args:
            num_nodes: Number of nodes
            graph_type: DIRECTED or UNDIRECTED

        Returns:
            Empty graph
        """
        return Graph(
            num_nodes=num_nodes,
            adjacency_list={i: [] for i in range(num_nodes)},
            graph_type=graph_type
        )

    @staticmethod
    @operator(
        domain="graph",
        category=OpCategory.TRANSFORM,
        signature="(graph: Graph, node1: int, node2: int, weight: float) -> Graph",
        deterministic=True,
        doc="Add an edge to the graph"
    )
    def add_edge(graph: Graph, node1: int, node2: int, weight: float = 1.0) -> Graph:
        """Add an edge to the graph

        Args:
            graph: Input graph
            node1: First node
            node2: Second node
            weight: Edge weight (default 1.0)

        Returns:
            New graph with edge added
        """
        result = graph.copy()

        # Add edge from node1 to node2
        result.adjacency_list[node1].append((node2, weight))

        # If undirected, add reverse edge
        if graph.graph_type == GraphType.UNDIRECTED:
            result.adjacency_list[node2].append((node1, weight))

        return result

    @staticmethod
    @operator(
        domain="graph",
        category=OpCategory.TRANSFORM,
        signature="(graph: Graph, node1: int, node2: int) -> Graph",
        deterministic=True,
        doc="Remove an edge from the graph"
    )
    def remove_edge(graph: Graph, node1: int, node2: int) -> Graph:
        """Remove an edge from the graph

        Args:
            graph: Input graph
            node1: First node
            node2: Second node

        Returns:
            New graph with edge removed
        """
        result = graph.copy()

        # Remove edge from node1 to node2
        result.adjacency_list[node1] = [
            (n, w) for n, w in result.adjacency_list[node1] if n != node2
        ]

        # If undirected, remove reverse edge
        if graph.graph_type == GraphType.UNDIRECTED:
            result.adjacency_list[node2] = [
                (n, w) for n, w in result.adjacency_list[node2] if n != node1
            ]

        return result

    @staticmethod
    @operator(
        domain="graph",
        category=OpCategory.CONSTRUCT,
        signature="(adj_matrix: np.ndarray, graph_type: GraphType, threshold: float) -> Graph",
        deterministic=True,
        doc="Create graph from adjacency matrix"
    )
    def from_adjacency_matrix(adj_matrix: np.ndarray,
                            graph_type: GraphType = GraphType.UNDIRECTED,
                            threshold: float = 0.0) -> Graph:
        """Create graph from adjacency matrix

        Args:
            adj_matrix: NxN adjacency matrix
            graph_type: DIRECTED or UNDIRECTED
            threshold: Minimum weight to create edge (default 0.0)

        Returns:
            Graph created from matrix
        """
        num_nodes = adj_matrix.shape[0]
        adjacency_list = {i: [] for i in range(num_nodes)}

        for i in range(num_nodes):
            for j in range(num_nodes):
                weight = adj_matrix[i, j]
                if weight > threshold:
                    adjacency_list[i].append((j, float(weight)))

        return Graph(
            num_nodes=num_nodes,
            adjacency_list=adjacency_list,
            graph_type=graph_type
        )

    @staticmethod
    @operator(
        domain="graph",
        category=OpCategory.TRANSFORM,
        signature="(graph: Graph) -> np.ndarray",
        deterministic=True,
        doc="Convert graph to adjacency matrix"
    )
    def to_adjacency_matrix(graph: Graph) -> np.ndarray:
        """Convert graph to adjacency matrix

        Args:
            graph: Input graph

        Returns:
            NxN adjacency matrix
        """
        matrix = np.zeros((graph.num_nodes, graph.num_nodes))

        for node, neighbors in graph.adjacency_list.items():
            for neighbor, weight in neighbors:
                matrix[node, neighbor] = weight

        return matrix

    @staticmethod
    @operator(
        domain="graph",
        category=OpCategory.QUERY,
        signature="(graph: Graph, start: int) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]",
        deterministic=True,
        doc="Dijkstra's shortest path algorithm"
    )
    def dijkstra(graph: Graph, start: int) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
        """Dijkstra's shortest path algorithm

        Args:
            graph: Input graph
            start: Starting node

        Returns:
            Tuple of (distances dict, predecessors dict)
        """
        distances = {i: float('inf') for i in range(graph.num_nodes)}
        distances[start] = 0.0
        predecessors = {i: None for i in range(graph.num_nodes)}

        # Priority queue: (distance, node)
        pq = [(0.0, start)]
        visited = set()

        while pq:
            current_dist, current = heapq.heappop(pq)

            if current in visited:
                continue
            visited.add(current)

            for neighbor, weight in graph.adjacency_list[current]:
                distance = current_dist + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))

        return distances, predecessors

    @staticmethod
    @operator(
        domain="graph",
        category=OpCategory.QUERY,
        signature="(graph: Graph, start: int, end: int) -> Tuple[List[int], float]",
        deterministic=True,
        doc="Find shortest path between two nodes"
    )
    def shortest_path(graph: Graph, start: int, end: int) -> Tuple[List[int], float]:
        """Find shortest path between two nodes

        Args:
            graph: Input graph
            start: Starting node
            end: Ending node

        Returns:
            Tuple of (path as list of nodes, total distance)
        """
        distances, predecessors = GraphOperations.dijkstra(graph, start)

        # Reconstruct path
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = predecessors[current]

        path.reverse()

        # Check if path exists
        if path[0] != start:
            return [], float('inf')

        return path, distances[end]

    @staticmethod
    @operator(
        domain="graph",
        category=OpCategory.QUERY,
        signature="(graph: Graph, start: int) -> Dict[int, int]",
        deterministic=True,
        doc="Breadth-first search traversal"
    )
    def bfs(graph: Graph, start: int) -> Dict[int, int]:
        """Breadth-first search

        Args:
            graph: Input graph
            start: Starting node

        Returns:
            Dict mapping node -> distance from start
        """
        distances = {i: -1 for i in range(graph.num_nodes)}
        distances[start] = 0

        queue = deque([start])

        while queue:
            current = queue.popleft()

            for neighbor, _ in graph.adjacency_list[current]:
                if distances[neighbor] == -1:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)

        return distances

    @staticmethod
    @operator(
        domain="graph",
        category=OpCategory.QUERY,
        signature="(graph: Graph, start: int, visited: Optional[Set[int]]) -> List[int]",
        deterministic=True,
        doc="Depth-first search traversal"
    )
    def dfs(graph: Graph, start: int, visited: Optional[Set[int]] = None) -> List[int]:
        """Depth-first search

        Args:
            graph: Input graph
            start: Starting node
            visited: Set of already visited nodes (for recursion)

        Returns:
            List of nodes in DFS order
        """
        if visited is None:
            visited = set()

        result = []

        if start not in visited:
            visited.add(start)
            result.append(start)

            for neighbor, _ in graph.adjacency_list[start]:
                if neighbor not in visited:
                    result.extend(GraphOperations.dfs(graph, neighbor, visited))

        return result

    @staticmethod
    @operator(
        domain="graph",
        category=OpCategory.QUERY,
        signature="(graph: Graph) -> List[List[int]]",
        deterministic=True,
        doc="Find connected components in undirected graph"
    )
    def connected_components(graph: Graph) -> List[List[int]]:
        """Find connected components in undirected graph

        Args:
            graph: Input graph (should be undirected)

        Returns:
            List of components, each component is a list of node ids
        """
        visited = set()
        components = []

        for node in range(graph.num_nodes):
            if node not in visited:
                component = GraphOperations.dfs(graph, node, visited)
                components.append(component)

        return components

    @staticmethod
    @operator(
        domain="graph",
        category=OpCategory.QUERY,
        signature="(graph: Graph) -> GraphMetrics",
        deterministic=True,
        doc="Calculate degree centrality for all nodes"
    )
    def degree_centrality(graph: Graph) -> GraphMetrics:
        """Calculate degree centrality for all nodes

        Args:
            graph: Input graph

        Returns:
            GraphMetrics with node_metrics containing centrality scores
        """
        node_metrics = {}

        for node in range(graph.num_nodes):
            degree = len(graph.adjacency_list[node])
            # Normalize by (n-1)
            centrality = degree / (graph.num_nodes - 1) if graph.num_nodes > 1 else 0.0
            node_metrics[node] = centrality

        return GraphMetrics(node_metrics=node_metrics)

    @staticmethod
    @operator(
        domain="graph",
        category=OpCategory.QUERY,
        signature="(graph: Graph) -> GraphMetrics",
        deterministic=True,
        doc="Calculate betweenness centrality (simplified)"
    )
    def betweenness_centrality(graph: Graph) -> GraphMetrics:
        """Calculate betweenness centrality (simplified)

        Args:
            graph: Input graph

        Returns:
            GraphMetrics with betweenness scores
        """
        betweenness = {i: 0.0 for i in range(graph.num_nodes)}

        # For each pair of nodes, find shortest paths
        for s in range(graph.num_nodes):
            distances, predecessors = GraphOperations.dijkstra(graph, s)

            for t in range(graph.num_nodes):
                if s != t:
                    # Count paths through each node
                    current = t
                    while predecessors[current] is not None:
                        if current != s and current != t:
                            betweenness[current] += 1.0
                        current = predecessors[current]

        # Normalize
        n = graph.num_nodes
        if n > 2:
            norm = (n - 1) * (n - 2)
            for node in betweenness:
                betweenness[node] /= norm

        return GraphMetrics(node_metrics=betweenness)

    @staticmethod
    @operator(
        domain="graph",
        category=OpCategory.QUERY,
        signature="(graph: Graph, damping: float, max_iter: int, tol: float) -> GraphMetrics",
        deterministic=True,
        doc="Calculate PageRank scores"
    )
    def pagerank(graph: Graph, damping: float = 0.85, max_iter: int = 100,
                tol: float = 1e-6) -> GraphMetrics:
        """Calculate PageRank scores

        Args:
            graph: Input graph
            damping: Damping factor (default 0.85)
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            GraphMetrics with PageRank scores
        """
        n = graph.num_nodes
        ranks = np.ones(n) / n

        # Build transition matrix
        out_degree = np.zeros(n)
        for node, neighbors in graph.adjacency_list.items():
            out_degree[node] = len(neighbors) if len(neighbors) > 0 else 1

        for iteration in range(max_iter):
            new_ranks = np.ones(n) * (1 - damping) / n

            for node in range(n):
                for neighbor, _ in graph.adjacency_list[node]:
                    new_ranks[neighbor] += damping * ranks[node] / out_degree[node]

            # Check convergence
            if np.linalg.norm(new_ranks - ranks, 1) < tol:
                break

            ranks = new_ranks

        node_metrics = {i: ranks[i] for i in range(n)}
        return GraphMetrics(node_metrics=node_metrics)

    @staticmethod
    @operator(
        domain="graph",
        category=OpCategory.QUERY,
        signature="(graph: Graph) -> GraphMetrics",
        deterministic=True,
        doc="Calculate local clustering coefficient for each node"
    )
    def clustering_coefficient(graph: Graph) -> GraphMetrics:
        """Calculate local clustering coefficient for each node

        Args:
            graph: Input graph (should be undirected)

        Returns:
            GraphMetrics with clustering coefficients
        """
        node_metrics = {}

        for node in range(graph.num_nodes):
            neighbors = [n for n, _ in graph.adjacency_list[node]]
            k = len(neighbors)

            if k < 2:
                node_metrics[node] = 0.0
                continue

            # Count triangles
            triangles = 0
            for i, n1 in enumerate(neighbors):
                for n2 in neighbors[i+1:]:
                    # Check if n1 and n2 are connected
                    n1_neighbors = [n for n, _ in graph.adjacency_list[n1]]
                    if n2 in n1_neighbors:
                        triangles += 1

            # Clustering coefficient
            possible_triangles = k * (k - 1) / 2
            node_metrics[node] = triangles / possible_triangles if possible_triangles > 0 else 0.0

        return GraphMetrics(node_metrics=node_metrics)

    @staticmethod
    @operator(
        domain="graph",
        category=OpCategory.TRANSFORM,
        signature="(graph: Graph) -> Graph",
        deterministic=True,
        doc="Find minimum spanning tree using Prim's algorithm"
    )
    def minimum_spanning_tree(graph: Graph) -> Graph:
        """Find minimum spanning tree using Prim's algorithm

        Args:
            graph: Input graph (should be undirected)

        Returns:
            New graph containing only MST edges
        """
        if graph.num_nodes == 0:
            return graph.copy()

        mst = GraphOperations.create_empty(graph.num_nodes, GraphType.UNDIRECTED)
        visited = set([0])
        edges = []

        # Add all edges from node 0
        for neighbor, weight in graph.adjacency_list[0]:
            heapq.heappush(edges, (weight, 0, neighbor))

        while edges and len(visited) < graph.num_nodes:
            weight, u, v = heapq.heappop(edges)

            if v in visited:
                continue

            # Add edge to MST
            mst = GraphOperations.add_edge(mst, u, v, weight)
            visited.add(v)

            # Add new edges
            for neighbor, w in graph.adjacency_list[v]:
                if neighbor not in visited:
                    heapq.heappush(edges, (w, v, neighbor))

        return mst

    @staticmethod
    @operator(
        domain="graph",
        category=OpCategory.QUERY,
        signature="(graph: Graph) -> Optional[List[int]]",
        deterministic=True,
        doc="Topological sort for directed acyclic graph"
    )
    def topological_sort(graph: Graph) -> Optional[List[int]]:
        """Topological sort for directed acyclic graph

        Args:
            graph: Input directed graph

        Returns:
            Topologically sorted list of nodes, or None if graph has cycles
        """
        if graph.graph_type != GraphType.DIRECTED:
            return None

        # Calculate in-degrees
        in_degree = {i: 0 for i in range(graph.num_nodes)}
        for node, neighbors in graph.adjacency_list.items():
            for neighbor, _ in neighbors:
                in_degree[neighbor] += 1

        # Queue of nodes with in-degree 0
        queue = deque([node for node in range(graph.num_nodes) if in_degree[node] == 0])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)

            for neighbor, _ in graph.adjacency_list[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check if all nodes were processed (no cycles)
        if len(result) != graph.num_nodes:
            return None

        return result

    @staticmethod
    @operator(
        domain="graph",
        category=OpCategory.QUERY,
        signature="(graph: Graph, source: int, sink: int) -> float",
        deterministic=True,
        doc="Calculate maximum flow using Ford-Fulkerson with BFS (Edmonds-Karp)"
    )
    def max_flow(graph: Graph, source: int, sink: int) -> float:
        """Calculate maximum flow using Ford-Fulkerson with BFS (Edmonds-Karp)

        Args:
            graph: Input directed graph
            source: Source node
            sink: Sink node

        Returns:
            Maximum flow value
        """
        # Create residual graph
        residual = graph.copy()
        max_flow_value = 0.0

        def bfs_find_path():
            """Find augmenting path using BFS"""
            visited = set([source])
            queue = deque([(source, [source])])

            while queue:
                node, path = queue.popleft()

                if node == sink:
                    return path

                for neighbor, capacity in residual.adjacency_list[node]:
                    if neighbor not in visited and capacity > 0:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))

            return None

        # Find augmenting paths
        while True:
            path = bfs_find_path()
            if path is None:
                break

            # Find minimum capacity along path
            min_capacity = float('inf')
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                for neighbor, capacity in residual.adjacency_list[u]:
                    if neighbor == v:
                        min_capacity = min(min_capacity, capacity)
                        break

            # Update residual capacities
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]

                # Decrease forward edge
                new_neighbors = []
                for neighbor, capacity in residual.adjacency_list[u]:
                    if neighbor == v:
                        new_cap = capacity - min_capacity
                        if new_cap > 0:
                            new_neighbors.append((neighbor, new_cap))
                    else:
                        new_neighbors.append((neighbor, capacity))
                residual.adjacency_list[u] = new_neighbors

                # Increase backward edge
                found = False
                new_neighbors = []
                for neighbor, capacity in residual.adjacency_list[v]:
                    if neighbor == u:
                        new_neighbors.append((neighbor, capacity + min_capacity))
                        found = True
                    else:
                        new_neighbors.append((neighbor, capacity))

                if not found:
                    new_neighbors.append((u, min_capacity))

                residual.adjacency_list[v] = new_neighbors

            max_flow_value += min_capacity

        return max_flow_value

    @staticmethod
    @operator(
        domain="graph",
        category=OpCategory.CONSTRUCT,
        signature="(num_nodes: int, edge_probability: float, graph_type: GraphType, seed: Optional[int]) -> Graph",
        deterministic=True,
        doc="Generate random graph (Erdős–Rényi model)"
    )
    def random_graph(num_nodes: int, edge_probability: float,
                    graph_type: GraphType = GraphType.UNDIRECTED,
                    seed: Optional[int] = None) -> Graph:
        """Generate random graph (Erdős–Rényi model)

        Args:
            num_nodes: Number of nodes
            edge_probability: Probability of edge between any two nodes
            graph_type: DIRECTED or UNDIRECTED
            seed: Random seed for reproducibility

        Returns:
            Random graph
        """
        if seed is not None:
            np.random.seed(seed)

        graph = GraphOperations.create_empty(num_nodes, graph_type)

        for i in range(num_nodes):
            for j in range(i + 1 if graph_type == GraphType.UNDIRECTED else 0, num_nodes):
                if i != j and np.random.random() < edge_probability:
                    weight = np.random.uniform(0.1, 1.0)
                    graph = GraphOperations.add_edge(graph, i, j, weight)

        return graph

    @staticmethod
    @operator(
        domain="graph",
        category=OpCategory.CONSTRUCT,
        signature="(rows: int, cols: int, diagonal: bool) -> Graph",
        deterministic=True,
        doc="Create a grid graph"
    )
    def grid_graph(rows: int, cols: int, diagonal: bool = False) -> Graph:
        """Create a grid graph

        Args:
            rows: Number of rows
            cols: Number of columns
            diagonal: Include diagonal connections

        Returns:
            Grid graph
        """
        num_nodes = rows * cols
        graph = GraphOperations.create_empty(num_nodes, GraphType.UNDIRECTED)

        def node_id(r, c):
            return r * cols + c

        for r in range(rows):
            for c in range(cols):
                current = node_id(r, c)

                # Right
                if c < cols - 1:
                    graph = GraphOperations.add_edge(graph, current, node_id(r, c + 1))

                # Down
                if r < rows - 1:
                    graph = GraphOperations.add_edge(graph, current, node_id(r + 1, c))

                if diagonal:
                    # Down-right
                    if r < rows - 1 and c < cols - 1:
                        graph = GraphOperations.add_edge(graph, current, node_id(r + 1, c + 1), 1.414)

                    # Down-left
                    if r < rows - 1 and c > 0:
                        graph = GraphOperations.add_edge(graph, current, node_id(r + 1, c - 1), 1.414)

        return graph


# Export singleton instance for DSL access
graph = GraphOperations()

# Export operators for domain registry discovery
create_empty = GraphOperations.create_empty
add_edge = GraphOperations.add_edge
remove_edge = GraphOperations.remove_edge
from_adjacency_matrix = GraphOperations.from_adjacency_matrix
to_adjacency_matrix = GraphOperations.to_adjacency_matrix
dijkstra = GraphOperations.dijkstra
shortest_path = GraphOperations.shortest_path
bfs = GraphOperations.bfs
dfs = GraphOperations.dfs
connected_components = GraphOperations.connected_components
degree_centrality = GraphOperations.degree_centrality
betweenness_centrality = GraphOperations.betweenness_centrality
pagerank = GraphOperations.pagerank
clustering_coefficient = GraphOperations.clustering_coefficient
minimum_spanning_tree = GraphOperations.minimum_spanning_tree
topological_sort = GraphOperations.topological_sort
max_flow = GraphOperations.max_flow
random_graph = GraphOperations.random_graph
grid_graph = GraphOperations.grid_graph
