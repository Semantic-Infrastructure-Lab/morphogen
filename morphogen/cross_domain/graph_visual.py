"""
Graph → Visual Domain Interface

Interface for network graph visualization.
"""

from typing import Any, Dict, Type
import numpy as np

from .base import DomainInterface


class GraphToVisualInterface(DomainInterface):
    """
    Graph → Visual: Network graph visualization.

    Use cases:
    - Network structure → visual layout
    - Graph metrics → node colors/sizes
    - Connectivity → edge rendering
    """

    source_domain = "graph"
    target_domain = "visual"

    def __init__(
        self,
        graph_data: Dict[str, Any],
        width: int = 512,
        height: int = 512,
        layout: str = "spring"
    ):
        """
        Args:
            graph_data: Dict with 'nodes' and 'edges' keys
            width: Output image width
            height: Output image height
            layout: Layout algorithm ("spring", "circular", "random")
        """
        super().__init__(source_data=graph_data)
        self.graph_data = graph_data
        self.width = width
        self.height = height
        self.layout = layout

    def transform(self, source_data: Any) -> Dict[str, Any]:
        """
        Convert graph to visual representation.

        Returns:
            Dict with 'node_positions', 'edge_list', 'image' keys
        """
        graph = source_data if source_data is not None else self.graph_data

        n_nodes = len(graph.get('nodes', []))

        # Simple layout algorithms
        if self.layout == "circular":
            # Circular layout
            angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
            radius = min(self.width, self.height) * 0.4
            cx, cy = self.width / 2, self.height / 2

            positions = np.zeros((n_nodes, 2))
            positions[:, 0] = cx + radius * np.cos(angles)
            positions[:, 1] = cy + radius * np.sin(angles)

        elif self.layout == "random":
            # Random layout
            positions = np.random.rand(n_nodes, 2)
            positions[:, 0] *= self.width
            positions[:, 1] *= self.height

        elif self.layout == "spring":
            # Simple spring layout (simplified force-directed)
            positions = np.random.rand(n_nodes, 2)
            positions[:, 0] *= self.width
            positions[:, 1] *= self.height

            # Simple relaxation (could be improved with proper spring algorithm)
            for _ in range(50):
                # Repulsion between all nodes
                for i in range(n_nodes):
                    for j in range(i + 1, n_nodes):
                        delta = positions[i] - positions[j]
                        dist = np.linalg.norm(delta) + 1e-10
                        force = delta / dist * (100.0 / dist)
                        positions[i] += force * 0.1
                        positions[j] -= force * 0.1

                # Clamp to bounds
                positions[:, 0] = np.clip(positions[:, 0], 0, self.width)
                positions[:, 1] = np.clip(positions[:, 1], 0, self.height)

        return {
            'node_positions': positions,
            'edge_list': graph.get('edges', []),
            'n_nodes': n_nodes,
            'width': self.width,
            'height': self.height,
        }

    def validate(self) -> bool:
        """Check graph data is valid."""
        if self.graph_data is None:
            return False

        if 'nodes' not in self.graph_data:
            raise ValueError("Graph data must have 'nodes' key")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {'graph_data': Dict}

    def get_output_interface(self) -> Dict[str, Type]:
        return {'visual_data': Dict[str, Any]}
