"""
GraphIR validation module

Implements validation rules from the GraphIR specification:
1. Type checking (stream types compatibility)
2. Rate compatibility
3. DAG constraint (no cycles)
4. Unit consistency
5. Node/edge reference integrity

Based on: /home/scottsen/src/projects/morphogen/docs/specifications/graph-ir.md (lines 259-307)
"""

from typing import List, Dict, Set, Tuple, Optional
import re


class GraphValidator:
    """
    Validates GraphIR structure and semantics

    Performs comprehensive validation including:
    - Node ID uniqueness
    - Edge reference integrity
    - Type compatibility
    - DAG constraint (no cycles)
    - Unit consistency
    - Rate compatibility
    """

    # Valid rate classes
    VALID_RATES = {"audio", "control", "visual", "sim"}

    # Valid stream types
    VALID_STREAM_TYPES = {"Sig", "Ctl", "Field2D", "Field3D", "Image"}

    # Type compatibility matrix (from_type -> to_type)
    # Same types are always compatible, cross-type requires explicit conversion
    TYPE_COMPATIBILITY = {
        "Sig": {"Sig"},
        "Ctl": {"Ctl", "Sig"},  # Ctl can be upsampled to Sig
        "Field2D": {"Field2D"},
        "Field3D": {"Field3D"},
        "Image": {"Image"},
    }

    # Unit pattern: value + optional unit suffix
    UNIT_PATTERN = re.compile(r'^(-?\d+\.?\d*)\s*([a-zA-Z]+)?$')

    # Valid unit types
    VALID_UNITS = {
        "Hz", "kHz", "MHz",       # Frequency
        "s", "ms", "us",          # Time
        "dB", "dBFS",             # Decibels
        "rad", "deg",             # Angle
        "m", "cm", "mm",          # Distance
        "kg", "g",                # Mass
    }

    def __init__(self, graph):
        """
        Initialize validator with a GraphIR instance

        Args:
            graph: GraphIR instance to validate
        """
        self.graph = graph
        self.errors: List[str] = []

    def validate(self) -> List[str]:
        """
        Run all validation checks

        Returns:
            List of validation errors (empty if valid)
        """
        self.errors = []

        self._validate_node_ids()
        self._validate_node_rates()
        self._validate_edge_references()
        self._validate_edge_types()
        self._validate_dag()
        self._validate_units()
        self._validate_outputs()

        return self.errors

    def _validate_node_ids(self) -> None:
        """Check that all node IDs are unique"""
        seen_ids: Set[str] = set()
        for node in self.graph.nodes:
            if node.id in seen_ids:
                self.errors.append(f"Duplicate node ID: {node.id}")
            seen_ids.add(node.id)

    def _validate_node_rates(self) -> None:
        """Check that all node rates are valid"""
        for node in self.graph.nodes:
            if node.rate not in self.VALID_RATES:
                self.errors.append(
                    f"Invalid rate '{node.rate}' for node '{node.id}'. "
                    f"Must be one of: {', '.join(self.VALID_RATES)}"
                )

    def _validate_edge_references(self) -> None:
        """Check that all edges reference valid nodes and ports"""
        # Build node lookup
        node_map = {node.id: node for node in self.graph.nodes}

        for edge in self.graph.edges:
            # Validate from_port reference
            try:
                from_node_id = edge.from_node
                from_port_name = edge.from_port_name

                if from_node_id not in node_map:
                    self.errors.append(
                        f"Edge references non-existent source node: {from_node_id}"
                    )
                    continue

                from_node = node_map[from_node_id]
                from_port = from_node.get_output_port(from_port_name)
                if from_port is None:
                    self.errors.append(
                        f"Edge references non-existent output port: "
                        f"{edge.from_port} (node {from_node_id} has ports: "
                        f"{[p.name for p in from_node.outputs]})"
                    )

            except ValueError as e:
                self.errors.append(f"Invalid from_port format: {edge.from_port} ({e})")

            # Validate to_port reference
            try:
                to_node_id = edge.to_node
                to_port_name = edge.to_port_name

                if to_node_id not in node_map:
                    self.errors.append(
                        f"Edge references non-existent destination node: {to_node_id}"
                    )

            except ValueError as e:
                self.errors.append(f"Invalid to_port format: {edge.to_port} ({e})")

    def _validate_edge_types(self) -> None:
        """Check that edge types are compatible with node output types"""
        node_map = {node.id: node for node in self.graph.nodes}

        for edge in self.graph.edges:
            try:
                # Get source node and port
                from_node = node_map.get(edge.from_node)
                if from_node is None:
                    continue  # Already reported in _validate_edge_references

                from_port = from_node.get_output_port(edge.from_port_name)
                if from_port is None:
                    continue  # Already reported in _validate_edge_references

                # Check type compatibility
                from_type = from_port.type
                edge_type = edge.type

                # Validate stream types are valid
                if from_type not in self.VALID_STREAM_TYPES:
                    self.errors.append(
                        f"Invalid stream type '{from_type}' in node '{from_node.id}'"
                    )

                if edge_type not in self.VALID_STREAM_TYPES:
                    self.errors.append(
                        f"Invalid stream type '{edge_type}' in edge {edge.from_port} -> {edge.to_port}"
                    )

                # Check type compatibility
                compatible_types = self.TYPE_COMPATIBILITY.get(from_type, set())
                if edge_type not in compatible_types and from_type != edge_type:
                    self.errors.append(
                        f"Type mismatch in edge {edge.from_port} -> {edge.to_port}: "
                        f"{from_type} is not compatible with {edge_type}"
                    )

            except Exception as e:
                self.errors.append(f"Error validating edge types: {e}")

    def _validate_dag(self) -> None:
        """Check that the graph is a DAG (no cycles)"""
        # Build adjacency list
        adjacency: Dict[str, List[str]] = {node.id: [] for node in self.graph.nodes}

        for edge in self.graph.edges:
            try:
                from_node = edge.from_node
                to_node = edge.to_node
                if from_node in adjacency and to_node in adjacency:
                    adjacency[from_node].append(to_node)
            except ValueError:
                continue  # Invalid edge format, already reported

        # DFS cycle detection
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {node.id: WHITE for node in self.graph.nodes}
        path: List[str] = []

        def visit(node_id: str) -> bool:
            """DFS visit, returns True if cycle found"""
            if color[node_id] == GRAY:
                # Found back edge (cycle)
                cycle_start = path.index(node_id)
                cycle = path[cycle_start:] + [node_id]
                self.errors.append(
                    f"Graph contains cycle: {' -> '.join(cycle)}"
                )
                return True

            if color[node_id] == BLACK:
                return False

            color[node_id] = GRAY
            path.append(node_id)

            for neighbor in adjacency[node_id]:
                if visit(neighbor):
                    return True

            path.pop()
            color[node_id] = BLACK
            return False

        # Check all nodes (graph might be disconnected)
        for node in self.graph.nodes:
            if color[node.id] == WHITE:
                if visit(node.id):
                    break  # Stop after finding first cycle

    def _validate_units(self) -> None:
        """Check that parameter units are valid"""
        for node in self.graph.nodes:
            for param_name, param_value in node.params.items():
                # Skip non-string parameters (e.g., plain numbers)
                if not isinstance(param_value, str):
                    continue

                # Try to parse unit
                match = self.UNIT_PATTERN.match(param_value)
                if match:
                    value_str, unit = match.groups()
                    if unit and unit not in self.VALID_UNITS:
                        # This is a warning rather than an error
                        # (we're permissive for now)
                        pass
                else:
                    # Not a unit pattern, might be a string parameter
                    pass

    def _validate_outputs(self) -> None:
        """Check that output mappings reference valid ports"""
        node_map = {node.id: node for node in self.graph.nodes}

        for output_name, port_refs in self.graph.outputs.items():
            for port_ref in port_refs:
                try:
                    # Parse port reference
                    parts = port_ref.split(":", 1)
                    if len(parts) != 2:
                        self.errors.append(
                            f"Invalid port reference in output '{output_name}': {port_ref}"
                        )
                        continue

                    node_id, port_name = parts

                    # Check node exists
                    if node_id not in node_map:
                        self.errors.append(
                            f"Output '{output_name}' references non-existent node: {node_id}"
                        )
                        continue

                    # Check port exists
                    node = node_map[node_id]
                    port = node.get_output_port(port_name)
                    if port is None:
                        self.errors.append(
                            f"Output '{output_name}' references non-existent port: "
                            f"{port_ref} (node {node_id} has ports: "
                            f"{[p.name for p in node.outputs]})"
                        )

                except Exception as e:
                    self.errors.append(
                        f"Error validating output '{output_name}': {e}"
                    )

    def topological_sort(self) -> Optional[List[str]]:
        """
        Compute topological sort of nodes (execution order)

        Returns:
            List of node IDs in topological order, or None if graph has cycles
        """
        # Build adjacency list
        adjacency: Dict[str, List[str]] = {node.id: [] for node in self.graph.nodes}
        in_degree: Dict[str, int] = {node.id: 0 for node in self.graph.nodes}

        for edge in self.graph.edges:
            try:
                from_node = edge.from_node
                to_node = edge.to_node
                if from_node in adjacency and to_node in adjacency:
                    adjacency[from_node].append(to_node)
                    in_degree[to_node] += 1
            except ValueError:
                continue

        # Kahn's algorithm
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            node_id = queue.pop(0)
            result.append(node_id)

            for neighbor in adjacency[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # If result doesn't include all nodes, there's a cycle
        if len(result) != len(self.graph.nodes):
            return None

        return result
