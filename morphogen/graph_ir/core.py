"""
Core GraphIR data structures

Implements the Morphogen Graph IR specification v1.0
Based on: /home/scottsen/src/projects/morphogen/docs/specifications/graph-ir.md
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple
import json
from pathlib import Path


@dataclass
class GraphIROutputPort:
    """Output port definition for a node"""
    name: str
    type: str  # Stream type: Sig, Ctl, Field2D, Field3D, Image, etc.

    def to_dict(self) -> Dict[str, str]:
        return {"name": self.name, "type": self.type}

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "GraphIROutputPort":
        return cls(name=data["name"], type=data["type"])


@dataclass
class GraphIRNode:
    """
    Graph IR node - represents an operator instance.

    Corresponds to the Node Object in the GraphIR specification.

    Attributes:
        id: Unique node identifier
        op: Operator name (from registry)
        params: Parameter values with units (e.g., {"freq": "440Hz"})
        rate: Execution rate (audio | control | visual | sim)
        outputs: List of output port definitions
        profile_overrides: Optional per-node profile settings
    """
    id: str
    op: str
    outputs: List[GraphIROutputPort]
    params: Dict[str, Any] = field(default_factory=dict)
    rate: str = "audio"
    profile_overrides: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary"""
        result = {
            "id": self.id,
            "op": self.op,
            "params": self.params,
            "rate": self.rate,
            "outputs": [port.to_dict() for port in self.outputs],
        }
        if self.profile_overrides:
            result["profile_overrides"] = self.profile_overrides
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphIRNode":
        """Create node from dictionary"""
        outputs = [GraphIROutputPort.from_dict(p) for p in data["outputs"]]
        return cls(
            id=data["id"],
            op=data["op"],
            params=data.get("params", {}),
            rate=data.get("rate", "audio"),
            outputs=outputs,
            profile_overrides=data.get("profile_overrides"),
        )

    def get_output_port(self, port_name: str) -> Optional[GraphIROutputPort]:
        """Get output port by name"""
        for port in self.outputs:
            if port.name == port_name:
                return port
        return None


@dataclass
class GraphIREdge:
    """
    Graph IR edge - represents a connection between nodes.

    Corresponds to the Edge Object in the GraphIR specification.

    Attributes:
        from_port: Source node:port reference (e.g., "osc1:out")
        to_port: Destination node:port reference (e.g., "lpf1:in")
        type: Stream type annotation (Sig, Ctl, etc.)
    """
    from_port: str  # "node_id:port_name"
    to_port: str    # "node_id:port_name"
    type: str       # Stream type (Sig, Ctl, Field2D, etc.)

    def parse_port_ref(self, port_ref: str) -> Tuple[str, str]:
        """Parse port reference into (node_id, port_name)"""
        parts = port_ref.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid port reference: {port_ref}")
        return parts[0], parts[1]

    @property
    def from_node(self) -> str:
        """Get source node ID"""
        return self.parse_port_ref(self.from_port)[0]

    @property
    def from_port_name(self) -> str:
        """Get source port name"""
        return self.parse_port_ref(self.from_port)[1]

    @property
    def to_node(self) -> str:
        """Get destination node ID"""
        return self.parse_port_ref(self.to_port)[0]

    @property
    def to_port_name(self) -> str:
        """Get destination port name"""
        return self.parse_port_ref(self.to_port)[1]

    def to_dict(self) -> Dict[str, str]:
        """Convert to JSON-serializable dictionary"""
        return {
            "from": self.from_port,
            "to": self.to_port,
            "type": self.type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "GraphIREdge":
        """Create edge from dictionary"""
        return cls(
            from_port=data["from"],
            to_port=data["to"],
            type=data["type"],
        )


@dataclass
class GraphIREvent:
    """
    Event stream definition

    Attributes:
        id: Event stream identifier
        type: Event payload type (e.g., "Evt<Note>")
        data: List of timestamped events, sorted by time
    """
    id: str
    type: str
    data: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphIREvent":
        return cls(
            id=data["id"],
            type=data["type"],
            data=data.get("data", []),
        )


@dataclass
class GraphIROutput:
    """Helper class to manage output mappings"""
    name: str
    port_refs: List[str]  # List of "node_id:port_name" references


@dataclass
class GraphIR:
    """
    Complete Graph IR representation.

    Top-level structure for Morphogen Graph IR, following the JSON schema
    defined in the specification.

    Attributes:
        version: GraphIR version (default: "1.0")
        profile: Execution profile (strict | repro | live)
        sample_rate: Audio sample rate in Hz
        nodes: List of graph nodes
        edges: List of graph edges
        outputs: Mapping of output names to node:port references
        seed: Optional random seed for deterministic execution
        events: Optional event stream definitions
        metadata: Optional metadata
    """
    version: str = "1.0"
    profile: str = "repro"
    sample_rate: int = 48000
    nodes: List[GraphIRNode] = field(default_factory=list)
    edges: List[GraphIREdge] = field(default_factory=list)
    outputs: Dict[str, List[str]] = field(default_factory=dict)
    seed: Optional[int] = None
    events: Optional[List[GraphIREvent]] = None
    metadata: Optional[Dict[str, Any]] = None

    def add_node(
        self,
        id: str,
        op: str,
        outputs: List[GraphIROutputPort],
        rate: str = "audio",
        params: Optional[Dict[str, Any]] = None,
        profile_overrides: Optional[Dict[str, Any]] = None,
    ) -> GraphIRNode:
        """Add a node to the graph"""
        node = GraphIRNode(
            id=id,
            op=op,
            outputs=outputs,
            rate=rate,
            params=params or {},
            profile_overrides=profile_overrides,
        )
        self.nodes.append(node)
        return node

    def add_edge(
        self,
        from_port: str,
        to_port: str,
        type: str,
    ) -> GraphIREdge:
        """Add an edge to the graph"""
        edge = GraphIREdge(from_port=from_port, to_port=to_port, type=type)
        self.edges.append(edge)
        return edge

    def add_output(self, name: str, port_refs: List[str]) -> None:
        """Add an output mapping"""
        self.outputs[name] = port_refs

    def get_node(self, node_id: str) -> Optional[GraphIRNode]:
        """Get node by ID"""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary"""
        result = {
            "version": self.version,
            "profile": self.profile,
            "sample_rate": self.sample_rate,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "outputs": self.outputs,
        }

        if self.seed is not None:
            result["seed"] = self.seed

        if self.events:
            result["events"] = [event.to_dict() for event in self.events]

        if self.metadata:
            result["metadata"] = self.metadata

        return result

    def to_json(self, filepath: str, indent: int = 2) -> None:
        """
        Save to .morph.json file

        Args:
            filepath: Path to save the JSON file
            indent: JSON indentation (default: 2)
        """
        path = Path(filepath)

        # Ensure .morph.json extension
        if not filepath.endswith('.morph.json'):
            path = path.with_suffix('.morph.json')

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphIR":
        """Create GraphIR from dictionary"""
        nodes = [GraphIRNode.from_dict(n) for n in data.get("nodes", [])]
        edges = [GraphIREdge.from_dict(e) for e in data.get("edges", [])]

        events = None
        if "events" in data:
            events = [GraphIREvent.from_dict(e) for e in data["events"]]

        return cls(
            version=data.get("version", "1.0"),
            profile=data.get("profile", "repro"),
            sample_rate=data.get("sample_rate", 48000),
            nodes=nodes,
            edges=edges,
            outputs=data.get("outputs", {}),
            seed=data.get("seed"),
            events=events,
            metadata=data.get("metadata"),
        )

    @classmethod
    def from_json(cls, filepath: str) -> "GraphIR":
        """
        Load from .morph.json file

        Args:
            filepath: Path to the JSON file

        Returns:
            GraphIR instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def validate(self) -> List[str]:
        """
        Validate graph structure

        Returns:
            List of validation errors (empty if valid)
        """
        from .validation import GraphValidator
        validator = GraphValidator(self)
        return validator.validate()

    def is_valid(self) -> bool:
        """Check if graph is valid"""
        return len(self.validate()) == 0
