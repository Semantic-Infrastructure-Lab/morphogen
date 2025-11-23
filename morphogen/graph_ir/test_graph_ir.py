"""
Unit tests for GraphIR core functionality

Tests cover:
1. Data structure creation and serialization
2. JSON load/save
3. Validation (types, DAG, units, references)
4. Complex graph construction
"""

import pytest
import tempfile
import json
from pathlib import Path

from .core import (
    GraphIR,
    GraphIRNode,
    GraphIREdge,
    GraphIROutputPort,
    GraphIREvent,
)
from .validation import GraphValidator


class TestGraphIRNode:
    """Test GraphIRNode class"""

    def test_create_node(self):
        """Test basic node creation"""
        outputs = [GraphIROutputPort(name="out", type="Sig")]
        node = GraphIRNode(
            id="osc1",
            op="sine",
            outputs=outputs,
            rate="audio",
            params={"freq": "440Hz"},
        )

        assert node.id == "osc1"
        assert node.op == "sine"
        assert node.rate == "audio"
        assert node.params["freq"] == "440Hz"
        assert len(node.outputs) == 1
        assert node.outputs[0].name == "out"

    def test_node_to_dict(self):
        """Test node serialization"""
        outputs = [GraphIROutputPort(name="out", type="Sig")]
        node = GraphIRNode(
            id="osc1",
            op="sine",
            outputs=outputs,
            params={"freq": "440Hz"},
        )

        data = node.to_dict()
        assert data["id"] == "osc1"
        assert data["op"] == "sine"
        assert data["params"]["freq"] == "440Hz"
        assert data["outputs"][0]["name"] == "out"

    def test_node_from_dict(self):
        """Test node deserialization"""
        data = {
            "id": "osc1",
            "op": "sine",
            "params": {"freq": "440Hz"},
            "rate": "audio",
            "outputs": [{"name": "out", "type": "Sig"}],
        }

        node = GraphIRNode.from_dict(data)
        assert node.id == "osc1"
        assert node.op == "sine"
        assert node.params["freq"] == "440Hz"
        assert len(node.outputs) == 1

    def test_get_output_port(self):
        """Test output port lookup"""
        outputs = [
            GraphIROutputPort(name="left", type="Sig"),
            GraphIROutputPort(name="right", type="Sig"),
        ]
        node = GraphIRNode(id="pan1", op="pan", outputs=outputs)

        left_port = node.get_output_port("left")
        assert left_port is not None
        assert left_port.name == "left"

        missing_port = node.get_output_port("missing")
        assert missing_port is None


class TestGraphIREdge:
    """Test GraphIREdge class"""

    def test_create_edge(self):
        """Test basic edge creation"""
        edge = GraphIREdge(
            from_port="osc1:out",
            to_port="lpf1:in",
            type="Sig",
        )

        assert edge.from_port == "osc1:out"
        assert edge.to_port == "lpf1:in"
        assert edge.type == "Sig"

    def test_parse_port_ref(self):
        """Test port reference parsing"""
        edge = GraphIREdge(from_port="osc1:out", to_port="lpf1:in", type="Sig")

        assert edge.from_node == "osc1"
        assert edge.from_port_name == "out"
        assert edge.to_node == "lpf1"
        assert edge.to_port_name == "in"

    def test_edge_to_dict(self):
        """Test edge serialization"""
        edge = GraphIREdge(from_port="osc1:out", to_port="lpf1:in", type="Sig")
        data = edge.to_dict()

        assert data["from"] == "osc1:out"
        assert data["to"] == "lpf1:in"
        assert data["type"] == "Sig"

    def test_edge_from_dict(self):
        """Test edge deserialization"""
        data = {
            "from": "osc1:out",
            "to": "lpf1:in",
            "type": "Sig",
        }

        edge = GraphIREdge.from_dict(data)
        assert edge.from_port == "osc1:out"
        assert edge.to_port == "lpf1:in"


class TestGraphIR:
    """Test GraphIR class"""

    def test_create_empty_graph(self):
        """Test empty graph creation"""
        graph = GraphIR()

        assert graph.version == "1.0"
        assert graph.profile == "repro"
        assert graph.sample_rate == 48000
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    def test_add_node(self):
        """Test adding nodes to graph"""
        graph = GraphIR()
        outputs = [GraphIROutputPort(name="out", type="Sig")]

        node = graph.add_node(
            id="osc1",
            op="sine",
            outputs=outputs,
            params={"freq": "440Hz"},
        )

        assert len(graph.nodes) == 1
        assert graph.nodes[0].id == "osc1"
        assert node.id == "osc1"

    def test_add_edge(self):
        """Test adding edges to graph"""
        graph = GraphIR()

        edge = graph.add_edge(
            from_port="osc1:out",
            to_port="lpf1:in",
            type="Sig",
        )

        assert len(graph.edges) == 1
        assert graph.edges[0].from_port == "osc1:out"

    def test_get_node(self):
        """Test node lookup"""
        graph = GraphIR()
        outputs = [GraphIROutputPort(name="out", type="Sig")]
        graph.add_node(id="osc1", op="sine", outputs=outputs)

        node = graph.get_node("osc1")
        assert node is not None
        assert node.id == "osc1"

        missing = graph.get_node("missing")
        assert missing is None

    def test_graph_to_dict(self):
        """Test graph serialization"""
        graph = GraphIR(sample_rate=44100, seed=42)
        outputs = [GraphIROutputPort(name="out", type="Sig")]
        graph.add_node(id="osc1", op="sine", outputs=outputs)
        graph.add_edge(from_port="osc1:out", to_port="lpf1:in", type="Sig")

        data = graph.to_dict()

        assert data["version"] == "1.0"
        assert data["sample_rate"] == 44100
        assert data["seed"] == 42
        assert len(data["nodes"]) == 1
        assert len(data["edges"]) == 1

    def test_graph_from_dict(self):
        """Test graph deserialization"""
        data = {
            "version": "1.0",
            "profile": "repro",
            "sample_rate": 48000,
            "nodes": [
                {
                    "id": "osc1",
                    "op": "sine",
                    "params": {"freq": "440Hz"},
                    "rate": "audio",
                    "outputs": [{"name": "out", "type": "Sig"}],
                }
            ],
            "edges": [
                {"from": "osc1:out", "to": "lpf1:in", "type": "Sig"}
            ],
            "outputs": {"mono": ["lpf1:out"]},
        }

        graph = GraphIR.from_dict(data)

        assert graph.version == "1.0"
        assert len(graph.nodes) == 1
        assert len(graph.edges) == 1
        assert "mono" in graph.outputs

    def test_json_save_load(self):
        """Test JSON file save and load"""
        # Create graph
        graph = GraphIR(sample_rate=44100, seed=1337)
        outputs = [GraphIROutputPort(name="out", type="Sig")]
        graph.add_node(id="osc1", op="sine", outputs=outputs, params={"freq": "440Hz"})
        graph.add_output("mono", ["osc1:out"])

        # Save to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.morph.json"
            graph.to_json(str(filepath))

            # Check file exists
            assert filepath.exists()

            # Load back
            loaded = GraphIR.from_json(str(filepath))

            assert loaded.sample_rate == 44100
            assert loaded.seed == 1337
            assert len(loaded.nodes) == 1
            assert loaded.nodes[0].id == "osc1"

    def test_json_auto_extension(self):
        """Test automatic .morph.json extension"""
        graph = GraphIR()
        outputs = [GraphIROutputPort(name="out", type="Sig")]
        graph.add_node(id="osc1", op="sine", outputs=outputs)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"
            graph.to_json(str(filepath))

            # Should create test.morph.json
            expected = Path(tmpdir) / "test.morph.json"
            assert expected.exists()


class TestGraphValidator:
    """Test GraphValidator class"""

    def test_valid_simple_graph(self):
        """Test validation of a simple valid graph"""
        graph = GraphIR()
        outputs = [GraphIROutputPort(name="out", type="Sig")]
        graph.add_node(id="osc1", op="sine", outputs=outputs, rate="audio")

        errors = graph.validate()
        assert len(errors) == 0
        assert graph.is_valid()

    def test_duplicate_node_ids(self):
        """Test detection of duplicate node IDs"""
        graph = GraphIR()
        outputs = [GraphIROutputPort(name="out", type="Sig")]
        graph.add_node(id="osc1", op="sine", outputs=outputs)
        graph.add_node(id="osc1", op="saw", outputs=outputs)  # Duplicate!

        errors = graph.validate()
        assert len(errors) > 0
        assert any("Duplicate node ID" in err for err in errors)

    def test_invalid_rate(self):
        """Test detection of invalid rate"""
        graph = GraphIR()
        outputs = [GraphIROutputPort(name="out", type="Sig")]
        graph.add_node(id="osc1", op="sine", outputs=outputs, rate="invalid")

        errors = graph.validate()
        assert len(errors) > 0
        assert any("Invalid rate" in err for err in errors)

    def test_edge_reference_nonexistent_node(self):
        """Test detection of edges referencing non-existent nodes"""
        graph = GraphIR()
        graph.add_edge(from_port="osc1:out", to_port="lpf1:in", type="Sig")

        errors = graph.validate()
        assert len(errors) > 0
        assert any("non-existent" in err for err in errors)

    def test_edge_reference_nonexistent_port(self):
        """Test detection of edges referencing non-existent ports"""
        graph = GraphIR()
        outputs = [GraphIROutputPort(name="out", type="Sig")]
        graph.add_node(id="osc1", op="sine", outputs=outputs)
        graph.add_edge(from_port="osc1:missing", to_port="lpf1:in", type="Sig")

        errors = graph.validate()
        assert len(errors) > 0
        assert any("non-existent output port" in err for err in errors)

    def test_cycle_detection(self):
        """Test detection of cycles in graph"""
        graph = GraphIR()
        outputs = [GraphIROutputPort(name="out", type="Sig")]

        graph.add_node(id="node1", op="sine", outputs=outputs)
        graph.add_node(id="node2", op="lpf", outputs=outputs)
        graph.add_node(id="node3", op="delay", outputs=outputs)

        # Create cycle: node1 -> node2 -> node3 -> node1
        graph.add_edge(from_port="node1:out", to_port="node2:in", type="Sig")
        graph.add_edge(from_port="node2:out", to_port="node3:in", type="Sig")
        graph.add_edge(from_port="node3:out", to_port="node1:in", type="Sig")

        errors = graph.validate()
        assert len(errors) > 0
        assert any("cycle" in err.lower() for err in errors)

    def test_type_mismatch(self):
        """Test detection of incompatible types"""
        graph = GraphIR()

        # Create node with Sig output
        sig_outputs = [GraphIROutputPort(name="out", type="Sig")]
        graph.add_node(id="osc1", op="sine", outputs=sig_outputs)

        # Create node with Ctl input (would need resampling)
        ctl_outputs = [GraphIROutputPort(name="out", type="Ctl")]
        graph.add_node(id="env1", op="adsr", outputs=ctl_outputs)

        # Try to connect incompatibly (this is OK: Ctl -> Sig allowed)
        graph.add_edge(from_port="env1:out", to_port="osc1:in", type="Sig")

        # But Field2D -> Sig should fail
        field_outputs = [GraphIROutputPort(name="out", type="Field2D")]
        graph.add_node(id="field1", op="field", outputs=field_outputs)
        graph.add_edge(from_port="field1:out", to_port="osc1:in", type="Sig")

        errors = graph.validate()
        # Should have errors for field type mismatch
        assert len(errors) > 0

    def test_topological_sort(self):
        """Test topological sort of valid DAG"""
        graph = GraphIR()
        outputs = [GraphIROutputPort(name="out", type="Sig")]

        graph.add_node(id="osc1", op="sine", outputs=outputs)
        graph.add_node(id="lpf1", op="lpf", outputs=outputs)
        graph.add_node(id="mul1", op="multiply", outputs=outputs)

        graph.add_edge(from_port="osc1:out", to_port="lpf1:in", type="Sig")
        graph.add_edge(from_port="lpf1:out", to_port="mul1:in1", type="Sig")

        validator = GraphValidator(graph)
        topo_order = validator.topological_sort()

        assert topo_order is not None
        assert len(topo_order) == 3
        # osc1 must come before lpf1, lpf1 before mul1
        assert topo_order.index("osc1") < topo_order.index("lpf1")
        assert topo_order.index("lpf1") < topo_order.index("mul1")

    def test_topological_sort_with_cycle(self):
        """Test topological sort returns None for cyclic graph"""
        graph = GraphIR()
        outputs = [GraphIROutputPort(name="out", type="Sig")]

        graph.add_node(id="node1", op="sine", outputs=outputs)
        graph.add_node(id="node2", op="lpf", outputs=outputs)

        # Create cycle
        graph.add_edge(from_port="node1:out", to_port="node2:in", type="Sig")
        graph.add_edge(from_port="node2:out", to_port="node1:in", type="Sig")

        validator = GraphValidator(graph)
        topo_order = validator.topological_sort()

        assert topo_order is None


class TestComplexGraphs:
    """Test complex graph scenarios"""

    def test_simple_synth_example(self):
        """
        Test the simple synth example from the spec

        Graph: sine -> lpf -> multiply <- adsr
        """
        graph = GraphIR(sample_rate=48000, seed=1337)

        # Create nodes
        graph.add_node(
            id="osc1",
            op="sine",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
            params={"freq": "440Hz"},
        )

        graph.add_node(
            id="lpf1",
            op="lpf",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
            params={"cutoff": "2kHz", "q": 0.707},
        )

        graph.add_node(
            id="env1",
            op="adsr",
            outputs=[GraphIROutputPort(name="out", type="Ctl")],
            rate="control",
            params={
                "attack": "0.01s",
                "decay": "0.1s",
                "sustain": 0.7,
                "release": "0.3s",
            },
        )

        graph.add_node(
            id="mul1",
            op="multiply",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
        )

        # Create edges
        graph.add_edge(from_port="osc1:out", to_port="lpf1:in", type="Sig")
        graph.add_edge(from_port="lpf1:out", to_port="mul1:in1", type="Sig")
        graph.add_edge(from_port="env1:out", to_port="mul1:in2", type="Ctl")

        # Add output
        graph.add_output("mono", ["mul1:out"])

        # Validate
        errors = graph.validate()
        assert len(errors) == 0

        # Test serialization
        data = graph.to_dict()
        assert len(data["nodes"]) == 4
        assert len(data["edges"]) == 3

        # Test deserialization
        loaded = GraphIR.from_dict(data)
        assert len(loaded.nodes) == 4

    def test_stereo_panner(self):
        """Test graph with stereo output"""
        graph = GraphIR()

        # Mono source
        graph.add_node(
            id="osc1",
            op="sine",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
        )

        # Stereo panner
        graph.add_node(
            id="pan1",
            op="pan",
            outputs=[
                GraphIROutputPort(name="left", type="Sig"),
                GraphIROutputPort(name="right", type="Sig"),
            ],
            rate="audio",
            params={"pos": 0.0},
        )

        graph.add_edge(from_port="osc1:out", to_port="pan1:in", type="Sig")

        # Stereo output
        graph.add_output("stereo", ["pan1:left", "pan1:right"])

        errors = graph.validate()
        assert len(errors) == 0

    def test_multirate_graph(self):
        """Test graph with multiple rate classes"""
        graph = GraphIR()

        # Audio rate oscillator
        graph.add_node(
            id="osc1",
            op="sine",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
            params={"freq": "440Hz"},
        )

        # Control rate LFO
        graph.add_node(
            id="lfo1",
            op="sine",
            outputs=[GraphIROutputPort(name="out", type="Ctl")],
            rate="control",
            params={"freq": "2Hz"},
        )

        # Visual rate display
        graph.add_node(
            id="scope1",
            op="scope",
            outputs=[GraphIROutputPort(name="out", type="Image")],
            rate="visual",
        )

        errors = graph.validate()
        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
