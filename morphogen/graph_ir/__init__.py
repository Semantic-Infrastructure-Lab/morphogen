"""
Morphogen Graph IR - Intermediate Representation for Morphogen Graphs

This module provides the canonical JSON-based intermediate representation
between Morphogen frontends (DSL, RiffStack) and the Morphogen Kernel.

Key Features:
- Frontend-agnostic graph representation
- Type-safe with full annotations (types, units, domains, rates)
- DAG validation and type checking
- Human-readable JSON serialization
- Deterministic execution guarantees

Version: 1.0
"""

from .core import (
    GraphIRNode,
    GraphIREdge,
    GraphIROutputPort,
    GraphIROutput,
    GraphIREvent,
    GraphIR,
)
from .validation import GraphValidator

__all__ = [
    "GraphIRNode",
    "GraphIREdge",
    "GraphIROutputPort",
    "GraphIROutput",
    "GraphIREvent",
    "GraphIR",
    "GraphValidator",
]

__version__ = "1.0.0"
