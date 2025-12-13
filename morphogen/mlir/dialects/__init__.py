"""Morphogen Custom MLIR Dialects

This package contains custom MLIR dialect definitions for Morphogen's
domain-specific operations.

Dialects:
- field: Field operations (gradients, diffusion, advection) - Phase 2 ✅
- temporal: Temporal execution with flow blocks and state - Phase 3 ✅
- agent: Agent-based modeling operations - Phase 4 ✅
- audio: Audio synthesis operations - Phase 5 ✅
- visual: Visual rendering operations - Phase 6 (TODO)

These dialects are implemented progressively during v0.7.0 development.
"""

# Phase 2: Field dialect
from .field import FieldDialect, FieldType, MLIR_AVAILABLE

# Phase 3: Temporal dialect
from .temporal import TemporalDialect, FlowType, StateType

# Phase 4: Agent dialect
from .agent import AgentDialect, AgentType

# Phase 5: Audio dialect
from .audio import AudioDialect, AudioType

# TODO: Future phases
# from .visual import MorphogenVisualDialect

__all__ = [
    "FieldDialect",
    "FieldType",
    "TemporalDialect",
    "FlowType",
    "StateType",
    "AgentDialect",
    "AgentType",
    "AudioDialect",
    "AudioType",
    "MLIR_AVAILABLE",
]
