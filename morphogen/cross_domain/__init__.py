"""
Cross-Domain Integration for Morphogen

This module provides the infrastructure for composing operators across different
computational domains (Field, Agent, Audio, Physics, Geometry, etc.).

Key components:
- DomainInterface: Base class for inter-domain data flows
- Transform functions: Convert data between domain formats
- Type validation: Ensure compatibility across domain boundaries
- Composition operators: compose() and link() support
- Transform composition: Automatic path finding and pipeline execution
"""

from .interface import (
    DomainInterface,
    DomainTransform,
    FieldToAgentInterface,
    AgentToFieldInterface,
    PhysicsToAudioInterface,
    AudioToVisualInterface,
    FieldToAudioInterface,
    TerrainToFieldInterface,
    FieldToTerrainInterface,
    VisionToFieldInterface,
    GraphToVisualInterface,
    CellularToFieldInterface,
    FluidToAcousticsInterface,
    AcousticsToAudioInterface,
)
from .registry import CrossDomainRegistry, register_transform
from .validators import (
    validate_cross_domain_flow,
    CrossDomainTypeError,
    CrossDomainValidationError,
)
from .composer import (
    TransformComposer,
    TransformPipeline,
    BatchTransformComposer,
    compose,
    find_transform_path,
    auto_compose,
)

__all__ = [
    # Base infrastructure
    'DomainInterface',
    'DomainTransform',
    'CrossDomainRegistry',
    'register_transform',
    'validate_cross_domain_flow',
    'CrossDomainTypeError',
    'CrossDomainValidationError',
    # Phase 1 transforms
    'FieldToAgentInterface',
    'AgentToFieldInterface',
    'PhysicsToAudioInterface',
    # Phase 2 transforms
    'AudioToVisualInterface',
    'FieldToAudioInterface',
    'TerrainToFieldInterface',
    'FieldToTerrainInterface',
    'VisionToFieldInterface',
    'GraphToVisualInterface',
    'CellularToFieldInterface',
    # Phase 3 transforms (3-domain pipeline)
    'FluidToAcousticsInterface',
    'AcousticsToAudioInterface',
    # Composition engine
    'TransformComposer',
    'TransformPipeline',
    'BatchTransformComposer',
    'compose',
    'find_transform_path',
    'auto_compose',
]

# Note: Cross-domain transforms are auto-registered via
# registry.register_builtin_transforms() on module import.
# No explicit registration needed here.
