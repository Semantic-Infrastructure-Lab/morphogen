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

# Base classes
from .base import (
    DomainInterface,
    DomainMetadata,
    DomainTransform,
)

# Field ↔ Agent
from .field_agent import (
    FieldToAgentInterface,
    AgentToFieldInterface,
)

# Audio ↔ Visual
from .audio_visual import (
    AudioToVisualInterface,
    FieldToAudioInterface,
)

# Physics → Audio
from .physics_audio import (
    PhysicsToAudioInterface,
    FluidToAcousticsInterface,
    AcousticsToAudioInterface,
)

# Terrain ↔ Field
from .terrain import (
    TerrainToFieldInterface,
    FieldToTerrainInterface,
)

# Vision → Field
from .vision_field import VisionToFieldInterface

# Graph → Visual
from .graph_visual import GraphToVisualInterface

# Cellular → Field
from .cellular import CellularToFieldInterface

# Time-Frequency Transforms
from .spectral import (
    TimeToCepstralInterface,
    CepstralToTimeInterface,
    TimeToWaveletInterface,
)

# Spatial Transforms
from .spatial import (
    SpatialAffineInterface,
    CartesianToPolarInterface,
    PolarToCartesianInterface,
)

# Registry
from .registry import CrossDomainRegistry, register_transform

# Validators
from .validators import (
    validate_cross_domain_flow,
    CrossDomainTypeError,
    CrossDomainValidationError,
)

# Composer
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
    'DomainMetadata',
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
    # Time-Frequency transforms
    'TimeToCepstralInterface',
    'CepstralToTimeInterface',
    'TimeToWaveletInterface',
    # Spatial transforms
    'SpatialAffineInterface',
    'CartesianToPolarInterface',
    'PolarToCartesianInterface',
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
