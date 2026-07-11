"""
Cross-Domain Integration for Morphogen

This module provides the infrastructure for composing operators across different
computational domains (Field, Agent, Audio, Physics, Geometry, etc.).

Key components:
- DomainInterface: Base class for inter-domain data flows
- Transform functions: Convert data between domain formats
- Type validation: Ensure compatibility across domain boundaries
- Composition operators: compose() and link() support
- Path finding: find_transform_path() discovers whether registered bridges connect two domains

Note: the old "composition engine" (TransformComposer/TransformPipeline/auto_compose)
was retired 2026-07-11 — it never executed (see composer.py). For time-coupled,
feedback co-simulation across domains, use morphogen.coupling.
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

# Controls ↔ Physics
from .controls_physics import (
    PhysicsToControlsInterface,
    ControlsToPhysicsInterface,
)

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

# Path finding + explicit composition (the composition *engine* was retired — see below)
from .composer import (
    compose,
    find_transform_path,
)

# Retired 2026-07-11 (BACKLOG P0-2): the auto-composition engine never executed.
# Raise a clear, guiding error instead of silently 404-ing an import.
_RETIRED_COMPOSER = {
    "TransformComposer", "TransformPipeline", "BatchTransformComposer", "auto_compose",
}


def __getattr__(name):
    if name in _RETIRED_COMPOSER:
        raise AttributeError(
            f"{name!r} was retired on 2026-07-11: the cross-domain auto-composition "
            f"engine never executed (every built-in transform rejected its source_data= "
            f"kwarg). Use compose(*explicitly_built_interfaces) for one-shot chaining, "
            f"find_transform_path() to discover routes, or morphogen.coupling.couple() "
            f"for time-coupled feedback co-simulation."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
    # Controls ↔ Physics
    'PhysicsToControlsInterface',
    'ControlsToPhysicsInterface',
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
    # Explicit composition + path finding (auto-composition engine retired 2026-07-11)
    'compose',
    'find_transform_path',
]

# Note: Cross-domain transforms are auto-registered via
# registry.register_builtin_transforms() on module import.
# No explicit registration needed here.
