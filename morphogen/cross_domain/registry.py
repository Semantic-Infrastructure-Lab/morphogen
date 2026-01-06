"""
Cross-Domain Transform Registry

Maintains a global registry of all available cross-domain transformations.
Supports discovery, validation, and automatic selection of transforms.

Built-in transforms are automatically registered when this module is imported.
No explicit initialization required for normal usage.
"""

from typing import Dict, List, Optional, Tuple, Type
from .base import DomainInterface


class CrossDomainRegistry:
    """
    Global registry for cross-domain transforms.

    Maintains mappings:
        (source_domain, target_domain) → DomainInterface subclass

    Usage:
        # Register a transform
        CrossDomainRegistry.register("field", "agent", FieldToAgentInterface)

        # Lookup a transform
        transform_class = CrossDomainRegistry.get("field", "agent")

        # List all transforms for a domain
        transforms = CrossDomainRegistry.list_transforms("field")
    """

    _registry: Dict[Tuple[str, str], Type[DomainInterface]] = {}
    _metadata: Dict[Tuple[str, str], Dict] = {}
    _builtins_registered: bool = False

    @classmethod
    def register(
        cls,
        source: str,
        target: str,
        interface_class: Type[DomainInterface],
        metadata: Optional[Dict] = None
    ):
        """
        Register a cross-domain transform.

        Args:
            source: Source domain name
            target: Target domain name
            interface_class: DomainInterface subclass
            metadata: Optional metadata (description, examples, etc.)
        """
        key = (source, target)

        if key in cls._registry:
            # Allow re-registration (for updates), but warn
            print(f"Warning: Overwriting transform {source} → {target}")

        cls._registry[key] = interface_class
        cls._metadata[key] = metadata or {}

    @classmethod
    def get(
        cls,
        source: str,
        target: str,
        raise_if_missing: bool = True
    ) -> Optional[Type[DomainInterface]]:
        """
        Get a registered transform.

        Args:
            source: Source domain name
            target: Target domain name
            raise_if_missing: If True, raise KeyError if not found

        Returns:
            DomainInterface subclass, or None if not found

        Raises:
            KeyError: If transform not registered and raise_if_missing=True
        """
        key = (source, target)

        if key not in cls._registry:
            if raise_if_missing:
                raise KeyError(
                    f"No transform registered for {source} → {target}. "
                    f"Available transforms: {cls.list_all()}"
                )
            return None

        return cls._registry[key]

    @classmethod
    def list_transforms(cls, domain: str, direction: str = "both") -> List[Tuple[str, str]]:
        """
        List all transforms involving a domain.

        Args:
            domain: Domain name
            direction: "source", "target", or "both"

        Returns:
            List of (source, target) tuples
        """
        transforms = []

        for (source, target) in cls._registry.keys():
            if direction == "source" and source == domain:
                transforms.append((source, target))
            elif direction == "target" and target == domain:
                transforms.append((source, target))
            elif direction == "both" and (source == domain or target == domain):
                transforms.append((source, target))

        return transforms

    @classmethod
    def list_all(cls) -> List[Tuple[str, str]]:
        """List all registered transforms."""
        return list(cls._registry.keys())

    @classmethod
    def has_transform(cls, source: str, target: str) -> bool:
        """Check if a transform is registered."""
        return (source, target) in cls._registry

    @classmethod
    def get_metadata(cls, source: str, target: str) -> Dict:
        """Get metadata for a transform."""
        key = (source, target)
        return cls._metadata.get(key, {})

    @classmethod
    def clear(cls):
        """Clear all registered transforms (useful for testing)."""
        cls._registry.clear()
        cls._metadata.clear()
        cls._builtins_registered = False

    @classmethod
    def visualize(cls) -> str:
        """
        Create a text visualization of the transform graph.

        Returns:
            String representation of domain connections
        """
        if not cls._registry:
            return "No transforms registered."

        # Group by source domain
        from collections import defaultdict
        graph = defaultdict(list)

        for (source, target) in cls._registry.keys():
            graph[source].append(target)

        lines = ["Cross-Domain Transform Graph:", ""]

        for source in sorted(graph.keys()):
            targets = ", ".join(sorted(graph[source]))
            lines.append(f"  {source} → {targets}")

        return "\n".join(lines)


def register_transform(
    source: str,
    target: str,
    metadata: Optional[Dict] = None
):
    """
    Decorator to register a DomainInterface subclass.

    Example:
        @register_transform("field", "agent", metadata={"version": "1.0"})
        class FieldToAgentInterface(DomainInterface):
            ...
    """
    def decorator(interface_class: Type[DomainInterface]):
        CrossDomainRegistry.register(source, target, interface_class, metadata)
        return interface_class

    return decorator


# ============================================================================
# Auto-register built-in transforms
# ============================================================================

def register_builtin_transforms():
    """Register all built-in cross-domain transforms.

    Idempotent: Safe to call multiple times. Only registers once.
    Transforms are auto-registered on module import, but explicit calls
    are harmless (useful for ensuring initialization).
    """
    # Guard: Only register once
    if CrossDomainRegistry._builtins_registered:
        return

    from .field_agent import FieldToAgentInterface, AgentToFieldInterface
    from .audio_visual import AudioToVisualInterface, FieldToAudioInterface
    from .physics_audio import PhysicsToAudioInterface, FluidToAcousticsInterface, AcousticsToAudioInterface
    from .terrain import TerrainToFieldInterface, FieldToTerrainInterface
    from .vision_field import VisionToFieldInterface
    from .graph_visual import GraphToVisualInterface
    from .cellular import CellularToFieldInterface
    from .spectral import TimeToCepstralInterface, CepstralToTimeInterface, TimeToWaveletInterface
    from .spatial import SpatialAffineInterface, CartesianToPolarInterface, PolarToCartesianInterface

    # Original Phase 1 transforms
    CrossDomainRegistry.register(
        "field", "agent",
        FieldToAgentInterface,
        metadata={
            "description": "Sample field values at agent positions",
            "use_cases": [
                "Flow field → particle forces",
                "Temperature field → agent behavior",
                "Density field → agent sensing"
            ]
        }
    )

    CrossDomainRegistry.register(
        "agent", "field",
        AgentToFieldInterface,
        metadata={
            "description": "Deposit agent properties onto field grid",
            "use_cases": [
                "Particle positions → density field",
                "Agent velocities → velocity field",
                "Agent heat → temperature sources"
            ]
        }
    )

    CrossDomainRegistry.register(
        "physics", "audio",
        PhysicsToAudioInterface,
        metadata={
            "description": "Sonification of physical events",
            "use_cases": [
                "Collision forces → percussion",
                "Body velocities → pitch/volume",
                "Contact points → spatial audio"
            ]
        }
    )

    # Phase 2 transforms - Audio ↔ Visual
    CrossDomainRegistry.register(
        "audio", "visual",
        AudioToVisualInterface,
        metadata={
            "description": "Audio-reactive visual generation",
            "use_cases": [
                "FFT spectrum → color palette",
                "Amplitude → particle emission",
                "Beat detection → visual effects"
            ]
        }
    )

    # Phase 2 transforms - Field ↔ Audio
    CrossDomainRegistry.register(
        "field", "audio",
        FieldToAudioInterface,
        metadata={
            "description": "Field-driven audio synthesis",
            "use_cases": [
                "Temperature → synthesis parameters",
                "Vorticity → frequency modulation",
                "Field evolution → audio sequences"
            ]
        }
    )

    # Phase 2 transforms - Terrain ↔ Field
    CrossDomainRegistry.register(
        "terrain", "field",
        TerrainToFieldInterface,
        metadata={
            "description": "Convert terrain heightmap to scalar field",
            "use_cases": [
                "Heightmap → diffusion initial conditions",
                "Elevation → potential field",
                "Terrain features → field patterns"
            ]
        }
    )

    CrossDomainRegistry.register(
        "field", "terrain",
        FieldToTerrainInterface,
        metadata={
            "description": "Convert scalar field to terrain heightmap",
            "use_cases": [
                "Procedural field → terrain generation",
                "Simulation result → landscape"
            ]
        }
    )

    # Phase 2 transforms - Vision → Field
    CrossDomainRegistry.register(
        "vision", "field",
        VisionToFieldInterface,
        metadata={
            "description": "Convert computer vision features to fields",
            "use_cases": [
                "Edge map → scalar field",
                "Optical flow → vector field",
                "Feature map → field initialization"
            ]
        }
    )

    # Phase 2 transforms - Graph → Visual
    CrossDomainRegistry.register(
        "graph", "visual",
        GraphToVisualInterface,
        metadata={
            "description": "Network graph visualization",
            "use_cases": [
                "Network structure → visual layout",
                "Graph metrics → node colors/sizes",
                "Connectivity → edge rendering"
            ]
        }
    )

    # Phase 2 transforms - Cellular → Field
    CrossDomainRegistry.register(
        "cellular", "field",
        CellularToFieldInterface,
        metadata={
            "description": "Convert cellular automata state to field",
            "use_cases": [
                "CA state → PDE initial conditions",
                "Game of Life → density field",
                "Pattern state → field patterns"
            ]
        }
    )

    # Time-Frequency Domain Transforms
    CrossDomainRegistry.register(
        "time", "cepstral",
        TimeToCepstralInterface,
        metadata={
            "description": "Time → Cepstral via DCT",
            "use_cases": [
                "Audio compression (MP3, AAC)",
                "MFCC computation",
                "Pitch detection",
                "Speech feature extraction"
            ]
        }
    )

    CrossDomainRegistry.register(
        "cepstral", "time",
        CepstralToTimeInterface,
        metadata={
            "description": "Cepstral → Time via IDCT",
            "use_cases": [
                "Signal reconstruction from DCT coefficients",
                "Audio decompression",
                "Cepstral-based synthesis"
            ]
        }
    )

    CrossDomainRegistry.register(
        "time", "wavelet",
        TimeToWaveletInterface,
        metadata={
            "description": "Time → Wavelet via CWT",
            "use_cases": [
                "Non-stationary signal analysis",
                "Multi-scale feature detection",
                "Transient analysis",
                "Time-frequency localization"
            ]
        }
    )

    # Spatial Domain Transforms
    CrossDomainRegistry.register(
        "spatial", "spatial",
        SpatialAffineInterface,
        metadata={
            "description": "Spatial affine transformations (translate, rotate, scale, shear)",
            "use_cases": [
                "Image/field registration",
                "Data augmentation",
                "Coordinate alignment",
                "Geometric transformations"
            ]
        }
    )

    CrossDomainRegistry.register(
        "cartesian", "polar",
        CartesianToPolarInterface,
        metadata={
            "description": "Cartesian → Polar coordinate conversion",
            "use_cases": [
                "Radial pattern analysis",
                "Rotational symmetry detection",
                "Circular data visualization",
                "Fourier-Bessel transforms"
            ]
        }
    )

    CrossDomainRegistry.register(
        "polar", "cartesian",
        PolarToCartesianInterface,
        metadata={
            "description": "Polar → Cartesian coordinate conversion",
            "use_cases": [
                "Field reconstruction after radial processing",
                "Polar data visualization",
                "Inverse transforms"
            ]
        }
    )

    # Phase 3 transforms - Fluid-Acoustics-Audio Pipeline
    CrossDomainRegistry.register(
        "fluid", "acoustics",
        FluidToAcousticsInterface,
        metadata={
            "description": (
                "Fluid → Acoustics: Couple fluid pressure to "
                "acoustic wave propagation"
            ),
            "use_cases": [
                "CFD pressure fields → acoustic wave equation",
                "Turbulent flow → aeroacoustic sound",
                "Vortex shedding → acoustic radiation",
                "Fluid-structure interaction → sound generation"
            ]
        }
    )

    CrossDomainRegistry.register(
        "acoustics", "audio",
        AcousticsToAudioInterface,
        metadata={
            "description": (
                "Acoustics → Audio: Sample acoustic field at "
                "microphones and synthesize audio"
            ),
            "use_cases": [
                "Acoustic pressure → audio waveform",
                "Virtual microphone sampling",
                "Spatial audio from acoustic fields",
                "CFD aeroacoustics → audio rendering"
            ]
        }
    )

    # Mark builtins as registered
    CrossDomainRegistry._builtins_registered = True


# Auto-register on import
register_builtin_transforms()
