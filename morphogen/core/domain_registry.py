"""
Domain registration and lookup system.

This module provides the central registry for all Morphogen domains,
enabling runtime discovery of operators and type information.
"""

from typing import Dict, Callable, Any, List
from dataclasses import dataclass, field
import importlib
import inspect

from morphogen.core.operator import get_operator_metadata, OperatorMetadata


@dataclass
class DomainDescriptor:
    """Metadata about a domain."""
    name: str
    module_path: str
    operators: Dict[str, Callable] = field(default_factory=dict)
    types: Dict[str, type] = field(default_factory=dict)
    version: str = "0.10.0"
    description: str = ""

    def get_operator(self, name: str) -> Callable:
        """Get operator by name."""
        if name not in self.operators:
            raise ValueError(f"Operator '{name}' not found in domain '{self.name}'")
        return self.operators[name]

    def list_operators(self) -> List[str]:
        """List all operator names."""
        return sorted(self.operators.keys())

    def get_operator_metadata(self, name: str) -> OperatorMetadata:
        """Get metadata for an operator."""
        op = self.get_operator(name)
        metadata = get_operator_metadata(op)
        if metadata is None:
            raise ValueError(f"Operator '{name}' has no metadata")
        return metadata


class DomainRegistry:
    """Central registry for all Morphogen domains."""

    _domains: Dict[str, DomainDescriptor] = {}
    _initialized: bool = False

    @classmethod
    def register(cls, name: str, module_path: str, description: str = "") -> DomainDescriptor:
        """
        Register a domain.

        Args:
            name: Domain name (e.g., "graph", "audio")
            module_path: Python module path (e.g., "morphogen.stdlib.graph")
            description: Optional description

        Returns:
            DomainDescriptor for the registered domain

        Raises:
            ImportError: If module cannot be imported
            ValueError: If domain is already registered
        """
        if name in cls._domains:
            raise ValueError(f"Domain '{name}' is already registered")

        # Import module
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(f"Failed to import domain module '{module_path}': {e}")

        # Discover operators (functions decorated with @operator)
        operators = {}
        for attr_name in dir(module):
            # Skip private attributes
            if attr_name.startswith('_'):
                continue

            attr = getattr(module, attr_name)

            # Check if it's an operator (has _operator_metadata)
            if hasattr(attr, '_operator_metadata'):
                operators[attr_name] = attr

        # Create descriptor
        descriptor = DomainDescriptor(
            name=name,
            module_path=module_path,
            operators=operators,
            types={},  # TODO: type discovery
            version="0.10.0",
            description=description
        )

        cls._domains[name] = descriptor
        return descriptor

    @classmethod
    def get(cls, name: str) -> DomainDescriptor:
        """
        Get domain descriptor.

        Args:
            name: Domain name

        Returns:
            DomainDescriptor

        Raises:
            ValueError: If domain not registered
        """
        if name not in cls._domains:
            raise ValueError(f"Domain '{name}' not registered. Available domains: {cls.list_domains()}")
        return cls._domains[name]

    @classmethod
    def list_domains(cls) -> List[str]:
        """
        List all registered domains.

        Returns:
            Sorted list of domain names
        """
        return sorted(cls._domains.keys())

    @classmethod
    def has_domain(cls, name: str) -> bool:
        """
        Check if a domain is registered.

        Args:
            name: Domain name

        Returns:
            True if domain is registered, False otherwise
        """
        return name in cls._domains

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Unregister a domain (mainly for testing).

        Args:
            name: Domain name
        """
        if name in cls._domains:
            del cls._domains[name]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered domains (mainly for testing)."""
        cls._domains.clear()
        cls._initialized = False

    @classmethod
    def initialize(cls) -> None:
        """Initialize the registry with all stdlib domains."""
        if cls._initialized:
            return

        register_stdlib_domains()
        cls._initialized = True

    @classmethod
    def get_operator(cls, domain_name: str, operator_name: str) -> Callable:
        """
        Get an operator from a domain.

        Args:
            domain_name: Domain name
            operator_name: Operator name

        Returns:
            Operator function

        Raises:
            ValueError: If domain or operator not found
        """
        domain = cls.get(domain_name)
        return domain.get_operator(operator_name)


def register_stdlib_domains() -> None:
    """
    Register all built-in stdlib domains.

    This function is called automatically during initialization.
    """
    domains = [
        ("field", "morphogen.stdlib.field", "Spatial field simulation and computation"),
        ("agent", "morphogen.stdlib.agents", "Agent-based modeling and simulation"),
        ("audio", "morphogen.stdlib.audio", "Audio synthesis and processing"),
        ("visual", "morphogen.stdlib.visual", "Visual rendering and graphics"),
        ("rigidbody", "morphogen.stdlib.rigidbody", "Rigid body physics simulation"),
        ("integrators", "morphogen.stdlib.integrators", "Numerical integration methods"),
        ("graph", "morphogen.stdlib.graph", "Graph and network algorithms"),
        ("signal", "morphogen.stdlib.signal", "Signal processing and analysis"),
        ("statemachine", "morphogen.stdlib.statemachine", "State machines and behavior trees"),
        ("terrain", "morphogen.stdlib.terrain", "Procedural terrain generation"),
        ("vision", "morphogen.stdlib.vision", "Computer vision and image processing"),
        ("cellular", "morphogen.stdlib.cellular", "Cellular automata"),
        ("optimization", "morphogen.stdlib.optimization", "Optimization algorithms"),
        ("neural", "morphogen.stdlib.neural", "Neural network primitives"),
        ("sparse_linalg", "morphogen.stdlib.sparse_linalg", "Sparse linear algebra"),
        ("io_storage", "morphogen.stdlib.io_storage", "I/O and data storage"),
        ("acoustics", "morphogen.stdlib.acoustics", "Room acoustics and propagation"),
        ("noise", "morphogen.stdlib.noise", "Noise generation algorithms"),
        ("color", "morphogen.stdlib.color", "Color space operations"),
        ("image", "morphogen.stdlib.image", "Image manipulation"),
        ("palette", "morphogen.stdlib.palette", "Color palette generation"),
        ("genetic", "morphogen.stdlib.genetic", "Genetic algorithms"),
        ("temporal", "morphogen.stdlib.temporal", "Temporal logic and scheduling"),
        ("geometry", "morphogen.stdlib.geometry", "2D/3D geometry, spatial operations, and mesh processing"),
        ("circuit", "morphogen.stdlib.circuit", "Circuit and electrical simulation"),
        ("molecular", "morphogen.stdlib.molecular", "Molecular structure, mechanics, and dynamics"),
        ("qchem", "morphogen.stdlib.qchem", "Quantum chemistry calculations and ML potential energy surfaces"),
        ("thermo", "morphogen.stdlib.thermo", "Thermodynamic properties and equations of state"),
        ("thermal_ode", "morphogen.stdlib.thermal_ode", "1D thermal modeling and heat transfer"),
        ("fluid_network", "morphogen.stdlib.fluid_network", "1D fluid network modeling"),
        ("kinetics", "morphogen.stdlib.kinetics", "Chemical reaction kinetics and reactor modeling"),
        ("electrochem", "morphogen.stdlib.electrochem", "Electrochemistry and battery simulation"),
        ("catalysis", "morphogen.stdlib.catalysis", "Heterogeneous catalysis and surface kinetics"),
        ("transport", "morphogen.stdlib.transport", "Heat and mass transport phenomena"),
        ("multiphase", "morphogen.stdlib.multiphase", "Vapor-liquid equilibrium and multiphase systems"),
        ("combustion", "morphogen.stdlib.combustion_light", "Simplified combustion metrics"),
        ("fluid_jet", "morphogen.stdlib.fluid_jet", "Jet flow dynamics and mixing"),
        ("audio_analysis", "morphogen.stdlib.audio_analysis", "Audio signal processing and timbre analysis"),
        ("instrument_model", "morphogen.stdlib.instrument_model", "Musical instrument modeling and synthesis"),
    ]

    for name, module_path, description in domains:
        try:
            DomainRegistry.register(name, module_path, description)
        except ImportError as e:
            # Silently skip domains that can't be imported
            # (allows gradual migration to @operator decorators)
            pass
        except Exception as e:
            # Log other errors but continue
            print(f"Warning: Failed to register domain '{name}': {e}")
