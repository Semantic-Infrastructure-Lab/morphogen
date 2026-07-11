"""
Domain Interface Base Classes

Provides the foundational abstractions for cross-domain data flows in Morphogen.
Based on ADR-002: Cross-Domain Architectural Patterns.

This module contains the base classes that all domain interfaces inherit from,
extracted to break circular import dependencies between interface.py and registry.py.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Type
from dataclasses import dataclass


@dataclass
class DomainMetadata:
    """Metadata describing a domain's capabilities and interfaces."""

    name: str
    version: str
    input_types: Set[str]  # What types this domain can accept
    output_types: Set[str]  # What types this domain can provide
    dependencies: List[str]  # Other domains this depends on
    description: str


class DomainInterface(ABC):
    """
    Base class for inter-domain data flows.

    Each domain pair (source → target) that supports composition must implement
    a DomainInterface subclass that handles:
    1. Type validation
    2. Data transformation
    3. Metadata propagation

    Example:
        class FieldToAgentInterface(DomainInterface):
            source_domain = "field"
            target_domain = "agent"

            def transform(self, field_data):
                # Sample field at agent positions
                return sampled_values

            def validate(self):
                # Check field dimensions, agent count, etc.
                return True
    """

    source_domain: str = None  # Set by subclass
    target_domain: str = None  # Set by subclass

    def __init__(self, source_data: Any = None, metadata: Optional[Dict] = None):
        self.source_data = source_data
        self.metadata = metadata or {}
        self._validated = False

    @abstractmethod
    def transform(self, source_data: Any) -> Any:
        """
        Convert source domain data to target domain format.

        Args:
            source_data: Data in source domain format

        Returns:
            Data in target domain format

        Raises:
            ValueError: If data cannot be transformed
            TypeError: If data types are incompatible
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement transform()"
        )

    @abstractmethod
    def validate(self) -> bool:
        """
        Ensure data types are compatible across domains.

        Returns:
            True if transformation is valid, False otherwise

        Raises:
            CrossDomainTypeError: If types are fundamentally incompatible
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement validate()"
        )

    def get_input_interface(self) -> Dict[str, Type]:
        """
        Describe what data this interface can accept.

        Returns:
            Dict mapping parameter names to their types
        """
        return {}

    def get_output_interface(self) -> Dict[str, Type]:
        """
        Describe what data this interface can provide.

        Returns:
            Dict mapping output names to their types
        """
        return {}

    def __call__(self, source_data: Any) -> Any:
        """
        Convenience method: validate and transform in one call.

        Args:
            source_data: Data to transform

        Returns:
            Transformed data
        """
        self.source_data = source_data
        if not self._validated:
            if not self.validate():
                raise ValueError(
                    f"Cross-domain flow {self.source_domain} → {self.target_domain} "
                    f"failed validation"
                )
            self._validated = True

        return self.transform(source_data)


class DomainTransform:
    """
    Decorator for registering cross-domain transform functions.

    Example:
        @DomainTransform(source="field", target="agent")
        def field_to_agent_force(field, agent_positions):
            '''Sample field values at agent positions.'''
            return sample_field(field, agent_positions)
    """

    def __init__(
        self,
        source: str,
        target: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        input_types: Optional[Dict[str, Type]] = None,
        output_type: Optional[Type] = None,
    ):
        self.source = source
        self.target = target
        self.name = name
        self.description = description
        self.input_types = input_types or {}
        self.output_type = output_type
        self.transform_fn = None

    def __call__(self, fn):
        """Wrap the decorated function in a DomainInterface-compatible class.

        The original function stays callable as-is for inline use, and the
        generated ``DomainInterface`` subclass is attached to it as
        ``fn.interface`` so the transform can be validated and registered rather
        than silently discarded.
        """
        self.transform_fn = fn
        self.name = self.name or fn.__name__
        self.description = self.description or fn.__doc__

        # Capture self in closure for the inner class
        outer_self = self

        # Create a DomainInterface wrapper
        class TransformInterface(DomainInterface):
            source_domain = outer_self.source
            target_domain = outer_self.target

            def transform(iself, source_data: Any) -> Any:
                return fn(source_data)

            def validate(iself) -> bool:
                """Check that source_data matches the declared input types.

                When ``input_types`` is declared on the decorator, the first
                positional input (``source_data``) is validated against the
                first declared type. Declared types may be a single ``type``,
                a tuple of types, or ``None`` (meaning "accept anything").
                Raises ``TypeError`` on a mismatch so a bad cross-domain flow
                fails loudly instead of silently producing garbage.
                """
                if not outer_self.input_types:
                    return True

                if iself.source_data is None:
                    return True  # nothing to check yet

                # input_types maps parameter name -> expected type; the
                # transform receives source_data as its first argument, so
                # validate against the first declared parameter's type.
                first_param, expected = next(iter(outer_self.input_types.items()))
                if expected is None:
                    return True
                if not isinstance(iself.source_data, expected):
                    exp_name = getattr(expected, "__name__", str(expected))
                    raise TypeError(
                        f"{outer_self.source} -> {outer_self.target} transform "
                        f"'{outer_self.name}' expects '{first_param}' of type "
                        f"{exp_name}, got {type(iself.source_data).__name__}"
                    )
                return True

        # Store metadata
        TransformInterface.__name__ = f"{self.source}To{self.target.capitalize()}Transform"
        TransformInterface.__doc__ = self.description

        # Attach the interface class to the function so it isn't discarded.
        # Inline callers still use `fn` directly; interface-based callers use
        # `fn.interface()` to get validation + transformation.
        fn.interface = TransformInterface

        return fn
