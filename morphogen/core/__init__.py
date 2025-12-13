"""
Morphogen Core - Domain registry and operator system.

This module provides the core infrastructure for domain registration,
operator metadata, and language integration.
"""

from morphogen.core.operator import operator, OpCategory
from morphogen.core.domain_registry import DomainRegistry, DomainDescriptor

__all__ = [
    'operator',
    'OpCategory',
    'DomainRegistry',
    'DomainDescriptor',
]
