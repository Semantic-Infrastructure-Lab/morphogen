"""
Transform Composition Engine

Enables automatic chaining and optimization of cross-domain transforms.
Supports A→B→C pipelines with caching and performance optimization.
"""

from typing import Any, List, Optional, Tuple, Dict, Callable
from dataclasses import dataclass
import numpy as np
from .base import DomainInterface
from .registry import CrossDomainRegistry


@dataclass
class TransformNode:
    """Node in a transform composition graph."""

    source_domain: str
    target_domain: str
    transform_class: type
    params: Dict[str, Any]

    def __repr__(self):
        return f"{self.source_domain}→{self.target_domain}"


class TransformComposer:
    """
    Composes multiple cross-domain transforms into optimized pipelines.

    Features:
    - Automatic path finding (A→C via A→B→C)
    - Transform caching for repeated operations
    - Batch processing optimization
    - Pipeline visualization

    Example:
        # Create composer
        composer = TransformComposer()

        # Build pipeline: Field → Audio → Visual
        pipeline = composer.compose_path("field", "visual", via=["audio"])

        # Execute pipeline
        result = pipeline(field_data)
    """

    def __init__(self, enable_caching: bool = True):
        """
        Initialize transform composer.

        Args:
            enable_caching: If True, cache transform results for repeated inputs
        """
        self.enable_caching = enable_caching
        self._cache: Dict[int, Any] = {}
        self._stats = {
            'transforms_executed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        }

    def find_path(
        self,
        source: str,
        target: str,
        max_hops: int = 3
    ) -> Optional[List[TransformNode]]:
        """
        Find a path from source to target domain using available transforms.

        Uses breadth-first search to find shortest path.

        Args:
            source: Source domain name
            target: Target domain name
            max_hops: Maximum number of intermediate transforms

        Returns:
            List of TransformNode objects representing the path,
            or None if no path exists
        """
        if source == target:
            return []

        # Check for direct transform
        if CrossDomainRegistry.has_transform(source, target):
            transform_class = CrossDomainRegistry.get(source, target)
            return [TransformNode(source, target, transform_class, {})]

        # BFS to find shortest path
        from collections import deque

        queue = deque([(source, [])])
        visited = {source}

        while queue:
            current_domain, path = queue.popleft()

            if len(path) >= max_hops:
                continue

            # Explore all outgoing transforms from current domain
            for src, tgt in CrossDomainRegistry.list_all():
                if src == current_domain and tgt not in visited:
                    transform_class = CrossDomainRegistry.get(src, tgt)
                    new_path = path + [TransformNode(src, tgt, transform_class, {})]

                    if tgt == target:
                        return new_path

                    visited.add(tgt)
                    queue.append((tgt, new_path))

        return None  # No path found

    def compose_path(
        self,
        source: str,
        target: str,
        via: Optional[List[str]] = None
    ) -> 'TransformPipeline':
        """
        Create a transform pipeline from source to target.

        Args:
            source: Source domain name
            target: Target domain name
            via: Optional list of intermediate domains to route through

        Returns:
            TransformPipeline object

        Raises:
            ValueError: If no path can be found
        """
        if via:
            # Explicit routing: build path through specified domains
            nodes = []
            domains = [source] + via + [target]

            for i in range(len(domains) - 1):
                src, tgt = domains[i], domains[i + 1]

                if not CrossDomainRegistry.has_transform(src, tgt):
                    raise ValueError(
                        f"No transform registered for {src} → {tgt}. "
                        f"Cannot route through specified path."
                    )

                transform_class = CrossDomainRegistry.get(src, tgt)
                nodes.append(TransformNode(src, tgt, transform_class, {}))

            return TransformPipeline(nodes, self)

        else:
            # Automatic path finding
            path = self.find_path(source, target)

            if path is None:
                raise ValueError(
                    f"No transform path found from {source} to {target}. "
                    f"Available transforms: {CrossDomainRegistry.list_all()}"
                )

            return TransformPipeline(path, self)

    def clear_cache(self):
        """Clear the transform result cache."""
        self._cache.clear()
        self._stats['cache_hits'] = 0
        self._stats['cache_misses'] = 0

    def get_stats(self) -> Dict[str, int]:
        """Get composition statistics."""
        return self._stats.copy()


class TransformPipeline:
    """
    Executable pipeline of cross-domain transforms.

    Represents a sequence of transforms: A → B → C → D
    Supports execution, visualization, and optimization.
    """

    def __init__(self, nodes: List[TransformNode], composer: TransformComposer):
        """
        Initialize pipeline.

        Args:
            nodes: List of TransformNode objects
            composer: Parent composer for caching
        """
        self.nodes = nodes
        self.composer = composer
        self._instances: List[DomainInterface] = []

    @property
    def source_domain(self) -> str:
        """Get source domain of pipeline."""
        return self.nodes[0].source_domain if self.nodes else None

    @property
    def target_domain(self) -> str:
        """Get target domain of pipeline."""
        return self.nodes[-1].target_domain if self.nodes else None

    @property
    def length(self) -> int:
        """Get number of transforms in pipeline."""
        return len(self.nodes)

    def __call__(self, source_data: Any, **kwargs) -> Any:
        """
        Execute the transform pipeline.

        Args:
            source_data: Input data in source domain format
            **kwargs: Parameters passed to transform instances

        Returns:
            Transformed data in target domain format
        """
        if not self.nodes:
            return source_data

        # Execute transforms sequentially
        current_data = source_data

        for node in self.nodes:
            # Create transform instance (or reuse if already created)
            transform = node.transform_class(source_data=current_data, **kwargs)

            # Execute transform
            current_data = transform.transform(current_data)

            self.composer._stats['transforms_executed'] += 1

        return current_data

    def visualize(self) -> str:
        """
        Create text visualization of pipeline.

        Returns:
            String representation of pipeline
        """
        if not self.nodes:
            return "Empty pipeline"

        arrow = " → "
        domains = [self.nodes[0].source_domain]

        for node in self.nodes:
            domains.append(node.target_domain)

        return arrow.join(domains)

    def __repr__(self):
        return f"TransformPipeline({self.visualize()})"


class BatchTransformComposer(TransformComposer):
    """
    Transform composer optimized for batch processing.

    Processes multiple inputs through the same pipeline efficiently.
    """

    def batch_transform(
        self,
        pipeline: TransformPipeline,
        inputs: List[Any]
    ) -> List[Any]:
        """
        Process multiple inputs through a pipeline.

        Args:
            pipeline: TransformPipeline to execute
            inputs: List of input data

        Returns:
            List of transformed outputs
        """
        results = []

        for input_data in inputs:
            result = pipeline(input_data)
            results.append(result)

        return results


def compose(*transforms: DomainInterface, validate: bool = True) -> Callable:
    """
    Compose multiple transforms into a single function.

    Usage:
        # Define transforms
        field_to_agent = FieldToAgentInterface(field, positions)
        agent_to_audio = AgentToAudioInterface(...)

        # Compose into pipeline
        pipeline = compose(field_to_agent, agent_to_audio)

        # Execute
        result = pipeline(field_data)

    Args:
        *transforms: Variable number of DomainInterface instances
        validate: If True, validate compatibility between transforms

    Returns:
        Composed function that executes all transforms in sequence
    """
    if not transforms:
        return lambda x: x

    if validate:
        # Check that target of transform N matches source of transform N+1
        for i in range(len(transforms) - 1):
            t1, t2 = transforms[i], transforms[i + 1]

            if t1.target_domain != t2.source_domain:
                raise ValueError(
                    f"Transform {i} outputs {t1.target_domain} but "
                    f"transform {i+1} expects {t2.source_domain}"
                )

    def composed_transform(data: Any) -> Any:
        """Execute composed transforms."""
        current_data = data

        for transform in transforms:
            current_data = transform.transform(current_data)

        return current_data

    return composed_transform


# Singleton composer instance
_default_composer = TransformComposer()


def find_transform_path(source: str, target: str, max_hops: int = 3) -> Optional[List[str]]:
    """
    Find a transform path from source to target domain.

    Convenience function using default composer.

    Args:
        source: Source domain name
        target: Target domain name
        max_hops: Maximum number of intermediate transforms

    Returns:
        List of domain names representing the path, or None if no path exists
    """
    path = _default_composer.find_path(source, target, max_hops)

    if path is None:
        return None

    # Extract domain names
    domain_path = [path[0].source_domain]
    for node in path:
        domain_path.append(node.target_domain)

    return domain_path


def auto_compose(source: str, target: str, **kwargs) -> TransformPipeline:
    """
    Automatically find and create a transform pipeline.

    Convenience function using default composer.

    Args:
        source: Source domain name
        target: Target domain name
        **kwargs: Additional parameters

    Returns:
        TransformPipeline ready for execution
    """
    return _default_composer.compose_path(source, target, **kwargs)
