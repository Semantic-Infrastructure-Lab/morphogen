"""Cross-domain path finding + explicit transform chaining.

**History (2026-07-11):** this module used to host a "composition engine"
(``TransformComposer`` / ``TransformPipeline`` / ``auto_compose``) that claimed to
*automatically* instantiate and run a chain of cross-domain transforms from just a
source and target domain name. It never worked: every built-in transform's
``__init__`` rejected the ``source_data=`` kwarg the engine passed, so it raised
``ValueError`` on all five advertised pairs, and the one example that used it hid
the failure in a ``try/except``. The composition-vs-glue experiment
(``docs/reviews/2026-07-11-composition-vs-glue-experiment.md``) showed the engine
solved no real problem even in principle — hand-glue was equal-or-fewer lines — so
it was **retired** (BACKLOG P0-2). For time-coupled, feedback co-simulation across
domains, use :mod:`morphogen.coupling` instead.

What remains here is the part that was always honest and useful:

- :func:`find_transform_path` — discover *whether* a chain of registered bridges
  connects two domains (BFS over the registry). Returns domain names; runs nothing.
- :func:`compose` — chain **explicitly constructed** ``DomainInterface`` instances
  into one callable. You build each transform (so it gets its real parameters),
  then compose their ``.transform()`` calls. No auto-instantiation, no magic.
"""

from collections import deque
from typing import Any, Callable, List, Optional

from .base import DomainInterface
from .registry import CrossDomainRegistry


def find_transform_path(
    source: str, target: str, max_hops: int = 3
) -> Optional[List[str]]:
    """Find a path of registered cross-domain bridges from ``source`` to ``target``.

    Breadth-first search over the registry graph. This only *discovers* whether a
    chain of registered transforms connects the two domains — it does not
    instantiate or execute anything.

    Args:
        source: Source domain name.
        target: Target domain name.
        max_hops: Maximum number of intermediate transforms.

    Returns:
        A list of domain names ``[source, ..., target]`` describing the shortest
        path, or ``None`` if no path exists within ``max_hops``.
    """
    if source == target:
        return [source]

    if CrossDomainRegistry.has_transform(source, target):
        return [source, target]

    queue = deque([(source, [source])])
    visited = {source}

    while queue:
        current, path = queue.popleft()
        if len(path) - 1 >= max_hops:
            continue
        for src, tgt in CrossDomainRegistry.list_all():
            if src == current and tgt not in visited:
                new_path = path + [tgt]
                if tgt == target:
                    return new_path
                visited.add(tgt)
                queue.append((tgt, new_path))

    return None  # No path found


def compose(*transforms: DomainInterface, validate: bool = True) -> Callable[[Any], Any]:
    """Chain explicitly-constructed transforms into a single callable.

    Unlike the retired auto-composer, this takes transforms you have **already
    built** (so each received its real configuration) and runs their ``.transform``
    calls in sequence.

    Usage::

        field_to_audio = FieldToAudioInterface(field, mapping=..., sample_rate=44100)
        audio_to_visual = AudioToVisualInterface(signal, sample_rate=44100)
        pipeline = compose(field_to_audio, audio_to_visual)
        result = pipeline(field_data)

    Args:
        *transforms: Constructed ``DomainInterface`` instances.
        validate: If True, check that transform N's ``target_domain`` matches
            transform N+1's ``source_domain``.

    Returns:
        A callable that runs all transforms in sequence.

    Raises:
        ValueError: if ``validate`` and adjacent transforms don't connect.
    """
    if not transforms:
        return lambda x: x

    if validate:
        for i in range(len(transforms) - 1):
            t1, t2 = transforms[i], transforms[i + 1]
            if t1.target_domain != t2.source_domain:
                raise ValueError(
                    f"Transform {i} outputs {t1.target_domain} but "
                    f"transform {i + 1} expects {t2.source_domain}"
                )

    def composed_transform(data: Any) -> Any:
        current = data
        for transform in transforms:
            current = transform.transform(current)
        return current

    return composed_transform
