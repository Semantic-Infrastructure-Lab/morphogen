"""Agent operations implementation using NumPy backend.

This module provides NumPy-based implementations of all core agent operations
for sparse particle/agent-based modeling, including allocation, mapping, filtering,
force calculations, and field-agent coupling.
"""

from typing import Callable, Optional, Dict, Any, Tuple, Union
import numpy as np

from morphogen.core.operator import operator, OpCategory


class Agents:
    """Sparse agent collection with per-agent properties.

    Represents a collection of agents where each agent has multiple properties
    (position, velocity, mass, etc.) stored as separate NumPy arrays for efficient
    vectorized operations.

    Example:
        agents = Agents(
            count=1000,
            properties={
                'pos': np.random.rand(1000, 2),  # 2D positions
                'vel': np.zeros((1000, 2)),       # 2D velocities
                'mass': np.ones(1000)             # Scalar masses
            }
        )
    """

    def __init__(self, count: int, properties: Dict[str, np.ndarray]):
        """Initialize agent collection.

        Args:
            count: Number of agents
            properties: Dictionary mapping property names to NumPy arrays
                       Each array's first dimension must equal count
        """
        self.count = count
        self.properties = properties
        self.alive_mask = np.ones(count, dtype=bool)

        # Validate properties
        for name, arr in properties.items():
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"Property '{name}' must be a NumPy array")
            if arr.shape[0] != count:
                raise ValueError(
                    f"Property '{name}' has {arr.shape[0]} elements, expected {count}"
                )

    @property
    def alive_count(self) -> int:
        """Get number of currently alive agents."""
        return np.sum(self.alive_mask)

    def get(self, property_name: str) -> np.ndarray:
        """Get property array for all alive agents.

        Args:
            property_name: Name of property to retrieve

        Returns:
            NumPy array of property values for alive agents only

        Raises:
            KeyError: If property doesn't exist
        """
        if property_name not in self.properties:
            raise KeyError(f"Unknown property: {property_name}")
        return self.properties[property_name][self.alive_mask]

    def get_all(self, property_name: str) -> np.ndarray:
        """Get property array for ALL agents (including dead ones).

        Args:
            property_name: Name of property to retrieve

        Returns:
            NumPy array of property values for all agents
        """
        if property_name not in self.properties:
            raise KeyError(f"Unknown property: {property_name}")
        return self.properties[property_name]

    def set(self, property_name: str, values: np.ndarray) -> 'Agents':
        """Set property array for alive agents.

        Args:
            property_name: Name of property to set
            values: New values (length must match alive_count)

        Returns:
            self (for chaining)

        Raises:
            KeyError: If property doesn't exist
            ValueError: If values shape doesn't match alive agents
        """
        if property_name not in self.properties:
            raise KeyError(f"Unknown property: {property_name}")

        alive_indices = np.where(self.alive_mask)[0]
        if len(values) != len(alive_indices):
            raise ValueError(
                f"Expected {len(alive_indices)} values, got {len(values)}"
            )

        self.properties[property_name][self.alive_mask] = values
        return self

    def update(self, property_name: str, values: np.ndarray) -> 'Agents':
        """Update property for alive agents (alias for set, matches Morphogen syntax).

        Args:
            property_name: Name of property to update
            values: New values

        Returns:
            New Agents instance with updated property
        """
        # Create a copy to maintain immutability (like Field2D.copy())
        new_agents = self.copy()
        new_agents.set(property_name, values)
        return new_agents

    def copy(self) -> 'Agents':
        """Create a deep copy of this agent collection.

        Returns:
            New Agents instance with copied data
        """
        return Agents(
            count=self.count,
            properties={name: arr.copy() for name, arr in self.properties.items()}
        )

    def __repr__(self) -> str:
        """String representation of agents."""
        props = ', '.join(self.properties.keys())
        return f"Agents(count={self.alive_count}/{self.count}, properties=[{props}])"


class AgentOperations:
    """Namespace for agent operations (accessed as 'agents' in DSL)."""

    @staticmethod
    @operator(
        domain="agents",
        category=OpCategory.CONSTRUCT,
        signature="(count: int, properties: Dict[str, Any]) -> Agents",
        deterministic=True,
        doc="Allocate a new agent collection"
    )
    def alloc(count: int, properties: Dict[str, Any], **kwargs) -> Agents:
        """Allocate a new agent collection.

        Args:
            count: Number of agents to allocate
            properties: Dictionary mapping property names to initial values
                       Values can be:
                       - NumPy arrays (shape[0] must equal count)
                       - Scalars (broadcast to all agents)
                       - Field2D objects (will be converted to arrays)

        Returns:
            New Agents instance

        Example:
            agents = agents.alloc(
                count=100,
                properties={
                    'pos': np.random.rand(100, 2),
                    'vel': np.zeros((100, 2)),
                    'mass': 1.0  # Broadcast to all agents
                }
            )
        """
        processed_props = {}

        for name, value in properties.items():
            if isinstance(value, np.ndarray):
                # Already an array
                if value.shape[0] != count:
                    raise ValueError(
                        f"Property '{name}' array has {value.shape[0]} elements, "
                        f"expected {count}"
                    )
                processed_props[name] = value.copy()

            elif np.isscalar(value):
                # Broadcast scalar to all agents
                processed_props[name] = np.full(count, value, dtype=np.float32)

            else:
                # Try to convert to array
                try:
                    arr = np.array(value, dtype=np.float32)
                    if arr.shape[0] != count:
                        # Broadcast if needed
                        arr = np.full(count, arr, dtype=np.float32)
                    processed_props[name] = arr
                except Exception as e:
                    raise TypeError(
                        f"Cannot convert property '{name}' to array: {e}"
                    )

        return Agents(count=count, properties=processed_props)

    @staticmethod
    @operator(
        domain="agents",
        category=OpCategory.TRANSFORM,
        signature="(agents_obj: Agents, property_name: str, func: Callable) -> ndarray",
        deterministic=True,
        doc="Apply function to each agent's property"
    )
    def map(agents_obj: Agents, property_name: str, func: Callable) -> np.ndarray:
        """Apply function to each agent's property.

        Args:
            agents_obj: Agents collection
            property_name: Property to map over
            func: Function to apply element-wise

        Returns:
            Array of mapped values

        Example:
            # Move all agents right by 1.0
            new_pos = agents.map(agents_obj, 'pos', lambda p: p + np.array([1.0, 0.0]))
        """
        values = agents_obj.get(property_name)

        # For vectorized operations
        if callable(func):
            try:
                # Try vectorized operation first
                result = func(values)
                return result
            except Exception:
                # Fall back to element-wise if vectorization fails
                return np.array([func(v) for v in values])
        else:
            raise TypeError(f"Expected callable, got {type(func)}")

    @staticmethod
    @operator(
        domain="agents",
        category=OpCategory.TRANSFORM,
        signature="(agents_obj: Agents, property_name: str, condition: Callable) -> Agents",
        deterministic=True,
        doc="Keep only agents matching condition"
    )
    def filter(agents_obj: Agents, property_name: str, condition: Callable) -> Agents:
        """Keep only agents matching condition.

        Args:
            agents_obj: Agents collection
            property_name: Property to test
            condition: Function that returns bool for each value

        Returns:
            New Agents instance with filtered alive_mask

        Example:
            # Keep only agents with positive x position
            filtered = agents.filter(agents_obj, 'pos', lambda p: p[0] > 0.0)
        """
        values = agents_obj.get_all(property_name)

        # Apply condition to get mask
        try:
            # Try vectorized first
            mask = condition(values)
        except Exception:
            # Fall back to element-wise
            mask = np.array([condition(v) for v in values], dtype=bool)

        # Create new agents with updated mask
        new_agents = agents_obj.copy()
        new_agents.alive_mask = agents_obj.alive_mask & mask
        return new_agents

    @staticmethod
    @operator(
        domain="agents",
        category=OpCategory.QUERY,
        signature="(agents_obj: Agents, property_name: str, operation: str, initial: Optional[Any]) -> Any",
        deterministic=True,
        doc="Reduce agents to single value"
    )
    def reduce(agents_obj: Agents, property_name: str,
               operation: str = "sum", initial: Optional[Any] = None) -> Any:
        """Reduce agents to single value.

        Args:
            agents_obj: Agents collection
            property_name: Property to reduce
            operation: Reduction operation ("sum", "mean", "min", "max", "prod")
            initial: Initial value (not used for built-in operations)

        Returns:
            Reduced value

        Example:
            # Total mass of all agents
            total_mass = agents.reduce(agents_obj, 'mass', operation='sum')
        """
        values = agents_obj.get(property_name)

        if operation == "sum":
            return np.sum(values)
        elif operation == "mean":
            return np.mean(values)
        elif operation == "min":
            return np.min(values)
        elif operation == "max":
            return np.max(values)
        elif operation == "prod":
            return np.prod(values)
        else:
            raise ValueError(f"Unknown reduction operation: {operation}")

    @staticmethod
    @operator(
        domain="agents",
        category=OpCategory.QUERY,
        signature="(agents_obj: Agents, radius: float, force_func: Callable, position_property: str, mass_property: Optional[str], use_spatial_hashing: bool) -> ndarray",
        deterministic=True,
        doc="Compute forces between nearby agents"
    )
    def compute_pairwise_forces(
        agents_obj: Agents,
        radius: float,
        force_func: Callable,
        position_property: str = 'pos',
        mass_property: Optional[str] = None,
        use_spatial_hashing: bool = True
    ) -> np.ndarray:
        """Compute forces between nearby agents.

        Uses spatial hashing for O(n) performance when use_spatial_hashing=True,
        otherwise falls back to O(n²) brute force.

        Args:
            agents_obj: Agents collection
            radius: Interaction radius
            force_func: Function(pos_i, pos_j, [mass_i, mass_j]) -> force_vector
            position_property: Name of position property
            mass_property: Optional name of mass property
            use_spatial_hashing: Whether to use spatial hashing optimization

        Returns:
            Array of force vectors for each agent

        Example:
            # Gravitational forces
            forces = agents.compute_pairwise_forces(
                agents_obj,
                radius=100.0,
                force_func=lambda pi, pj, mi, mj: compute_gravity(pi, pj, mi, mj)
            )
        """
        positions = agents_obj.get(position_property)
        n_agents = len(positions)

        # Determine dimensionality
        if len(positions.shape) == 1:
            dim = 1
            forces = np.zeros(n_agents, dtype=np.float32)
        else:
            dim = positions.shape[1]
            forces = np.zeros((n_agents, dim), dtype=np.float32)

        # Get masses if provided
        masses = None
        if mass_property is not None:
            masses = agents_obj.get(mass_property)

        if use_spatial_hashing and dim >= 2:
            # Use spatial hashing for 2D/3D
            forces = AgentOperations._pairwise_forces_spatial_hash(
                positions, radius, force_func, masses, dim
            )
        else:
            # Brute force O(n²) for small counts or 1D
            forces = AgentOperations._pairwise_forces_brute(
                positions, radius, force_func, masses, dim
            )

        return forces

    @staticmethod
    def _pairwise_forces_brute(
        positions: np.ndarray,
        radius: float,
        force_func: Callable,
        masses: Optional[np.ndarray],
        dim: int
    ) -> np.ndarray:
        """Brute force O(n²) pairwise force calculation."""
        n_agents = len(positions)
        forces = np.zeros_like(positions, dtype=np.float32)

        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                # Compute distance
                if dim == 1:
                    delta = positions[j] - positions[i]
                    dist = abs(delta)
                else:
                    delta = positions[j] - positions[i]
                    dist = np.linalg.norm(delta)

                # Skip if too far
                if dist > radius:
                    continue

                # Compute force
                if masses is not None:
                    force = force_func(positions[i], positions[j], masses[i], masses[j])
                else:
                    force = force_func(positions[i], positions[j])

                # Newton's third law
                forces[i] += force
                forces[j] -= force

        return forces

    @staticmethod
    def _pairwise_forces_spatial_hash(
        positions: np.ndarray,
        radius: float,
        force_func: Callable,
        masses: Optional[np.ndarray],
        dim: int
    ) -> np.ndarray:
        """Spatial hashing O(n) pairwise force calculation."""
        n_agents = len(positions)
        forces = np.zeros_like(positions, dtype=np.float32)

        # Build spatial hash grid
        cell_size = radius
        grid = {}

        for i, pos in enumerate(positions):
            # Compute cell coordinates
            if dim == 2:
                cell = (int(pos[0] / cell_size), int(pos[1] / cell_size))
            else:  # dim == 3
                cell = (int(pos[0] / cell_size), int(pos[1] / cell_size),
                       int(pos[2] / cell_size))

            if cell not in grid:
                grid[cell] = []
            grid[cell].append(i)

        # For each agent, check neighboring cells
        neighbor_offsets = []
        if dim == 2:
            neighbor_offsets = [
                (dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]
            ]
        else:  # dim == 3
            neighbor_offsets = [
                (dx, dy, dz)
                for dx in [-1, 0, 1]
                for dy in [-1, 0, 1]
                for dz in [-1, 0, 1]
            ]

        for i, pos_i in enumerate(positions):
            # Get agent's cell
            if dim == 2:
                cell = (int(pos_i[0] / cell_size), int(pos_i[1] / cell_size))
            else:
                cell = (int(pos_i[0] / cell_size), int(pos_i[1] / cell_size),
                       int(pos_i[2] / cell_size))

            # Check neighboring cells
            for offset in neighbor_offsets:
                neighbor_cell = tuple(c + o for c, o in zip(cell, offset))

                if neighbor_cell not in grid:
                    continue

                for j in grid[neighbor_cell]:
                    if j <= i:  # Skip self and already-processed pairs
                        continue

                    # Compute distance
                    delta = positions[j] - pos_i
                    dist = np.linalg.norm(delta)

                    if dist > radius:
                        continue

                    # Compute force
                    if masses is not None:
                        force = force_func(pos_i, positions[j], masses[i], masses[j])
                    else:
                        force = force_func(pos_i, positions[j])

                    # Newton's third law
                    forces[i] += force
                    forces[j] -= force

        return forces

    @staticmethod
    @operator(
        domain="agents",
        category=OpCategory.QUERY,
        signature="(agents_obj: Agents, field: Field2D, position_property: str) -> ndarray",
        deterministic=True,
        doc="Sample field values at agent positions"
    )
    def sample_field(agents_obj: Agents, field, position_property: str = 'pos') -> np.ndarray:
        """Sample field values at agent positions.

        Uses bilinear interpolation for 2D fields.

        Args:
            agents_obj: Agents collection
            field: Field2D object to sample from
            position_property: Name of position property

        Returns:
            Array of sampled field values at each agent position

        Example:
            # Sample temperature at each agent
            temps = agents.sample_field(agents_obj, temperature_field, 'pos')
        """
        from .field import Field2D

        if not isinstance(field, Field2D):
            raise TypeError(f"Expected Field2D, got {type(field)}")

        positions = agents_obj.get(position_property)

        # Positions are in world coordinates, need to map to grid coordinates
        # Assume field covers [0, width) x [0, height) in world coords
        h, w = field.shape

        # Clamp positions to field bounds
        x = np.clip(positions[:, 0], 0, w - 1)
        y = np.clip(positions[:, 1], 0, h - 1)

        # Bilinear interpolation
        x0 = np.floor(x).astype(int)
        x1 = np.minimum(x0 + 1, w - 1)
        y0 = np.floor(y).astype(int)
        y1 = np.minimum(y0 + 1, h - 1)

        fx = x - x0
        fy = y - y0

        # Sample field at corners
        if len(field.data.shape) == 2:
            # Scalar field
            sampled = (
                field.data[y0, x0] * (1 - fx) * (1 - fy) +
                field.data[y0, x1] * fx * (1 - fy) +
                field.data[y1, x0] * (1 - fx) * fy +
                field.data[y1, x1] * fx * fy
            )
        else:
            # Vector field - sample each channel
            n_channels = field.data.shape[2]
            sampled = np.zeros((len(positions), n_channels), dtype=np.float32)

            for c in range(n_channels):
                sampled[:, c] = (
                    field.data[y0, x0, c] * (1 - fx) * (1 - fy) +
                    field.data[y0, x1, c] * fx * (1 - fy) +
                    field.data[y1, x0, c] * (1 - fx) * fy +
                    field.data[y1, x1, c] * fx * fy
                )

        return sampled

    # ========================================================================
    # PARTICLE EFFECTS / VFX EXTENSIONS (v0.9.0)
    # ========================================================================

    @staticmethod
    @operator(
        domain="agents",
        category=OpCategory.CONSTRUCT,
        signature="(count: int, position: Union[ndarray, Callable], velocity: Optional[Union[ndarray, Callable]], lifetime: Optional[Union[float, Tuple[float, float]]], properties: Optional[Dict[str, Any]], emission_shape: str, emission_radius: float, seed: Optional[int]) -> Agents",
        deterministic=False,
        doc="Emit new particles from a source"
    )
    def emit(count: int, position: Union[np.ndarray, Callable],
             velocity: Optional[Union[np.ndarray, Callable]] = None,
             lifetime: Optional[Union[float, Tuple[float, float]]] = None,
             properties: Optional[Dict[str, Any]] = None,
             emission_shape: str = "point",
             emission_radius: float = 0.0,
             seed: Optional[int] = None) -> Agents:
        """Emit new particles from a source.

        Args:
            count: Number of particles to emit
            position: Emission position (array or callable that returns array)
                     - If callable: called to generate positions
                     - If array: shape (2,) for single point or (count, 2) for multiple
            velocity: Initial velocity (optional)
                     - If callable: called to generate velocities
                     - If array: shape (2,) broadcast to all, or (count, 2) for each
                     - If None: random velocities in emission_shape pattern
            lifetime: Particle lifetime in simulation steps
                     - If float: all particles have same lifetime
                     - If tuple (min, max): random lifetime in range
                     - If None: particles live forever
            properties: Additional properties for particles
            emission_shape: Shape of emission ("point", "circle", "sphere", "cone")
            emission_radius: Radius for shaped emissions
            seed: Random seed for reproducibility

        Returns:
            New Agents instance with emitted particles

        Example:
            # Emit from point
            particles = agents.emit(
                count=100,
                position=np.array([64.0, 64.0]),
                velocity=lambda n: np.random.randn(n, 2) * 5.0,
                lifetime=(10.0, 50.0),
                properties={'color': (1.0, 0.5, 0.0)}
            )

            # Emit in circle pattern
            particles = agents.emit(
                count=50,
                position=np.array([32.0, 32.0]),
                emission_shape="circle",
                emission_radius=10.0,
                lifetime=100.0
            )
        """
        rng = np.random.RandomState(seed)

        # Generate positions
        if callable(position):
            positions = position(count)
            if positions.shape != (count, 2):
                raise ValueError(f"Position callable must return (count, 2) array, got {positions.shape}")
        elif isinstance(position, np.ndarray):
            if position.shape == (2,):
                # Single point - apply emission shape
                if emission_shape == "point":
                    positions = np.tile(position, (count, 1))
                elif emission_shape == "circle":
                    # Random angles
                    angles = rng.rand(count) * 2 * np.pi
                    radii = rng.rand(count) * emission_radius
                    offsets = np.stack([
                        radii * np.cos(angles),
                        radii * np.sin(angles)
                    ], axis=1)
                    positions = position + offsets
                elif emission_shape == "sphere":
                    # Uniform sphere sampling
                    angles = rng.rand(count) * 2 * np.pi
                    radii = emission_radius * np.cbrt(rng.rand(count))
                    offsets = np.stack([
                        radii * np.cos(angles),
                        radii * np.sin(angles)
                    ], axis=1)
                    positions = position + offsets
                elif emission_shape == "cone":
                    # Cone emission (upward)
                    angles = rng.randn(count) * 0.3  # Spread
                    speeds = rng.rand(count) * emission_radius
                    offsets = np.stack([
                        speeds * np.sin(angles),
                        speeds * np.cos(angles)
                    ], axis=1)
                    positions = position + offsets
                else:
                    raise ValueError(f"Unknown emission_shape: {emission_shape}")
            elif position.shape == (count, 2):
                positions = position.copy()
            else:
                raise ValueError(f"Position must be (2,) or (count, 2), got {position.shape}")
        else:
            raise TypeError(f"Position must be array or callable, got {type(position)}")

        # Generate velocities
        if velocity is None:
            # Default velocities based on emission shape
            if emission_shape == "circle" or emission_shape == "sphere":
                # Radial outward
                centers = positions - (position if isinstance(position, np.ndarray) and position.shape == (2,) else 0)
                norms = np.linalg.norm(centers, axis=1, keepdims=True)
                norms = np.where(norms > 1e-6, norms, 1.0)  # Avoid division by zero
                velocities = (centers / norms) * emission_radius * 0.1
            elif emission_shape == "cone":
                # Upward cone
                angles = rng.randn(count) * 0.3
                speeds = emission_radius * 0.2
                velocities = np.stack([
                    speeds * np.sin(angles),
                    speeds * np.cos(angles)
                ], axis=1)
            else:
                # Random
                velocities = rng.randn(count, 2) * 0.5
        elif callable(velocity):
            velocities = velocity(count)
            if velocities.shape != (count, 2):
                raise ValueError(f"Velocity callable must return (count, 2) array, got {velocities.shape}")
        elif isinstance(velocity, np.ndarray):
            if velocity.shape == (2,):
                velocities = np.tile(velocity, (count, 1))
            elif velocity.shape == (count, 2):
                velocities = velocity.copy()
            else:
                raise ValueError(f"Velocity must be (2,) or (count, 2), got {velocity.shape}")
        else:
            raise TypeError(f"Velocity must be array or callable, got {type(velocity)}")

        # Generate lifetimes
        if lifetime is None:
            lifetimes = np.full(count, np.inf, dtype=np.float32)
        elif isinstance(lifetime, tuple):
            min_life, max_life = lifetime
            lifetimes = rng.uniform(min_life, max_life, count).astype(np.float32)
        else:
            lifetimes = np.full(count, float(lifetime), dtype=np.float32)

        # Build properties dict
        particle_props = {
            'pos': positions.astype(np.float32),
            'vel': velocities.astype(np.float32),
            'age': np.zeros(count, dtype=np.float32),
            'lifetime': lifetimes
        }

        # Add user properties
        if properties is not None:
            for key, value in properties.items():
                if isinstance(value, np.ndarray):
                    if value.shape[0] != count:
                        raise ValueError(f"Property '{key}' has wrong shape: {value.shape}")
                    particle_props[key] = value.copy()
                elif np.isscalar(value):
                    # Scalar value - broadcast to all particles
                    particle_props[key] = np.full(count, value, dtype=np.float32)
                elif isinstance(value, (tuple, list)):
                    # Vector value - replicate for all particles
                    value_array = np.array(value, dtype=np.float32)
                    particle_props[key] = np.tile(value_array, (count, 1)) if value_array.ndim == 1 else np.array([value_array] * count)
                else:
                    raise TypeError(f"Property '{key}' must be array or scalar, got {type(value)}")

        return AgentOperations.alloc(count=count, properties=particle_props)

    @staticmethod
    @operator(
        domain="agents",
        category=OpCategory.TRANSFORM,
        signature="(agents_obj: Agents, dt: float, age_property: str, lifetime_property: str) -> Agents",
        deterministic=True,
        doc="Age particles and remove dead ones"
    )
    def age_particles(agents_obj: Agents, dt: float = 1.0,
                     age_property: str = 'age',
                     lifetime_property: str = 'lifetime') -> Agents:
        """Age particles and remove dead ones.

        Args:
            agents_obj: Agents collection
            dt: Time step to age by
            age_property: Name of age property
            lifetime_property: Name of lifetime property

        Returns:
            New Agents instance with updated ages and dead particles filtered

        Example:
            # Age particles by 1 step
            particles = agents.age_particles(particles, dt=1.0)
        """
        if age_property not in agents_obj.properties:
            raise KeyError(f"Agents missing '{age_property}' property")
        if lifetime_property not in agents_obj.properties:
            raise KeyError(f"Agents missing '{lifetime_property}' property")

        # Create copy
        new_agents = agents_obj.copy()

        # Increment age
        new_agents.properties[age_property] += dt

        # Filter dead particles
        ages = new_agents.get_all(age_property)
        lifetimes = new_agents.get_all(lifetime_property)
        alive = ages < lifetimes

        new_agents.alive_mask = new_agents.alive_mask & alive

        return new_agents

    @staticmethod
    @operator(
        domain="agents",
        category=OpCategory.QUERY,
        signature="(agents_obj: Agents, age_property: str, lifetime_property: str, fade_in: float, fade_out: float) -> ndarray",
        deterministic=True,
        doc="Calculate alpha transparency for particles based on age"
    )
    def get_particle_alpha(agents_obj: Agents,
                          age_property: str = 'age',
                          lifetime_property: str = 'lifetime',
                          fade_in: float = 0.0,
                          fade_out: float = 0.2) -> np.ndarray:
        """Calculate alpha transparency for particles based on age.

        Args:
            agents_obj: Agents collection
            age_property: Name of age property
            lifetime_property: Name of lifetime property
            fade_in: Fraction of lifetime to fade in (0.0 to 1.0)
            fade_out: Fraction of lifetime to fade out (0.0 to 1.0)

        Returns:
            Array of alpha values in [0, 1] for each particle

        Example:
            # Fade out last 20% of lifetime
            alphas = agents.get_particle_alpha(particles, fade_out=0.2)
        """
        ages = agents_obj.get(age_property)
        lifetimes = agents_obj.get(lifetime_property)

        # Normalize age to [0, 1]
        age_norm = np.clip(ages / np.maximum(lifetimes, 1e-6), 0.0, 1.0)

        alpha = np.ones(len(ages), dtype=np.float32)

        # Fade in
        if fade_in > 0:
            fade_in_mask = age_norm < fade_in
            alpha[fade_in_mask] = age_norm[fade_in_mask] / fade_in

        # Fade out
        if fade_out > 0:
            fade_out_start = 1.0 - fade_out
            fade_out_mask = age_norm > fade_out_start
            alpha[fade_out_mask] = (1.0 - age_norm[fade_out_mask]) / fade_out

        return alpha

    @staticmethod
    @operator(
        domain="agents",
        category=OpCategory.TRANSFORM,
        signature="(agents_obj: Agents, force: Union[ndarray, Callable], velocity_property: str, mass_property: Optional[str], dt: float) -> Agents",
        deterministic=True,
        doc="Apply force to particles (F = ma)"
    )
    def apply_force(agents_obj: Agents, force: Union[np.ndarray, Callable],
                   velocity_property: str = 'vel',
                   mass_property: Optional[str] = None,
                   dt: float = 1.0) -> Agents:
        """Apply force to particles (F = ma).

        Args:
            agents_obj: Agents collection
            force: Force to apply
                  - If array: shape (2,) for uniform force or (count, 2) for per-particle
                  - If callable: func(agents) -> forces
            velocity_property: Name of velocity property
            mass_property: Name of mass property (optional, default mass=1)
            dt: Time step

        Returns:
            New Agents instance with updated velocities

        Example:
            # Apply gravity
            particles = agents.apply_force(
                particles,
                force=np.array([0.0, -9.8]),
                dt=0.1
            )

            # Apply drag
            particles = agents.apply_force(
                particles,
                force=lambda a: -0.1 * a.get('vel'),
                dt=0.1
            )
        """
        # Get current velocity
        velocities = agents_obj.get(velocity_property)

        # Get mass
        if mass_property is not None:
            masses = agents_obj.get(mass_property)
        else:
            masses = np.ones(len(velocities), dtype=np.float32)

        # Calculate force
        if callable(force):
            forces = force(agents_obj)
        elif isinstance(force, np.ndarray):
            if force.shape == (2,):
                forces = np.tile(force, (len(velocities), 1))
            elif force.shape == (len(velocities), 2):
                forces = force
            else:
                raise ValueError(f"Force must be (2,) or (count, 2), got {force.shape}")
        else:
            raise TypeError(f"Force must be array or callable, got {type(force)}")

        # Apply F = ma -> dv = (F/m) * dt
        accelerations = forces / masses[:, np.newaxis]
        new_velocities = velocities + accelerations * dt

        return agents_obj.update(velocity_property, new_velocities)

    @staticmethod
    @operator(
        domain="agents",
        category=OpCategory.TRANSFORM,
        signature="(agents_obj: Agents, position_property: str, velocity_property: str, dt: float) -> Agents",
        deterministic=True,
        doc="Integrate particle positions using Euler integration"
    )
    def integrate(agents_obj: Agents,
                 position_property: str = 'pos',
                 velocity_property: str = 'vel',
                 dt: float = 1.0) -> Agents:
        """Integrate particle positions using Euler integration.

        Args:
            agents_obj: Agents collection
            position_property: Name of position property
            velocity_property: Name of velocity property
            dt: Time step

        Returns:
            New Agents instance with updated positions

        Example:
            # Update positions
            particles = agents.integrate(particles, dt=0.1)
        """
        positions = agents_obj.get(position_property)
        velocities = agents_obj.get(velocity_property)

        new_positions = positions + velocities * dt

        return agents_obj.update(position_property, new_positions)

    @staticmethod
    @operator(
        domain="agents",
        category=OpCategory.TRANSFORM,
        signature="(agents_obj: Agents, position_property: str, trail_length: int) -> Agents",
        deterministic=True,
        doc="Update particle trail history"
    )
    def update_trail(agents_obj: Agents,
                    position_property: str = 'pos',
                    trail_length: int = 10) -> Agents:
        """Update particle trail history.

        Creates or updates a 'trail_history' property that stores recent positions.

        Args:
            agents_obj: Agents collection
            position_property: Name of position property to record
            trail_length: Number of historical positions to keep

        Returns:
            New Agents instance with updated trail_history

        Example:
            # Update trail every frame
            particles = agents.update_trail(particles, trail_length=20)
        """
        positions = agents_obj.get_all(position_property)
        new_agents = agents_obj.copy()

        # Initialize trail_history if it doesn't exist
        if 'trail_history' not in new_agents.properties:
            # Create empty trail history filled with NaN
            trail_history = np.full((agents_obj.count, trail_length, 2), np.nan, dtype=np.float32)
            new_agents.properties['trail_history'] = trail_history

        # Shift trail history and add current position
        trail_history = new_agents.properties['trail_history']

        # Shift all positions back
        trail_history[:, :-1, :] = trail_history[:, 1:, :]

        # Add current position at end
        trail_history[:, -1, :] = positions

        return new_agents

    @staticmethod
    @operator(
        domain="agents",
        category=OpCategory.TRANSFORM,
        signature="(agents_list: list) -> Agents",
        deterministic=True,
        doc="Merge multiple agent collections into one"
    )
    def merge(agents_list: list) -> Agents:
        """Merge multiple agent collections into one.

        Useful for combining newly emitted particles with existing ones.

        Args:
            agents_list: List of Agents instances to merge

        Returns:
            New Agents instance with all agents combined

        Example:
            # Emit new particles and merge with existing
            new_particles = agents.emit(count=50, position=np.array([64, 64]))
            all_particles = agents.merge([existing_particles, new_particles])
        """
        if not agents_list:
            raise ValueError("agents_list cannot be empty")

        if len(agents_list) == 1:
            return agents_list[0].copy()

        # Get all property names (union of all properties)
        all_properties = set()
        for a in agents_list:
            all_properties.update(a.properties.keys())

        # Calculate total count
        total_count = sum(a.count for a in agents_list)

        # Build merged properties
        merged_props = {}
        for prop_name in all_properties:
            # Determine shape from first agent that has this property
            sample_prop = None
            for a in agents_list:
                if prop_name in a.properties:
                    sample_prop = a.properties[prop_name]
                    break

            if sample_prop is None:
                continue

            # Determine shape
            if len(sample_prop.shape) == 1:
                merged_array = np.zeros(total_count, dtype=np.float32)
            else:
                merged_array = np.zeros((total_count, *sample_prop.shape[1:]), dtype=np.float32)

            # Fill in values
            offset = 0
            for a in agents_list:
                if prop_name in a.properties:
                    merged_array[offset:offset + a.count] = a.properties[prop_name]
                else:
                    # Fill with default value (0 for scalars, NaN for structured)
                    if len(merged_array.shape) > 1:
                        merged_array[offset:offset + a.count] = np.nan
                    else:
                        merged_array[offset:offset + a.count] = 0.0

                offset += a.count

            merged_props[prop_name] = merged_array

        merged = Agents(count=total_count, properties=merged_props)

        # Merge alive masks
        offset = 0
        for a in agents_list:
            merged.alive_mask[offset:offset + a.count] = a.alive_mask
            offset += a.count

        return merged


class ParticleBehaviors:
    """Pre-built particle behavior helpers."""

    @staticmethod
    def vortex(center: np.ndarray, strength: float = 1.0) -> Callable:
        """Create a vortex force field.

        Args:
            center: Center position of vortex (2D)
            strength: Strength of vortex rotation

        Returns:
            Force function for use with agents.apply_force

        Example:
            # Apply vortex centered at (64, 64)
            vortex_force = particle_behaviors.vortex(
                center=np.array([64.0, 64.0]),
                strength=5.0
            )
            particles = agents.apply_force(particles, force=vortex_force, dt=0.1)
        """
        def force_func(agents_obj):
            positions = agents_obj.get('pos')
            # Vector from center to particle
            delta = positions - center
            dist = np.linalg.norm(delta, axis=1, keepdims=True)

            # Avoid division by zero
            dist = np.maximum(dist, 1e-6)

            # Perpendicular vector (rotate 90 degrees)
            perp = np.stack([-delta[:, 1], delta[:, 0]], axis=1)

            # Vortex force (tangential, inversely proportional to distance)
            force = (perp / (dist ** 2)) * strength

            return force.astype(np.float32)

        return force_func

    @staticmethod
    def attractor(center: np.ndarray, strength: float = 1.0) -> Callable:
        """Create an attractor/gravity well.

        Args:
            center: Center position of attractor (2D)
            strength: Strength of attraction

        Returns:
            Force function for use with agents.apply_force

        Example:
            # Attract particles to center
            gravity = particle_behaviors.attractor(
                center=np.array([64.0, 64.0]),
                strength=10.0
            )
            particles = agents.apply_force(particles, force=gravity, dt=0.1)
        """
        def force_func(agents_obj):
            positions = agents_obj.get('pos')
            delta = center - positions
            dist = np.linalg.norm(delta, axis=1, keepdims=True)

            # Avoid division by zero
            dist = np.maximum(dist, 1e-6)

            # Gravitational force (toward center, inversely proportional to distance squared)
            force = (delta / (dist ** 2)) * strength

            return force.astype(np.float32)

        return force_func

    @staticmethod
    def repulsor(center: np.ndarray, strength: float = 1.0, radius: float = 50.0) -> Callable:
        """Create a repulsive force field.

        Args:
            center: Center position of repulsor (2D)
            strength: Strength of repulsion
            radius: Maximum radius of effect

        Returns:
            Force function for use with agents.apply_force

        Example:
            # Repel particles from center
            repulsion = particle_behaviors.repulsor(
                center=np.array([64.0, 64.0]),
                strength=20.0,
                radius=30.0
            )
            particles = agents.apply_force(particles, force=repulsion, dt=0.1)
        """
        def force_func(agents_obj):
            positions = agents_obj.get('pos')
            delta = positions - center
            dist = np.linalg.norm(delta, axis=1, keepdims=True)

            # Avoid division by zero
            dist = np.maximum(dist, 1e-6)

            # Repulsive force (away from center, stronger when closer)
            force = (delta / (dist ** 2)) * strength

            # Apply only within radius
            mask = (dist.flatten() < radius)
            force[~mask] = 0.0

            return force.astype(np.float32)

        return force_func

    @staticmethod
    def drag(coefficient: float = 0.1) -> Callable:
        """Create velocity-proportional drag force.

        Args:
            coefficient: Drag coefficient (higher = more drag)

        Returns:
            Force function for use with agents.apply_force

        Example:
            # Apply drag to slow particles down
            particles = agents.apply_force(
                particles,
                force=particle_behaviors.drag(coefficient=0.05),
                dt=0.1
            )
        """
        def force_func(agents_obj):
            velocities = agents_obj.get('vel')
            # Drag opposes velocity
            return -velocities * coefficient

        return force_func

    @staticmethod
    def turbulence(scale: float = 1.0, seed: Optional[int] = None) -> Callable:
        """Create random turbulence force.

        Args:
            scale: Scale of random forces
            seed: Random seed for reproducibility

        Returns:
            Force function for use with agents.apply_force

        Example:
            # Add random turbulence
            particles = agents.apply_force(
                particles,
                force=particle_behaviors.turbulence(scale=2.0, seed=42),
                dt=0.1
            )
        """
        rng = np.random.RandomState(seed)

        def force_func(agents_obj):
            count = agents_obj.alive_count
            return rng.randn(count, 2).astype(np.float32) * scale

        return force_func


# Create singleton instance for use as 'agents' namespace
agents = AgentOperations()

# Create singleton for particle behaviors
particle_behaviors = ParticleBehaviors()

# Export operators for domain registry discovery
alloc = AgentOperations.alloc
map = AgentOperations.map
filter = AgentOperations.filter
reduce = AgentOperations.reduce
compute_pairwise_forces = AgentOperations.compute_pairwise_forces
sample_field = AgentOperations.sample_field
emit = AgentOperations.emit
age_particles = AgentOperations.age_particles
get_particle_alpha = AgentOperations.get_particle_alpha
apply_force = AgentOperations.apply_force
integrate = AgentOperations.integrate
update_trail = AgentOperations.update_trail
merge = AgentOperations.merge
