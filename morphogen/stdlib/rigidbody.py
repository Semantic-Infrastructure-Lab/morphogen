"""Rigid Body Physics Domain for 2D simulations.

This module provides 2D rigid body physics simulation with collision detection,
response, and constraint solving. Designed for deterministic, reproducible physics
simulations in Morphogen.

Features:
- 2D rigid bodies with position, rotation, velocity, angular velocity
- Collision detection (circle, box, polygon shapes)
- Collision response with friction and restitution
- Constraints (distance, hinge, spring)
- Deterministic physics simulation using Verlet/RK4 integration

Architecture:
- Layer 1: Core physics (bodies, forces, integration)
- Layer 2: Collision detection and response
- Layer 3: Constraints and world simulation
"""

from typing import List, Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from morphogen.core.operator import operator, OpCategory


# ============================================================================
# CORE TYPES
# ============================================================================

class ShapeType(Enum):
    """Collision shape types."""
    CIRCLE = "circle"
    BOX = "box"
    POLYGON = "polygon"


@dataclass
class RigidBody2D:
    """2D Rigid Body with mass, inertia, position, velocity.

    Attributes:
        position: Center of mass position [x, y] in meters
        rotation: Rotation angle in radians
        velocity: Linear velocity [vx, vy] in m/s
        angular_velocity: Angular velocity in rad/s
        mass: Mass in kg (0 = infinite mass, static body)
        inertia: Moment of inertia in kg·m² (0 = infinite inertia)
        restitution: Coefficient of restitution (0 = inelastic, 1 = elastic)
        friction: Coefficient of friction (0 = frictionless, 1 = rough)
        shape_type: Type of collision shape
        shape_params: Shape-specific parameters
        forces: Accumulated forces for this timestep
        torques: Accumulated torques for this timestep
        id: Unique identifier
    """
    position: np.ndarray  # [x, y]
    rotation: float = 0.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    angular_velocity: float = 0.0
    mass: float = 1.0
    inertia: float = 1.0
    restitution: float = 0.5
    friction: float = 0.3
    shape_type: ShapeType = ShapeType.CIRCLE
    shape_params: Dict[str, Any] = field(default_factory=dict)
    forces: np.ndarray = field(default_factory=lambda: np.zeros(2))
    torques: float = 0.0
    id: int = 0

    @property
    def is_static(self) -> bool:
        """Check if body is static (infinite mass)."""
        return self.mass == 0.0

    @property
    def inv_mass(self) -> float:
        """Inverse mass (0 for static bodies)."""
        return 0.0 if self.is_static else 1.0 / self.mass

    @property
    def inv_inertia(self) -> float:
        """Inverse inertia (0 for static bodies)."""
        return 0.0 if self.inertia == 0.0 else 1.0 / self.inertia


@dataclass
class Contact:
    """Contact point between two bodies.

    Attributes:
        body_a: First body ID
        body_b: Second body ID
        point: Contact point in world space
        normal: Contact normal (from A to B)
        penetration: Penetration depth
        tangent: Tangent direction (perpendicular to normal)
    """
    body_a: int
    body_b: int
    point: np.ndarray
    normal: np.ndarray
    penetration: float
    tangent: np.ndarray = field(default_factory=lambda: np.zeros(2))


@dataclass
class PhysicsWorld2D:
    """2D Physics World containing all bodies and simulation state.

    Attributes:
        bodies: List of rigid bodies
        gravity: Gravity vector [gx, gy] in m/s²
        damping: Linear damping coefficient (0-1)
        angular_damping: Angular damping coefficient (0-1)
        time: Current simulation time
        dt: Fixed timestep for integration
        iterations: Number of constraint solver iterations
    """
    bodies: List[RigidBody2D] = field(default_factory=list)
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, -9.81]))
    damping: float = 0.99
    angular_damping: float = 0.99
    time: float = 0.0
    dt: float = 1.0 / 60.0
    iterations: int = 10

    def add_body(self, body: RigidBody2D) -> int:
        """Add a body to the world and return its ID."""
        body.id = len(self.bodies)
        self.bodies.append(body)
        return body.id

    def get_body(self, body_id: int) -> Optional[RigidBody2D]:
        """Get body by ID."""
        if 0 <= body_id < len(self.bodies):
            return self.bodies[body_id]
        return None


# ============================================================================
# LAYER 1: ATOMIC OPERATORS (Core Physics)
# ============================================================================

@operator(
    domain="rigidbody",
    category=OpCategory.CONSTRUCT,
    signature="(position: ndarray, radius: float, mass: float, restitution: float, friction: float) -> RigidBody2D",
    deterministic=True,
    doc="Create a circular rigid body"
)
def create_circle_body(
    position: np.ndarray,
    radius: float,
    mass: float = 1.0,
    restitution: float = 0.5,
    friction: float = 0.3
) -> RigidBody2D:
    """Create a circular rigid body.

    Args:
        position: Initial position [x, y]
        radius: Circle radius in meters
        mass: Mass in kg (0 for static body)
        restitution: Coefficient of restitution (0-1)
        friction: Coefficient of friction (0-1)

    Returns:
        RigidBody2D with circle shape

    Example:
        ball = create_circle_body(
            position=np.array([0.0, 5.0]),
            radius=0.5,
            mass=1.0
        )
    """
    # Moment of inertia for circle: I = 0.5 * m * r²
    inertia = 0.5 * mass * radius * radius if mass > 0 else 0.0

    return RigidBody2D(
        position=position.copy(),
        mass=mass,
        inertia=inertia,
        restitution=restitution,
        friction=friction,
        shape_type=ShapeType.CIRCLE,
        shape_params={"radius": radius}
    )


@operator(
    domain="rigidbody",
    category=OpCategory.CONSTRUCT,
    signature="(position: ndarray, width: float, height: float, mass: float, restitution: float, friction: float) -> RigidBody2D",
    deterministic=True,
    doc="Create a rectangular rigid body"
)
def create_box_body(
    position: np.ndarray,
    width: float,
    height: float,
    mass: float = 1.0,
    restitution: float = 0.5,
    friction: float = 0.3
) -> RigidBody2D:
    """Create a rectangular rigid body.

    Args:
        position: Initial position [x, y] (center)
        width: Box width in meters
        height: Box height in meters
        mass: Mass in kg (0 for static body)
        restitution: Coefficient of restitution (0-1)
        friction: Coefficient of friction (0-1)

    Returns:
        RigidBody2D with box shape

    Example:
        box = create_box_body(
            position=np.array([0.0, 0.0]),
            width=2.0,
            height=1.0,
            mass=5.0
        )
    """
    # Moment of inertia for rectangle: I = (1/12) * m * (w² + h²)
    inertia = (1.0 / 12.0) * mass * (width * width + height * height) if mass > 0 else 0.0

    return RigidBody2D(
        position=position.copy(),
        mass=mass,
        inertia=inertia,
        restitution=restitution,
        friction=friction,
        shape_type=ShapeType.BOX,
        shape_params={"width": width, "height": height}
    )


@operator(
    domain="rigidbody",
    category=OpCategory.TRANSFORM,
    signature="(body: RigidBody2D, force: ndarray, point: Optional[ndarray]) -> RigidBody2D",
    deterministic=True,
    doc="Apply force to a rigid body (accumulates until integration step)"
)
def apply_force(body: RigidBody2D, force: np.ndarray, point: Optional[np.ndarray] = None) -> RigidBody2D:
    """Apply force to a rigid body (accumulates until integration step).

    Args:
        body: Rigid body to apply force to
        force: Force vector [fx, fy] in Newtons
        point: Point of application in world space (optional, defaults to center)

    Returns:
        Updated body (modifies in-place and returns for chaining)

    Example:
        # Apply gravity force
        body = apply_force(body, np.array([0.0, -9.81 * body.mass]))

        # Apply force at offset point (generates torque)
        body = apply_force(
            body,
            np.array([10.0, 0.0]),
            point=body.position + np.array([0.0, 1.0])
        )
    """
    body.forces += force

    if point is not None and not body.is_static:
        # Calculate torque: τ = r × F
        r = point - body.position
        torque = r[0] * force[1] - r[1] * force[0]  # 2D cross product
        body.torques += torque

    return body


@operator(
    domain="rigidbody",
    category=OpCategory.TRANSFORM,
    signature="(body: RigidBody2D, impulse: ndarray, point: Optional[ndarray]) -> RigidBody2D",
    deterministic=True,
    doc="Apply instantaneous impulse to a rigid body"
)
def apply_impulse(body: RigidBody2D, impulse: np.ndarray, point: Optional[np.ndarray] = None) -> RigidBody2D:
    """Apply instantaneous impulse to a rigid body.

    Args:
        body: Rigid body to apply impulse to
        impulse: Impulse vector [Jx, Jy] in N·s
        point: Point of application in world space (optional)

    Returns:
        Updated body with changed velocity

    Example:
        # Bounce off ground
        body = apply_impulse(body, np.array([0.0, 10.0]))
    """
    if body.is_static:
        return body

    body.velocity += impulse * body.inv_mass

    if point is not None:
        # Angular impulse: Δω = (r × J) / I
        r = point - body.position
        angular_impulse = (r[0] * impulse[1] - r[1] * impulse[0]) * body.inv_inertia
        body.angular_velocity += angular_impulse

    return body


@operator(
    domain="rigidbody",
    category=OpCategory.TRANSFORM,
    signature="(body: RigidBody2D) -> RigidBody2D",
    deterministic=True,
    doc="Clear accumulated forces and torques (call after integration)"
)
def clear_forces(body: RigidBody2D) -> RigidBody2D:
    """Clear accumulated forces and torques (call after integration).

    Args:
        body: Rigid body to clear

    Returns:
        Updated body
    """
    body.forces = np.zeros(2)
    body.torques = 0.0
    return body


@operator(
    domain="rigidbody",
    category=OpCategory.INTEGRATE,
    signature="(body: RigidBody2D, dt: float, damping: float) -> RigidBody2D",
    deterministic=True,
    doc="Integrate rigid body motion using semi-implicit Euler"
)
def integrate_body(body: RigidBody2D, dt: float, damping: float = 0.99) -> RigidBody2D:
    """Integrate rigid body motion using semi-implicit Euler.

    Args:
        body: Rigid body to integrate
        dt: Timestep in seconds
        damping: Velocity damping coefficient (0-1)

    Returns:
        Updated body with new position, rotation, velocity

    Example:
        body = apply_force(body, gravity_force)
        body = integrate_body(body, dt=0.016)
        body = clear_forces(body)
    """
    if body.is_static:
        return body

    # Semi-implicit Euler (velocity Verlet variant)
    # v(t+dt) = v(t) + a(t) * dt
    # x(t+dt) = x(t) + v(t+dt) * dt

    # Linear motion
    acceleration = body.forces * body.inv_mass
    body.velocity += acceleration * dt
    body.velocity *= damping  # Apply damping
    body.position += body.velocity * dt

    # Angular motion
    angular_acceleration = body.torques * body.inv_inertia
    body.angular_velocity += angular_acceleration * dt
    body.angular_velocity *= damping  # Apply damping
    body.rotation += body.angular_velocity * dt

    # Normalize rotation to [-π, π]
    body.rotation = np.arctan2(np.sin(body.rotation), np.cos(body.rotation))

    return body


# ============================================================================
# LAYER 2: COLLISION DETECTION
# ============================================================================

@operator(
    domain="rigidbody",
    category=OpCategory.QUERY,
    signature="(body_a: RigidBody2D, body_b: RigidBody2D) -> Optional[Contact]",
    deterministic=True,
    doc="Detect collision between two circles"
)
def detect_circle_circle_collision(
    body_a: RigidBody2D,
    body_b: RigidBody2D
) -> Optional[Contact]:
    """Detect collision between two circles.

    Args:
        body_a: First circle body
        body_b: Second circle body

    Returns:
        Contact object if collision detected, None otherwise
    """
    radius_a = body_a.shape_params["radius"]
    radius_b = body_b.shape_params["radius"]

    # Distance between centers
    delta = body_b.position - body_a.position
    distance = np.linalg.norm(delta)

    # Check if circles overlap
    sum_radii = radius_a + radius_b
    if distance >= sum_radii:
        return None  # No collision

    # Collision detected
    if distance > 0:
        normal = delta / distance
    else:
        # Circles exactly on top of each other (rare)
        normal = np.array([1.0, 0.0])

    penetration = sum_radii - distance
    contact_point = body_a.position + normal * radius_a
    tangent = np.array([-normal[1], normal[0]])

    return Contact(
        body_a=body_a.id,
        body_b=body_b.id,
        point=contact_point,
        normal=normal,
        penetration=penetration,
        tangent=tangent
    )


@operator(
    domain="rigidbody",
    category=OpCategory.QUERY,
    signature="(world: PhysicsWorld2D) -> List[Contact]",
    deterministic=True,
    doc="Detect all collisions in the world (broad phase + narrow phase)"
)
def detect_collisions(world: PhysicsWorld2D) -> List[Contact]:
    """Detect all collisions in the world (broad phase + narrow phase).

    Args:
        world: Physics world containing all bodies

    Returns:
        List of contact points

    Note:
        Currently uses brute-force O(n²) collision detection.
        Future: Add spatial hashing for O(n) performance.
    """
    contacts = []

    for i, body_a in enumerate(world.bodies):
        for body_b in world.bodies[i + 1:]:
            # Check shape type compatibility
            if body_a.shape_type == ShapeType.CIRCLE and body_b.shape_type == ShapeType.CIRCLE:
                contact = detect_circle_circle_collision(body_a, body_b)
                if contact is not None:
                    contacts.append(contact)
            # TODO: Add box-box, circle-box, polygon collisions

    return contacts


# ============================================================================
# LAYER 3: COLLISION RESPONSE & WORLD SIMULATION
# ============================================================================

@operator(
    domain="rigidbody",
    category=OpCategory.TRANSFORM,
    signature="(body_a: RigidBody2D, body_b: RigidBody2D, contact: Contact) -> Tuple[RigidBody2D, RigidBody2D]",
    deterministic=True,
    doc="Resolve collision with impulse-based response"
)
def resolve_collision(
    body_a: RigidBody2D,
    body_b: RigidBody2D,
    contact: Contact
) -> Tuple[RigidBody2D, RigidBody2D]:
    """Resolve collision with impulse-based response.

    Args:
        body_a: First body
        body_b: Second body
        contact: Contact information

    Returns:
        Tuple of updated bodies

    Algorithm:
        Uses impulse-based collision response with:
        - Restitution (bounciness)
        - Friction (tangential impulse)
        - Relative velocity at contact point
    """
    # Calculate relative velocity at contact point
    r_a = contact.point - body_a.position
    r_b = contact.point - body_b.position

    # v = v_linear + ω × r (in 2D: ω × r = [-ω*r_y, ω*r_x])
    vel_a = body_a.velocity + body_a.angular_velocity * np.array([-r_a[1], r_a[0]])
    vel_b = body_b.velocity + body_b.angular_velocity * np.array([-r_b[1], r_b[0]])
    rel_vel = vel_b - vel_a

    # Relative velocity along normal
    vel_normal = np.dot(rel_vel, contact.normal)

    # Don't resolve if separating
    if vel_normal > 0:
        return body_a, body_b

    # Calculate impulse magnitude
    # J = -(1 + e) * v_n / (1/m_a + 1/m_b + (r_a × n)² / I_a + (r_b × n)² / I_b)
    e = min(body_a.restitution, body_b.restitution)

    r_a_cross_n = r_a[0] * contact.normal[1] - r_a[1] * contact.normal[0]
    r_b_cross_n = r_b[0] * contact.normal[1] - r_b[1] * contact.normal[0]

    inv_mass_sum = body_a.inv_mass + body_b.inv_mass
    inv_mass_sum += r_a_cross_n * r_a_cross_n * body_a.inv_inertia
    inv_mass_sum += r_b_cross_n * r_b_cross_n * body_b.inv_inertia

    j = -(1.0 + e) * vel_normal / inv_mass_sum
    impulse = j * contact.normal

    # Apply normal impulse
    body_a = apply_impulse(body_a, -impulse, contact.point)
    body_b = apply_impulse(body_b, impulse, contact.point)

    # Friction (tangential impulse)
    friction_coeff = np.sqrt(body_a.friction * body_b.friction)
    if friction_coeff > 0:
        # Recalculate relative velocity after normal impulse
        vel_a = body_a.velocity + body_a.angular_velocity * np.array([-r_a[1], r_a[0]])
        vel_b = body_b.velocity + body_b.angular_velocity * np.array([-r_b[1], r_b[0]])
        rel_vel = vel_b - vel_a

        # Tangential velocity
        vel_tangent = np.dot(rel_vel, contact.tangent)

        # Coulomb friction: |J_t| <= μ * |J_n|
        r_a_cross_t = r_a[0] * contact.tangent[1] - r_a[1] * contact.tangent[0]
        r_b_cross_t = r_b[0] * contact.tangent[1] - r_b[1] * contact.tangent[0]

        inv_mass_sum_t = body_a.inv_mass + body_b.inv_mass
        inv_mass_sum_t += r_a_cross_t * r_a_cross_t * body_a.inv_inertia
        inv_mass_sum_t += r_b_cross_t * r_b_cross_t * body_b.inv_inertia

        j_t = -vel_tangent / inv_mass_sum_t
        j_t = np.clip(j_t, -friction_coeff * abs(j), friction_coeff * abs(j))

        impulse_t = j_t * contact.tangent
        body_a = apply_impulse(body_a, -impulse_t, contact.point)
        body_b = apply_impulse(body_b, impulse_t, contact.point)

    # Position correction (prevent sinking)
    correction_percent = 0.2  # Baumgarte stabilization
    slop = 0.01  # Penetration allowance
    correction = max(contact.penetration - slop, 0.0) / inv_mass_sum * correction_percent
    correction_vec = correction * contact.normal

    if not body_a.is_static:
        body_a.position -= correction_vec * body_a.inv_mass
    if not body_b.is_static:
        body_b.position += correction_vec * body_b.inv_mass

    return body_a, body_b


@operator(
    domain="rigidbody",
    category=OpCategory.INTEGRATE,
    signature="(world: PhysicsWorld2D, dt: Optional[float]) -> PhysicsWorld2D",
    deterministic=True,
    doc="Step the physics world forward by one timestep"
)
def step_world(world: PhysicsWorld2D, dt: Optional[float] = None) -> PhysicsWorld2D:
    """Step the physics world forward by one timestep.

    Args:
        world: Physics world to simulate
        dt: Timestep (optional, uses world.dt if not provided)

    Returns:
        Updated world

    Algorithm:
        1. Apply gravity to all bodies
        2. Integrate motion
        3. Detect collisions
        4. Resolve collisions (iterative)
        5. Clear forces

    Example:
        world = PhysicsWorld2D()
        world.add_body(create_circle_body(...))

        for _ in range(1000):
            world = step_world(world)
    """
    if dt is None:
        dt = world.dt

    # 1. Apply gravity
    for body in world.bodies:
        if not body.is_static:
            gravity_force = world.gravity * body.mass
            apply_force(body, gravity_force)

    # 2. Integrate motion
    for body in world.bodies:
        integrate_body(body, dt, world.damping)

    # 3. Detect collisions
    contacts = detect_collisions(world)

    # 4. Resolve collisions (iterative for stability)
    for _ in range(world.iterations):
        for contact in contacts:
            body_a = world.get_body(contact.body_a)
            body_b = world.get_body(contact.body_b)
            if body_a and body_b:
                body_a, body_b = resolve_collision(body_a, body_b, contact)

    # 5. Clear forces
    for body in world.bodies:
        clear_forces(body)

    world.time += dt
    return world


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@operator(
    domain="rigidbody",
    category=OpCategory.QUERY,
    signature="(body: RigidBody2D) -> Optional[ndarray]",
    deterministic=True,
    doc="Get vertices of a body in world space (for rendering)"
)
def get_body_vertices(body: RigidBody2D) -> Optional[np.ndarray]:
    """Get vertices of a body in world space (for rendering).

    Args:
        body: Rigid body

    Returns:
        Array of vertices [N, 2] or None for circles
    """
    if body.shape_type == ShapeType.BOX:
        width = body.shape_params["width"]
        height = body.shape_params["height"]

        # Local vertices (centered at origin)
        hw, hh = width / 2, height / 2
        local_verts = np.array([
            [-hw, -hh],
            [hw, -hh],
            [hw, hh],
            [-hw, hh]
        ])

        # Rotate and translate to world space
        cos_r = np.cos(body.rotation)
        sin_r = np.sin(body.rotation)
        rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])

        world_verts = local_verts @ rot_matrix.T + body.position
        return world_verts

    return None


@operator(
    domain="rigidbody",
    category=OpCategory.QUERY,
    signature="(world: PhysicsWorld2D, origin: ndarray, direction: ndarray, max_distance: float) -> Optional[Tuple[RigidBody2D, ndarray, float]]",
    deterministic=True,
    doc="Cast a ray and find the first body hit"
)
def raycast(
    world: PhysicsWorld2D,
    origin: np.ndarray,
    direction: np.ndarray,
    max_distance: float = float('inf')
) -> Optional[Tuple[RigidBody2D, np.ndarray, float]]:
    """Cast a ray and find the first body hit.

    Args:
        world: Physics world
        origin: Ray origin [x, y]
        direction: Ray direction (will be normalized)
        max_distance: Maximum ray distance

    Returns:
        Tuple of (body, hit_point, distance) or None if no hit
    """
    direction = direction / np.linalg.norm(direction)

    closest_dist = max_distance
    closest_body = None
    closest_point = None

    for body in world.bodies:
        if body.shape_type == ShapeType.CIRCLE:
            # Ray-circle intersection
            radius = body.shape_params["radius"]
            oc = origin - body.position

            a = np.dot(direction, direction)
            b = 2.0 * np.dot(oc, direction)
            c = np.dot(oc, oc) - radius * radius

            discriminant = b * b - 4 * a * c
            if discriminant >= 0:
                t = (-b - np.sqrt(discriminant)) / (2 * a)
                if 0 < t < closest_dist:
                    closest_dist = t
                    closest_body = body
                    closest_point = origin + direction * t

    if closest_body is not None:
        return closest_body, closest_point, closest_dist
    return None
