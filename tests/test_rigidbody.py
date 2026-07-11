"""Comprehensive tests for rigid body physics domain.

Test coverage:
- Layer 1: Body creation, force/impulse application, integration
- Layer 2: Collision detection (circle-circle)
- Layer 3: Collision response, world simulation
- Physics properties: Energy conservation, restitution, friction
- Edge cases: Static bodies, overlapping bodies, zero mass
"""

try:
    import pytest
except ImportError:
    pytest = None

import numpy as np
from morphogen.stdlib.rigidbody import (
    RigidBody2D,
    PhysicsWorld2D,
    Contact,
    ShapeType,
    create_circle_body,
    create_box_body,
    apply_force,
    apply_impulse,
    clear_forces,
    integrate_body,
    detect_circle_circle_collision,
    detect_circle_box_collision,
    detect_box_box_collision,
    detect_collisions,
    resolve_collision,
    step_world,
    get_body_vertices,
    raycast,
)


# ============================================================================
# LAYER 1 TESTS: Body Creation and Dynamics
# ============================================================================

def test_create_circle_body():
    """Test circle body creation with correct inertia."""
    body = create_circle_body(
        position=np.array([1.0, 2.0]),
        radius=0.5,
        mass=2.0
    )

    assert body.shape_type == ShapeType.CIRCLE
    assert np.allclose(body.position, [1.0, 2.0])
    assert body.shape_params["radius"] == 0.5
    assert body.mass == 2.0
    # I = 0.5 * m * r²
    assert np.isclose(body.inertia, 0.5 * 2.0 * 0.5 * 0.5)


def test_create_box_body():
    """Test box body creation with correct inertia."""
    body = create_box_body(
        position=np.array([0.0, 0.0]),
        width=2.0,
        height=1.0,
        mass=3.0
    )

    assert body.shape_type == ShapeType.BOX
    assert np.allclose(body.position, [0.0, 0.0])
    assert body.shape_params["width"] == 2.0
    assert body.shape_params["height"] == 1.0
    assert body.mass == 3.0
    # I = (1/12) * m * (w² + h²)
    expected_inertia = (1.0 / 12.0) * 3.0 * (4.0 + 1.0)
    assert np.isclose(body.inertia, expected_inertia)


def test_static_body():
    """Test static body (mass = 0) properties."""
    body = create_circle_body(
        position=np.array([0.0, 0.0]),
        radius=1.0,
        mass=0.0  # Static
    )

    assert body.is_static
    assert body.inv_mass == 0.0
    assert body.inertia == 0.0


def test_apply_force():
    """Test force application accumulates correctly."""
    body = create_circle_body(
        position=np.array([0.0, 0.0]),
        radius=0.5,
        mass=1.0
    )

    # Apply force at center (no torque)
    body = apply_force(body, np.array([10.0, 0.0]))
    assert np.allclose(body.forces, [10.0, 0.0])
    assert body.torques == 0.0

    # Apply another force (accumulates)
    body = apply_force(body, np.array([0.0, 5.0]))
    assert np.allclose(body.forces, [10.0, 5.0])


def test_apply_force_with_torque():
    """Test force application at offset point generates torque."""
    body = create_circle_body(
        position=np.array([0.0, 0.0]),
        radius=0.5,
        mass=1.0
    )

    # Apply force at offset point
    # Torque = r × F = (0, 1) × (1, 0) = -1
    body = apply_force(
        body,
        np.array([1.0, 0.0]),
        point=np.array([0.0, 1.0])
    )

    assert np.allclose(body.forces, [1.0, 0.0])
    assert np.isclose(body.torques, -1.0)


def test_apply_impulse():
    """Test impulse changes velocity instantly."""
    body = create_circle_body(
        position=np.array([0.0, 0.0]),
        radius=0.5,
        mass=2.0
    )

    # Apply impulse
    body = apply_impulse(body, np.array([10.0, 0.0]))
    assert np.allclose(body.velocity, [5.0, 0.0])  # J/m = 10/2


def test_clear_forces():
    """Test force clearing."""
    body = create_circle_body(np.array([0.0, 0.0]), 0.5, 1.0)
    body = apply_force(body, np.array([10.0, 10.0]))

    body = clear_forces(body)
    assert np.allclose(body.forces, [0.0, 0.0])
    assert body.torques == 0.0


def test_integrate_body_free_fall():
    """Test body integration under gravity."""
    body = create_circle_body(
        position=np.array([0.0, 10.0]),
        radius=0.5,
        mass=1.0
    )

    # Apply gravity
    dt = 0.1
    gravity = np.array([0.0, -9.81])
    body = apply_force(body, gravity * body.mass)

    # Integrate
    initial_y = body.position[1]
    body = integrate_body(body, dt, damping=1.0)  # No damping

    # Check motion: y(t+dt) = y(t) + v(t+dt)*dt, v(t+dt) = v(t) + a*dt
    expected_vel_y = -9.81 * dt
    expected_y = initial_y + expected_vel_y * dt

    assert np.isclose(body.velocity[1], expected_vel_y, atol=1e-6)
    assert np.isclose(body.position[1], expected_y, atol=1e-6)


def test_integrate_body_rotation():
    """Test angular integration."""
    body = create_circle_body(
        position=np.array([0.0, 0.0]),
        radius=0.5,
        mass=1.0
    )

    # Apply torque
    body.torques = 1.0
    dt = 0.1

    body = integrate_body(body, dt, damping=1.0)

    # ω(t+dt) = ω(t) + α*dt, α = τ/I
    expected_ang_vel = 1.0 * body.inv_inertia * dt
    expected_rotation = expected_ang_vel * dt

    assert np.isclose(body.angular_velocity, expected_ang_vel, atol=1e-6)
    assert np.isclose(body.rotation, expected_rotation, atol=1e-6)


# ============================================================================
# LAYER 2 TESTS: Collision Detection
# ============================================================================

def test_circle_circle_collision_no_overlap():
    """Test no collision when circles don't overlap."""
    body_a = create_circle_body(np.array([0.0, 0.0]), radius=1.0, mass=1.0)
    body_b = create_circle_body(np.array([5.0, 0.0]), radius=1.0, mass=1.0)
    body_a.id = 0
    body_b.id = 1

    contact = detect_circle_circle_collision(body_a, body_b)
    assert contact is None


def test_circle_circle_collision_overlap():
    """Test collision detection when circles overlap."""
    body_a = create_circle_body(np.array([0.0, 0.0]), radius=1.0, mass=1.0)
    body_b = create_circle_body(np.array([1.5, 0.0]), radius=1.0, mass=1.0)
    body_a.id = 0
    body_b.id = 1

    contact = detect_circle_circle_collision(body_a, body_b)

    assert contact is not None
    assert contact.body_a == 0
    assert contact.body_b == 1
    assert np.isclose(contact.penetration, 0.5)  # 2.0 - 1.5
    assert np.allclose(contact.normal, [1.0, 0.0])


def test_circle_circle_collision_exact_overlap():
    """Test edge case: circles exactly on top of each other."""
    body_a = create_circle_body(np.array([0.0, 0.0]), radius=1.0, mass=1.0)
    body_b = create_circle_body(np.array([0.0, 0.0]), radius=1.0, mass=1.0)
    body_a.id = 0
    body_b.id = 1

    contact = detect_circle_circle_collision(body_a, body_b)

    assert contact is not None
    assert np.isclose(contact.penetration, 2.0)
    # Normal should be valid (fallback to [1, 0])
    assert np.allclose(contact.normal, [1.0, 0.0])


def test_detect_collisions_world():
    """Test collision detection in a world with multiple bodies."""
    world = PhysicsWorld2D()
    world.add_body(create_circle_body(np.array([0.0, 0.0]), radius=1.0, mass=1.0))
    world.add_body(create_circle_body(np.array([1.5, 0.0]), radius=1.0, mass=1.0))
    world.add_body(create_circle_body(np.array([10.0, 0.0]), radius=1.0, mass=1.0))

    contacts = detect_collisions(world)

    # Should detect collision between first two bodies
    assert len(contacts) == 1
    assert contacts[0].body_a == 0
    assert contacts[0].body_b == 1


def test_circle_box_collision_no_overlap():
    """Test no collision when a circle is well clear of a box."""
    circle = create_circle_body(np.array([5.0, 0.0]), radius=0.5, mass=1.0)
    box = create_box_body(np.array([0.0, 0.0]), width=1.0, height=1.0, mass=1.0)
    circle.id = 0
    box.id = 1

    contact = detect_circle_box_collision(circle, box)
    assert contact is None


def test_circle_box_collision_overlap():
    """Test collision detection when a circle overlaps an axis-aligned box edge."""
    circle = create_circle_body(np.array([1.2, 0.0]), radius=0.5, mass=1.0)
    box = create_box_body(np.array([0.0, 0.0]), width=2.0, height=2.0, mass=1.0)
    circle.id = 0
    box.id = 1

    contact = detect_circle_box_collision(circle, box)

    assert contact is not None
    assert contact.body_a == 0  # circle
    assert contact.body_b == 1  # box
    # Box half-width is 1.0; circle center at 1.2 with radius 0.5
    assert np.isclose(contact.penetration, 0.3)
    assert np.allclose(contact.normal, [-1.0, 0.0])  # circle -> box points left


def test_circle_box_collision_rotated_box():
    """Test circle-box collision against a 45-degree rotated box."""
    box = create_box_body(np.array([0.0, 0.0]), width=2.0, height=2.0, mass=1.0)
    box.rotation = np.pi / 4  # rotate 45 degrees
    circle = create_circle_body(np.array([1.5, 0.0]), radius=0.3, mass=1.0)
    circle.id = 0
    box.id = 1

    contact = detect_circle_box_collision(circle, box)
    assert contact is not None
    assert contact.penetration > 0


def test_circle_box_collision_center_inside_box():
    """Test the edge case where the circle center is embedded in the box."""
    circle = create_circle_body(np.array([0.3, 0.0]), radius=0.2, mass=1.0)
    box = create_box_body(np.array([0.0, 0.0]), width=4.0, height=4.0, mass=1.0)
    circle.id = 0
    box.id = 1

    contact = detect_circle_box_collision(circle, box)
    assert contact is not None
    assert contact.penetration > 0


def test_box_box_collision_no_overlap():
    """Test no collision when two axis-aligned boxes don't overlap."""
    body_a = create_box_body(np.array([0.0, 0.0]), width=1.0, height=1.0, mass=1.0)
    body_b = create_box_body(np.array([5.0, 0.0]), width=1.0, height=1.0, mass=1.0)
    body_a.id = 0
    body_b.id = 1

    contact = detect_box_box_collision(body_a, body_b)
    assert contact is None


def test_box_box_collision_overlap():
    """Test collision detection when two axis-aligned boxes overlap."""
    body_a = create_box_body(np.array([0.0, 0.0]), width=2.0, height=2.0, mass=1.0)
    body_b = create_box_body(np.array([1.5, 0.0]), width=2.0, height=2.0, mass=1.0)
    body_a.id = 0
    body_b.id = 1

    contact = detect_box_box_collision(body_a, body_b)

    assert contact is not None
    assert contact.body_a == 0
    assert contact.body_b == 1
    # Half-widths are 1.0 each; centers 1.5 apart -> overlap of 0.5 on the x axis
    assert np.isclose(contact.penetration, 0.5)
    assert np.allclose(contact.normal, [1.0, 0.0])  # A -> B points right


def test_box_box_collision_rotated():
    """Test SAT box-box collision when one box is rotated 45 degrees."""
    body_a = create_box_body(np.array([0.0, 0.0]), width=2.0, height=2.0, mass=1.0)
    body_b = create_box_body(np.array([1.4, 0.0]), width=1.0, height=1.0, mass=1.0)
    body_b.rotation = np.pi / 4
    body_a.id = 0
    body_b.id = 1

    contact = detect_box_box_collision(body_a, body_b)
    assert contact is not None
    assert contact.penetration > 0


def test_detect_collisions_world_mixed_shapes():
    """Test broad-phase dispatch handles circle-box and box-box pairs."""
    world = PhysicsWorld2D()
    world.add_body(create_circle_body(np.array([0.0, 0.0]), radius=0.5, mass=1.0))
    world.add_body(create_box_body(np.array([0.7, 0.0]), width=1.0, height=1.0, mass=1.0))
    world.add_body(create_box_body(np.array([0.7, 5.0]), width=1.0, height=1.0, mass=1.0))

    contacts = detect_collisions(world)

    # Only the circle (id=0) and first box (id=1) should be in contact
    assert len(contacts) == 1
    assert {contacts[0].body_a, contacts[0].body_b} == {0, 1}


# ============================================================================
# LAYER 3 TESTS: Collision Response & World Simulation
# ============================================================================

def test_resolve_collision_bounce():
    """Test collision response with restitution (bounce)."""
    # Body A moving right, Body B stationary
    body_a = create_circle_body(
        np.array([0.0, 0.0]),
        radius=1.0,
        mass=1.0,
        restitution=1.0  # Elastic
    )
    body_b = create_circle_body(
        np.array([1.9, 0.0]),
        radius=1.0,
        mass=1.0,
        restitution=1.0
    )
    body_a.id = 0
    body_b.id = 1

    # Give body_a velocity toward body_b
    body_a.velocity = np.array([10.0, 0.0])

    # Detect collision
    contact = detect_circle_circle_collision(body_a, body_b)
    assert contact is not None

    # Resolve
    body_a, body_b = resolve_collision(body_a, body_b, contact)

    # For elastic collision between equal masses, velocities should swap
    # (approximately, due to penetration correction)
    assert body_a.velocity[0] < 5.0  # Slowed down
    assert body_b.velocity[0] > 5.0  # Sped up


def test_resolve_collision_static_body():
    """Test collision with static body (infinite mass)."""
    # Moving body
    body_a = create_circle_body(
        np.array([0.0, 0.0]),
        radius=1.0,
        mass=1.0,
        restitution=0.8
    )
    # Static body (ground)
    body_b = create_circle_body(
        np.array([1.9, 0.0]),
        radius=1.0,
        mass=0.0,  # Static
        restitution=0.8
    )
    body_a.id = 0
    body_b.id = 1

    body_a.velocity = np.array([5.0, 0.0])

    contact = detect_circle_circle_collision(body_a, body_b)
    body_a, body_b = resolve_collision(body_a, body_b, contact)

    # Body A should bounce back, Body B should remain stationary
    assert body_a.velocity[0] < 0  # Reversed
    assert np.allclose(body_b.velocity, [0.0, 0.0])  # Static


def test_step_world_gravity():
    """Test world simulation with gravity."""
    world = PhysicsWorld2D(gravity=np.array([0.0, -10.0]))
    world.add_body(create_circle_body(
        np.array([0.0, 10.0]),
        radius=0.5,
        mass=1.0
    ))

    initial_y = world.bodies[0].position[1]

    # Step simulation
    world = step_world(world, dt=0.1)

    # Body should have fallen
    assert world.bodies[0].position[1] < initial_y
    assert world.bodies[0].velocity[1] < 0  # Falling


def test_step_world_collision():
    """Test world simulation with collision between two bodies."""
    world = PhysicsWorld2D(gravity=np.array([0.0, 0.0]))

    # Two bodies moving toward each other
    body_a = create_circle_body(np.array([0.0, 0.0]), radius=1.0, mass=1.0)
    body_b = create_circle_body(np.array([3.0, 0.0]), radius=1.0, mass=1.0)

    body_a.velocity = np.array([1.0, 0.0])
    body_b.velocity = np.array([-1.0, 0.0])

    world.add_body(body_a)
    world.add_body(body_b)

    # Simulate until collision
    for _ in range(20):
        world = step_world(world, dt=0.1)

    # Bodies should have collided and bounced
    # (exact values depend on timing, but they should have reversed)
    assert world.bodies[0].velocity[0] < 0.5  # Slowed or reversed
    assert world.bodies[1].velocity[0] > -0.5


def test_step_world_ground_bounce():
    """Test bouncing ball on ground."""
    world = PhysicsWorld2D(gravity=np.array([0.0, -10.0]))

    # Ball above ground
    ball = create_circle_body(
        np.array([0.0, 5.0]),
        radius=0.5,
        mass=1.0,
        restitution=0.8
    )
    # Ground (static)
    ground = create_circle_body(
        np.array([0.0, -10.0]),
        radius=10.0,  # Large radius for flat surface
        mass=0.0,  # Static
        restitution=0.8
    )

    world.add_body(ball)
    world.add_body(ground)

    # Simulate
    positions = []
    for _ in range(200):
        world = step_world(world, dt=0.016)
        positions.append(world.bodies[0].position[1])

    # Ball should bounce (multiple local maxima in y position)
    # Find peaks
    peaks = 0
    for i in range(1, len(positions) - 1):
        if positions[i] > positions[i - 1] and positions[i] > positions[i + 1]:
            peaks += 1

    assert peaks >= 2  # At least 2 bounces


# ============================================================================
# PHYSICS PROPERTY TESTS
# ============================================================================

def test_damping():
    """Test velocity damping reduces motion."""
    body = create_circle_body(np.array([0.0, 0.0]), 0.5, 1.0)
    body.velocity = np.array([10.0, 0.0])

    initial_speed = np.linalg.norm(body.velocity)

    # Integrate with damping
    body = integrate_body(body, dt=0.1, damping=0.9)

    final_speed = np.linalg.norm(body.velocity)
    assert final_speed < initial_speed


def test_restitution_zero():
    """Test perfectly inelastic collision (e=0)."""
    body_a = create_circle_body(np.array([0.0, 0.0]), 1.0, 1.0, restitution=0.0)
    body_b = create_circle_body(np.array([1.9, 0.0]), 1.0, 1.0, restitution=0.0)
    body_a.id = 0
    body_b.id = 1

    body_a.velocity = np.array([5.0, 0.0])

    contact = detect_circle_circle_collision(body_a, body_b)
    body_a, body_b = resolve_collision(body_a, body_b, contact)

    # Inelastic: relative velocity along normal should be ~0 after collision
    rel_vel = body_b.velocity - body_a.velocity
    vel_normal = np.dot(rel_vel, contact.normal)
    assert abs(vel_normal) < 1e-3  # Nearly zero


def test_friction():
    """Test friction opposes tangential motion."""
    body_a = create_circle_body(
        np.array([0.0, 0.0]),
        radius=1.0,
        mass=1.0,
        friction=0.8
    )
    body_b = create_circle_body(
        np.array([1.9, 0.0]),
        radius=1.0,
        mass=0.0,  # Static
        friction=0.8
    )
    body_a.id = 0
    body_b.id = 1

    # Give body_a velocity with tangential component
    body_a.velocity = np.array([1.0, 5.0])

    contact = detect_circle_circle_collision(body_a, body_b)
    initial_tangent_vel = abs(np.dot(body_a.velocity, contact.tangent))

    body_a, body_b = resolve_collision(body_a, body_b, contact)

    final_tangent_vel = abs(np.dot(body_a.velocity, contact.tangent))

    # Friction should reduce tangential velocity
    assert final_tangent_vel < initial_tangent_vel


# ============================================================================
# UTILITY FUNCTION TESTS
# ============================================================================

def test_get_body_vertices_box():
    """Test box vertex calculation."""
    body = create_box_body(
        np.array([0.0, 0.0]),
        width=2.0,
        height=1.0,
        mass=1.0
    )

    verts = get_body_vertices(body)
    assert verts is not None
    assert verts.shape == (4, 2)

    # Check vertices are correct (unrotated)
    expected = np.array([
        [-1.0, -0.5],
        [1.0, -0.5],
        [1.0, 0.5],
        [-1.0, 0.5]
    ])
    assert np.allclose(verts, expected)


def test_get_body_vertices_rotated():
    """Test box vertices with rotation."""
    body = create_box_body(
        np.array([0.0, 0.0]),
        width=2.0,
        height=1.0,
        mass=1.0
    )
    body.rotation = np.pi / 2  # 90 degrees

    verts = get_body_vertices(body)

    # After 90° rotation, width and height should swap
    # Check first vertex approximately
    assert np.allclose(verts[0], [0.5, -1.0], atol=1e-6)


def test_get_body_vertices_circle():
    """Test circle returns None (no vertices)."""
    body = create_circle_body(np.array([0.0, 0.0]), 1.0, 1.0)
    verts = get_body_vertices(body)
    assert verts is None


def test_raycast_hit():
    """Test raycast hitting a body."""
    world = PhysicsWorld2D()
    world.add_body(create_circle_body(
        np.array([5.0, 0.0]),
        radius=1.0,
        mass=1.0
    ))

    result = raycast(
        world,
        origin=np.array([0.0, 0.0]),
        direction=np.array([1.0, 0.0])
    )

    assert result is not None
    body, hit_point, distance = result
    assert body.id == 0
    assert np.isclose(distance, 4.0, atol=0.1)  # Hit at x=4 (circle center at 5, radius 1)


def test_raycast_miss():
    """Test raycast missing all bodies."""
    world = PhysicsWorld2D()
    world.add_body(create_circle_body(
        np.array([5.0, 5.0]),
        radius=1.0,
        mass=1.0
    ))

    result = raycast(
        world,
        origin=np.array([0.0, 0.0]),
        direction=np.array([1.0, 0.0])
    )

    assert result is None


def test_raycast_max_distance():
    """Test raycast with max distance limit."""
    world = PhysicsWorld2D()
    world.add_body(create_circle_body(
        np.array([10.0, 0.0]),
        radius=1.0,
        mass=1.0
    ))

    result = raycast(
        world,
        origin=np.array([0.0, 0.0]),
        direction=np.array([1.0, 0.0]),
        max_distance=5.0
    )

    assert result is None  # Too far


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_stack_of_circles():
    """Test stable stack of circles."""
    world = PhysicsWorld2D(gravity=np.array([0.0, -10.0]))

    # Ground
    ground = create_circle_body(
        np.array([0.0, -10.0]),
        radius=10.0,
        mass=0.0
    )

    # Stack of circles
    circle1 = create_circle_body(np.array([0.0, 0.0]), radius=0.5, mass=1.0)
    circle2 = create_circle_body(np.array([0.0, 1.1]), radius=0.5, mass=1.0)

    world.add_body(ground)
    world.add_body(circle1)
    world.add_body(circle2)

    # Simulate
    for _ in range(200):
        world = step_world(world, dt=0.016)

    # Stack should be stable (bodies at rest)
    assert abs(world.bodies[1].velocity[1]) < 0.1
    assert abs(world.bodies[2].velocity[1]) < 0.1


def test_determinism():
    """Test simulation is deterministic (same inputs -> same outputs)."""
    def run_simulation():
        world = PhysicsWorld2D(gravity=np.array([0.0, -10.0]))
        world.add_body(create_circle_body(np.array([0.0, 10.0]), 0.5, 1.0))
        world.add_body(create_circle_body(np.array([0.5, 5.0]), 0.5, 1.0))

        for _ in range(100):
            world = step_world(world, dt=0.016)

        return world.bodies[0].position.copy(), world.bodies[1].position.copy()

    pos1_a, pos1_b = run_simulation()
    pos2_a, pos2_b = run_simulation()

    assert np.allclose(pos1_a, pos2_a)
    assert np.allclose(pos1_b, pos2_b)


def test_energy_conservation_elastic():
    """Test energy conservation in elastic collision.

    Fixed: Energy conservation requires damping=1.0 (no damping).
    The default damping=0.99 was causing 87% energy loss over 100 steps.
    """
    # Two bodies, elastic collision, no gravity, no damping
    world = PhysicsWorld2D(
        gravity=np.array([0.0, 0.0]),
        damping=1.0,  # No damping for energy conservation
        angular_damping=1.0
    )

    body_a = create_circle_body(np.array([0.0, 0.0]), 1.0, 1.0, restitution=1.0)
    body_b = create_circle_body(np.array([3.0, 0.0]), 1.0, 1.0, restitution=1.0)

    body_a.velocity = np.array([2.0, 0.0])
    body_b.velocity = np.array([0.0, 0.0])

    world.add_body(body_a)
    world.add_body(body_b)

    # Initial kinetic energy
    initial_ke = 0.5 * body_a.mass * np.dot(body_a.velocity, body_a.velocity)

    # Simulate
    for _ in range(100):
        world = step_world(world, dt=0.01)

    # Final kinetic energy
    final_ke = sum(
        0.5 * body.mass * np.dot(body.velocity, body.velocity)
        for body in world.bodies if not body.is_static
    )

    # Energy should be conserved (within numerical error)
    assert np.isclose(initial_ke, final_ke, rtol=0.1)  # 10% tolerance for numerical integration
