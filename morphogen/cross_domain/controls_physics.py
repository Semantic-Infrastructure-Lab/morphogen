"""
Controls ↔ Physics Domain Interfaces

Interfaces for closing feedback loops between rigidbody physics and
the controls domain (PID, Kalman, LQR).
"""

from typing import Any, Dict, List, Optional, Tuple, Type
import numpy as np

from .base import DomainInterface


class PhysicsToControlsInterface(DomainInterface):
    """
    RigidBody2D → Controls: Convert physics state to measurement vector.

    Extracts position, velocity, angle, and angular_velocity from a
    RigidBody2D and maps user-selected components to a measurement
    vector y for Kalman estimation or PID feedback.

    Use cases:
    - Quadrotor: y = [x, y, theta] from body state
    - Ball-on-plate: y = [x, y] (position only, velocity estimated by Kalman)
    - Pendulum: y = [theta, theta_dot]
    """

    source_domain = "rigidbody"
    target_domain = "controls"

    # Available state components and their indices in the full state vector
    _COMPONENT_MAP = {
        "x":               ("position", 0),
        "y":               ("position", 1),
        "vx":              ("velocity", 0),
        "vy":              ("velocity", 1),
        "angle":           ("rotation", None),
        "angular_velocity": ("angular_velocity", None),
    }

    def __init__(
        self,
        body,
        sensor_mapping: Optional[List[str]] = None,
    ):
        """
        Args:
            body: RigidBody2D instance (or list of them)
            sensor_mapping: Ordered list of component names to include in y.
                Available: 'x', 'y', 'vx', 'vy', 'angle', 'angular_velocity'.
                Defaults to ['x', 'y'] (position only).
        """
        super().__init__(source_data=body)
        self.body = body
        self.sensor_mapping = sensor_mapping or ["x", "y"]

        unknown = [k for k in self.sensor_mapping if k not in self._COMPONENT_MAP]
        if unknown:
            raise ValueError(
                f"Unknown sensor components: {unknown}. "
                f"Available: {list(self._COMPONENT_MAP)}"
            )

    def transform(self, source_data: Any = None) -> np.ndarray:
        """
        Extract measurement vector y from RigidBody2D state.

        Args:
            source_data: RigidBody2D instance. If None, uses self.body.

        Returns:
            y: Measurement vector (p,) with components per sensor_mapping.
        """
        body = source_data if source_data is not None else self.body
        y = []

        for component in self.sensor_mapping:
            attr, idx = self._COMPONENT_MAP[component]
            raw = getattr(body, attr)
            if idx is not None:
                value = float(raw[idx])
            else:
                value = float(raw)
            y.append(value)

        return np.array(y, dtype=float)

    def validate(self) -> bool:
        """Check body has required attributes."""
        required = {"position", "velocity", "rotation", "angular_velocity"}
        for attr in required:
            if not hasattr(self.body, attr):
                raise AttributeError(
                    f"Body missing required attribute '{attr}'. "
                    f"Expected a RigidBody2D instance."
                )
        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {"body": Any, "sensor_mapping": List[str]}

    def get_output_interface(self) -> Dict[str, Type]:
        return {"y": np.ndarray}


class ControlsToPhysicsInterface(DomainInterface):
    """
    Controls → RigidBody2D: Apply control output vector u as forces/torques.

    Maps elements of the control output u to forces and torques on a
    RigidBody2D, enabling LQR/PID controllers to actuate simulated bodies.

    Use cases:
    - Rocket: u[0]=thrust_x, u[1]=thrust_y
    - Drone: u[0]=force_x, u[1]=force_y, u[2]=torque
    - Pendulum: u[0]=torque only
    """

    source_domain = "controls"
    target_domain = "rigidbody"

    def __init__(
        self,
        body,
        actuator_mapping: Optional[Dict[int, str]] = None,
    ):
        """
        Args:
            body: RigidBody2D instance to actuate.
            actuator_mapping: Maps u-index to effect.
                Keys: int (index into u vector)
                Values: 'force_x', 'force_y', 'torque'
                Defaults to {0: 'force_x', 1: 'force_y'}.
        """
        super().__init__(source_data=body)
        self.body = body
        self.actuator_mapping = actuator_mapping or {0: "force_x", 1: "force_y"}

        valid_effects = {"force_x", "force_y", "torque"}
        unknown = [v for v in self.actuator_mapping.values() if v not in valid_effects]
        if unknown:
            raise ValueError(
                f"Unknown actuator effects: {unknown}. Available: {valid_effects}"
            )

    def transform(self, source_data: Any = None) -> Any:
        """
        Apply control output u to the body.

        Calls rigidbody.apply_force() and/or modifies torque directly.

        Args:
            source_data: Control output vector u (m,). If None, applies zeros.

        Returns:
            RigidBody2D with updated forces and torques (ready for physics step).
        """
        from morphogen.stdlib import rigidbody

        u = source_data
        if u is None:
            u = np.zeros(max(self.actuator_mapping.keys()) + 1)
        u = np.atleast_1d(np.asarray(u, dtype=float))

        import copy
        body = copy.copy(self.body)
        body.forces = body.forces.copy()

        force = np.zeros(2)
        torque = 0.0

        for idx, effect in self.actuator_mapping.items():
            if idx < len(u):
                val = u[idx]
                if effect == "force_x":
                    force[0] += val
                elif effect == "force_y":
                    force[1] += val
                elif effect == "torque":
                    torque += val

        # Apply accumulated forces via rigidbody operator
        if np.any(force != 0):
            body = rigidbody.apply_force(body, force)
        if torque != 0.0:
            # apply_torque is available in rigidbody
            if hasattr(rigidbody, "apply_torque"):
                body = rigidbody.apply_torque(body, torque)
            else:
                body.torques = body.torques + torque

        return body

    def validate(self) -> bool:
        """Check body has forces and torques attributes."""
        if not hasattr(self.body, "forces"):
            raise AttributeError("Body missing 'forces' attribute. Expected RigidBody2D.")
        if not hasattr(self.body, "torques"):
            raise AttributeError("Body missing 'torques' attribute. Expected RigidBody2D.")
        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {"u": np.ndarray, "actuator_mapping": Dict[int, str]}

    def get_output_interface(self) -> Dict[str, Type]:
        return {"body": Any}
