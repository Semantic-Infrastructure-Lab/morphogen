"""FluidNetworkDomain - 1D lumped flow networks (pipes, ducts, circuits).

This module implements fluid network analysis using Modified Nodal Analysis (MNA),
similar to electrical circuit analysis. Essential for fire pits, HVAC systems,
mufflers, intake manifolds, and pneumatic circuits.

Specification: docs/specifications/physics-domains.md
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from enum import Enum

from morphogen.core.operator import operator, OpCategory


# ============================================================================
# Core Types
# ============================================================================

@dataclass
class Fluid:
    """Fluid properties."""
    density: float  # kg/m³
    viscosity: float  # Pa·s
    specific_heat: float  # J/(kg·K)

    @staticmethod
    def air(temp: float = 293.0) -> 'Fluid':
        """Create air properties at given temperature (K)."""
        # Ideal gas law: ρ = P/(R·T), using P=101325 Pa, R=287 J/(kg·K)
        density = 101325.0 / (287.0 * temp)
        # Sutherland's formula for viscosity
        mu_ref = 1.716e-5  # Pa·s at 273K
        T_ref = 273.0
        S = 110.4
        viscosity = mu_ref * (temp / T_ref)**1.5 * (T_ref + S) / (temp + S)
        specific_heat = 1005.0  # J/(kg·K)
        return Fluid(density, viscosity, specific_heat)


@dataclass
class Nozzle:
    """Nozzle geometry."""
    diameter: float  # m
    discharge_coeff: float = 0.95  # Typical value


@dataclass
class Tube:
    """Pipe geometry + flow properties."""
    diameter: float  # m
    length: float  # m
    roughness: float  # m (surface roughness)
    bends: int = 0  # Number of bends
    nozzle: Optional[Nozzle] = None
    fluid: Fluid = None

    def __post_init__(self):
        if self.fluid is None:
            self.fluid = Fluid.air()

    @property
    def area(self) -> float:
        """Cross-sectional area (m²)."""
        return np.pi * (self.diameter / 2.0)**2


@dataclass
class Junction:
    """Connection point in network."""
    position: Tuple[float, float, float]  # (x, y, z) in meters
    connected_tubes: List[int]  # Tube indices
    boundary_condition: Optional[Tuple[str, float]] = None  # ("pressure", value) or ("flow", value)


@dataclass
class FlowNet:
    """Complete fluid network (graph)."""
    tubes: List[Tuple[int, int, Tube]]  # (junction_from, junction_to, tube)
    junctions: List[Junction]

    def num_junctions(self) -> int:
        return len(self.junctions)

    def num_tubes(self) -> int:
        return len(self.tubes)


# ============================================================================
# Operators
# ============================================================================

@operator(
    domain="fluid_network",
    category=OpCategory.QUERY,
    signature="(height: float, T_amb: float, T_hot: float, g: float) -> float",
    deterministic=True,
    doc="Compute draft pressure from stack effect"
)
def draft_pressure(
    height: float,
    T_amb: float,
    T_hot: float,
    g: float = 9.81
) -> float:
    """Compute draft pressure from stack effect in vertical chamber.

    Formula: ΔP = ρ·g·H·(1/T_amb - 1/T_hot)

    Args:
        height: Chimney/chamber height (m)
        T_amb: Ambient temperature (K)
        T_hot: Hot gas temperature (K)
        g: Gravitational acceleration (m/s²)

    Returns:
        delta_p: Draft pressure (Pa)

    Determinism: strict
    """
    # Use ambient air density
    rho = 101325.0 / (287.0 * T_amb)
    delta_p = rho * g * height * (1.0 / T_amb - 1.0 / T_hot)
    return delta_p


@operator(
    domain="fluid_network",
    category=OpCategory.QUERY,
    signature="(tube: Tube, Re_guess: float) -> float",
    deterministic=True,
    doc="Compute flow resistance for a tube"
)
def tube_resistance(
    tube: Tube,
    Re_guess: float = 10000.0
) -> float:
    """Compute flow resistance for a tube (Darcy-Weisbach + minor losses).

    Formula: R = (f·L/D + K_bends)·(1/(2·ρ·A²))

    Args:
        tube: Tube geometry and properties
        Re_guess: Estimated Reynolds number for friction factor

    Returns:
        R_tube: Flow resistance (Pa·s²/kg²)

    Determinism: strict
    """
    # Colebrook-White equation for friction factor
    # For turbulent flow (Re > 2300), use approximate formula
    epsilon = tube.roughness / tube.diameter  # Relative roughness

    if Re_guess < 2300:
        # Laminar flow
        f = 64.0 / Re_guess
    else:
        # Turbulent flow - Swamee-Jain approximation of Colebrook-White
        f = 0.25 / (np.log10(epsilon / 3.7 + 5.74 / Re_guess**0.9))**2

    # Minor loss coefficient for bends (K ≈ 0.3 per 90° bend)
    K_bends = 0.3 * tube.bends

    # Total resistance
    R = (f * tube.length / tube.diameter + K_bends) / (2.0 * tube.fluid.density * tube.area**2)

    return R


def _compute_resistances(net, flows_guess, n_tubes) -> np.ndarray:
    """Compute tube resistances based on current flow guess (for friction factor iteration)."""
    resistances = np.zeros(n_tubes)
    for i, (_, _, tube) in enumerate(net.tubes):
        velocity = flows_guess[i] / (tube.fluid.density * tube.area)
        Re = max(tube.fluid.density * velocity * tube.diameter / tube.fluid.viscosity, 1.0)
        resistances[i] = tube_resistance(tube, Re)
    return resistances


def _build_mna_system(net, resistances, n_junctions, delta_p):
    """Build conductance matrix A and RHS b for Modified Nodal Analysis."""
    A = np.zeros((n_junctions, n_junctions))
    b = np.zeros(n_junctions)

    for i, (j_from, j_to, _) in enumerate(net.tubes):
        conductance = 1.0 / resistances[i] if resistances[i] > 0 else 1e10
        A[j_from, j_from] += conductance
        A[j_to, j_to] += conductance
        A[j_from, j_to] -= conductance
        A[j_to, j_from] -= conductance

    for j_idx, junction in enumerate(net.junctions):
        if junction.boundary_condition is not None:
            bc_type, bc_value = junction.boundary_condition
            if bc_type == "pressure":
                A[j_idx, :] = 0
                A[j_idx, j_idx] = 1.0
                b[j_idx] = bc_value

    if net.junctions[0].boundary_condition is None:
        b[0] += delta_p

    return A, b


def _solve_pressures(A, b, solver, tolerance, max_iterations, n_junctions):
    """Solve Ax=b for junction pressures. Returns None on singular matrix."""
    try:
        if solver == "direct":
            return np.linalg.solve(A, b)
        elif solver == "cg":
            pressures, _ = np.linalg.cg(A, b, tol=tolerance, maxiter=max_iterations)
            return pressures
        elif solver == "gmres":
            pressures, _ = np.linalg.gmres(A, b, tol=tolerance, maxiter=max_iterations)
            return pressures
        else:
            raise ValueError(f"Unknown solver: {solver}")
    except np.linalg.LinAlgError:
        return None


@operator(
    domain="fluid_network",
    category=OpCategory.QUERY,
    signature="(net: FlowNet, delta_p: float, solver: str, tolerance: float, max_iterations: int) -> Tuple[np.ndarray, np.ndarray]",
    deterministic=True,
    doc="Solve fluid network for flows and pressures"
)
def network_solve(
    net: FlowNet,
    delta_p: float,
    solver: str = "direct",
    tolerance: float = 1e-6,
    max_iterations: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve fluid network for flows and pressures using Modified Nodal Analysis.

    This is analogous to circuit analysis:
    - Pressure → Voltage
    - Flow → Current
    - Resistance → Resistance

    Args:
        net: Flow network
        delta_p: Driving pressure (Pa)
        solver: Solver method ("direct", "cg", "gmres")
        tolerance: Solver tolerance
        max_iterations: Maximum iterations for iterative solvers

    Returns:
        flows: Mass flow per tube (kg/s)
        pressures: Pressure per junction (Pa)

    Determinism: repro (iterative solver)
    """
    n_junctions = net.num_junctions()
    n_tubes = net.num_tubes()
    flows_guess = np.ones(n_tubes) * 0.01  # Initial guess (kg/s)

    for _ in range(max_iterations):
        resistances = _compute_resistances(net, flows_guess, n_tubes)
        A, b = _build_mna_system(net, resistances, n_junctions, delta_p)
        pressures = _solve_pressures(A, b, solver, tolerance, max_iterations, n_junctions)

        if pressures is None:
            return np.zeros(n_tubes), np.zeros(n_junctions)

        flows = np.array([
            np.sign(pressures[j_from] - pressures[j_to]) *
            np.sqrt(abs(pressures[j_from] - pressures[j_to]) / resistances[i])
            for i, (j_from, j_to, _) in enumerate(net.tubes)
        ])

        if np.max(np.abs(flows - flows_guess)) < tolerance:
            break
        flows_guess = flows

    return flows, pressures


# ============================================================================
# Convenience Functions
# ============================================================================

@operator(
    domain="fluid_network",
    category=OpCategory.CONSTRUCT,
    signature="(tube_diameter: float, tube_length: float, n_tubes: int) -> FlowNet",
    deterministic=True,
    doc="Create a J-tube fire pit air supply network"
)
def create_j_tube_network(
    tube_diameter: float,
    tube_length: float,
    n_tubes: int = 8
) -> FlowNet:
    """Create a simple J-tube fire pit air supply network.

    Args:
        tube_diameter: Internal diameter of J-tubes (m)
        tube_length: Length of each J-tube (m)
        n_tubes: Number of J-tubes around the fire pit

    Returns:
        FlowNet representing the J-tube system
    """
    # Create junctions
    junctions = [
        Junction((0, 0, 0), [], ("pressure", 0.0)),  # Ambient inlet
    ]

    # Add one junction per tube exit
    for i in range(n_tubes):
        angle = 2 * np.pi * i / n_tubes
        x = 0.5 * np.cos(angle)
        y = 0.5 * np.sin(angle)
        z = tube_length
        junctions.append(Junction((x, y, z), [], ("pressure", 0.0)))

    # Create tubes (from inlet to each exit)
    tubes = []
    for i in range(n_tubes):
        tube = Tube(
            diameter=tube_diameter,
            length=tube_length,
            roughness=1e-5,  # Smooth metal
            bends=1,  # One 90° bend for J-shape
            fluid=Fluid.air()
        )
        tubes.append((0, i + 1, tube))

    return FlowNet(tubes, junctions)


# ============================================================================
# Domain Registration
# ============================================================================

class FluidNetworkOperations:
    """Fluid network domain operations."""

    @staticmethod
    def draft_pressure(height, T_amb, T_hot, g=9.81):
        return draft_pressure(height, T_amb, T_hot, g)

    @staticmethod
    def tube_resistance(tube, Re_guess=10000.0):
        return tube_resistance(tube, Re_guess)

    @staticmethod
    def network_solve(net, delta_p, solver="direct", tolerance=1e-6, max_iterations=100):
        return network_solve(net, delta_p, solver, tolerance, max_iterations)

    @staticmethod
    def create_j_tube_network(tube_diameter, tube_length, n_tubes=8):
        return create_j_tube_network(tube_diameter, tube_length, n_tubes)


# Create domain instance
fluid_network = FluidNetworkOperations()


__all__ = [
    'Fluid', 'Nozzle', 'Tube', 'Junction', 'FlowNet',
    'draft_pressure', 'tube_resistance', 'network_solve',
    'create_j_tube_network', 'fluid_network', 'FluidNetworkOperations'
]
