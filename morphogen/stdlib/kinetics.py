"""KineticsDomain - Chemical reaction kinetics and reactor modeling.

This module implements reaction rate laws, reaction network integration,
ideal and non-ideal reactor models. Essential for process engineering,
combustion modeling, and chemical synthesis optimization.

Specification: docs/specifications/chemistry.md
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional, Tuple
from scipy.integrate import ode, solve_ivp
from scipy.optimize import root
from enum import Enum
from morphogen.core.operator import operator, OpCategory


# ============================================================================
# Constants
# ============================================================================

R_GAS = 8.314  # J/(mol·K) - Universal gas constant


# ============================================================================
# Core Types
# ============================================================================

class RateLawType(Enum):
    """Types of rate laws."""
    ARRHENIUS = "arrhenius"
    MODIFIED_ARRHENIUS = "modified_arrhenius"
    CUSTOM = "custom"


@dataclass
class RateLaw:
    """Chemical reaction rate law."""
    type: RateLawType
    A: float = 0.0  # Pre-exponential factor
    Ea: float = 0.0  # Activation energy (J/mol)
    n: float = 0.0  # Temperature exponent (for modified Arrhenius)
    custom_func: Optional[Callable] = None  # Custom rate function


@dataclass
class Reaction:
    """Chemical reaction definition."""
    reactants: Dict[str, float]  # species -> stoichiometry
    products: Dict[str, float]
    rate_law: RateLaw
    reversible: bool = False
    reverse_rate_law: Optional[RateLaw] = None

    @property
    def all_species(self) -> List[str]:
        """Get all species involved in reaction."""
        return list(set(list(self.reactants.keys()) + list(self.products.keys())))


class ReactorType(Enum):
    """Types of chemical reactors."""
    BATCH = "batch"
    CSTR = "cstr"
    PFR = "pfr"
    PBR = "pbr"  # Packed bed reactor


@dataclass
class Reactor:
    """Chemical reactor specification."""
    type: ReactorType
    volume: float  # m³
    temp: float  # K
    pressure: float  # Pa
    reactions: List[Reaction] = field(default_factory=list)


# ============================================================================
# Rate Law Operators
# ============================================================================

@operator(
    domain="kinetics",
    category=OpCategory.QUERY,
    signature="(temp: float, A: float, Ea: float) -> float",
    deterministic=True,
    doc="Compute Arrhenius rate constant"
)
def arrhenius(temp: float, A: float, Ea: float) -> float:
    """Compute Arrhenius rate constant.

    k = A * exp(-Ea / (R * T))

    Args:
        temp: Temperature (K)
        A: Pre-exponential factor (units depend on reaction order)
        Ea: Activation energy (J/mol)

    Returns:
        k: Rate constant

    Determinism: strict
    """
    k = A * np.exp(-Ea / (R_GAS * temp))
    return k


@operator(
    domain="kinetics",
    category=OpCategory.QUERY,
    signature="(temp: float, A: float, n: float, Ea: float) -> float",
    deterministic=True,
    doc="Compute modified Arrhenius rate constant"
)
def modified_arrhenius(temp: float, A: float, n: float, Ea: float) -> float:
    """Compute modified Arrhenius rate constant.

    k = A * T^n * exp(-Ea / (R * T))

    Args:
        temp: Temperature (K)
        A: Pre-exponential factor
        n: Temperature exponent
        Ea: Activation energy (J/mol)

    Returns:
        k: Rate constant

    Determinism: strict
    """
    k = A * (temp ** n) * np.exp(-Ea / (R_GAS * temp))
    return k


@operator(
    domain="kinetics",
    category=OpCategory.QUERY,
    signature="(temp: float, delta_H: float, delta_S: float, T_ref: float) -> float",
    deterministic=True,
    doc="Compute equilibrium constant from van't Hoff equation"
)
def vant_hoff(temp: float, delta_H: float, delta_S: float, T_ref: float = 298.15) -> float:
    """Compute equilibrium constant from van't Hoff equation.

    ln(K) = -ΔH/(R·T) + ΔS/R

    Args:
        temp: Temperature (K)
        delta_H: Enthalpy of reaction (J/mol)
        delta_S: Entropy of reaction (J/(mol·K))
        T_ref: Reference temperature (K)

    Returns:
        K_eq: Equilibrium constant

    Determinism: strict
    """
    K_eq = np.exp(-delta_H / (R_GAS * temp) + delta_S / R_GAS)
    return K_eq


# ============================================================================
# Reaction Network Operators
# ============================================================================

@operator(
    domain="kinetics",
    category=OpCategory.QUERY,
    signature="(conc: Dict[str, float], temp: float, reactions: List[Reaction]) -> Dict[str, float]",
    deterministic=True,
    doc="Compute reaction rates for all species"
)
def reaction_rates(
    conc: Dict[str, float],
    temp: float,
    reactions: List[Reaction]
) -> Dict[str, float]:
    """Compute reaction rates for all species.

    Args:
        conc: Species concentrations (mol/L)
        temp: Temperature (K)
        reactions: List of reactions

    Returns:
        rates: Rate of change for each species (mol/(L·s))

    Determinism: strict
    """
    # Initialize rates
    rates = {species: 0.0 for rxn in reactions for species in rxn.all_species}

    for reaction in reactions:
        k = _compute_rate_constant(reaction.rate_law, conc, temp)

        forward_rate = k
        for species, stoich in reaction.reactants.items():
            forward_rate *= conc.get(species, 0.0) ** stoich

        reverse_rate = 0.0
        if reaction.reversible and reaction.reverse_rate_law:
            k_rev = _compute_rate_constant(reaction.reverse_rate_law, conc, temp)
            reverse_rate = k_rev
            for species, stoich in reaction.products.items():
                reverse_rate *= conc.get(species, 0.0) ** stoich

        # Net rate
        net_rate = forward_rate - reverse_rate

        # Apply stoichiometry to species rates
        for species, stoich in reaction.reactants.items():
            rates[species] -= stoich * net_rate

        for species, stoich in reaction.products.items():
            rates[species] += stoich * net_rate

    return rates


def _compute_rate_constant(rate_law, conc: Dict[str, float], temp: float) -> float:
    """Compute rate constant k from a RateLaw and current conditions."""
    if rate_law.type == RateLawType.ARRHENIUS:
        return arrhenius(temp, rate_law.A, rate_law.Ea)
    if rate_law.type == RateLawType.MODIFIED_ARRHENIUS:
        return modified_arrhenius(temp, rate_law.A, rate_law.n, rate_law.Ea)
    if rate_law.type == RateLawType.CUSTOM:
        return rate_law.custom_func(conc, temp)
    return 0.0


@operator(
    domain="kinetics",
    category=OpCategory.INTEGRATE,
    signature="(conc_initial: Dict[str, float], reactions: List[Reaction], temp: float, time: float, method: str, time_points: Optional[np.ndarray]) -> Dict[str, np.ndarray]",
    deterministic=True,
    doc="Integrate reaction network ODEs over time"
)
def integrate_ode(
    conc_initial: Dict[str, float],
    reactions: List[Reaction],
    temp: float,
    time: float,
    method: str = "bdf",
    time_points: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """Integrate reaction network ODEs over time.

    Args:
        conc_initial: Initial concentrations (mol/L)
        reactions: List of reactions
        temp: Temperature (K)
        time: Total integration time (s)
        method: Integration method ("euler", "rk4", "bdf", "lsoda")
        time_points: Specific time points to evaluate (optional)

    Returns:
        conc_history: Concentration vs time for each species

    Determinism: repro (adaptive step size in some methods)
    """
    # Get all species
    all_species = list(conc_initial.keys())
    n_species = len(all_species)

    # Initial state vector
    y0 = np.array([conc_initial.get(s, 0.0) for s in all_species])

    # Define ODE system
    def dydt(t, y):
        conc_dict = {species: max(y[i], 0.0) for i, species in enumerate(all_species)}
        rates = reaction_rates(conc_dict, temp, reactions)
        return np.array([rates[species] for species in all_species])

    # Time points
    if time_points is None:
        time_points = np.linspace(0, time, 1000)

    # Integrate
    if method == "euler":
        # Simple Euler method
        dt = time_points[1] - time_points[0]
        y = y0.copy()
        y_history = [y.copy()]

        for t in time_points[1:]:
            dydt_val = dydt(t, y)
            y += dydt_val * dt
            y = np.maximum(y, 0.0)  # Prevent negative concentrations
            y_history.append(y.copy())

        y_history = np.array(y_history).T

    elif method == "rk4":
        # Runge-Kutta 4th order
        dt = time_points[1] - time_points[0]
        y = y0.copy()
        y_history = [y.copy()]

        for t in time_points[1:]:
            k1 = dydt(t, y)
            k2 = dydt(t + dt/2, y + k1*dt/2)
            k3 = dydt(t + dt/2, y + k2*dt/2)
            k4 = dydt(t + dt, y + k3*dt)

            y += (k1 + 2*k2 + 2*k3 + k4) * dt / 6
            y = np.maximum(y, 0.0)
            y_history.append(y.copy())

        y_history = np.array(y_history).T

    else:
        # Use scipy's solve_ivp (BDF, LSODA, etc.)
        sol = solve_ivp(
            dydt,
            (0, time),
            y0,
            method=method.upper() if method in ["bdf", "lsoda"] else "RK45",
            t_eval=time_points,
            dense_output=False
        )
        y_history = sol.y

    # Convert back to dictionary
    conc_history = {species: y_history[i] for i, species in enumerate(all_species)}
    conc_history['time'] = time_points

    return conc_history


# ============================================================================
# Ideal Reactor Operators
# ============================================================================

@operator(
    domain="kinetics",
    category=OpCategory.INTEGRATE,
    signature="(initial_conc: Dict[str, float], reactions: List[Reaction], temp: float, time: float, method: str) -> Dict[str, float]",
    deterministic=True,
    doc="Simulate batch reactor (closed, well-mixed)"
)
def batch_reactor(
    initial_conc: Dict[str, float],
    reactions: List[Reaction],
    temp: float,
    time: float,
    method: str = "bdf"
) -> Dict[str, float]:
    """Simulate batch reactor (closed, well-mixed).

    Args:
        initial_conc: Initial concentrations (mol/L)
        reactions: List of reactions
        temp: Temperature (K)
        time: Reaction time (s)
        method: ODE integration method

    Returns:
        final_conc: Final concentrations (mol/L)

    Determinism: repro
    """
    conc_history = integrate_ode(initial_conc, reactions, temp, time, method)

    # Return final concentrations
    final_conc = {}
    for species in initial_conc.keys():
        if species in conc_history:
            final_conc[species] = conc_history[species][-1]

    return final_conc


@operator(
    domain="kinetics",
    category=OpCategory.QUERY,
    signature="(feed_conc: Dict[str, float], feed_flow: float, volume: float, reactions: List[Reaction], temp: float) -> Dict[str, float]",
    deterministic=True,
    doc="Solve CSTR (Continuous Stirred Tank Reactor) steady-state"
)
def cstr(
    feed_conc: Dict[str, float],
    feed_flow: float,
    volume: float,
    reactions: List[Reaction],
    temp: float
) -> Dict[str, float]:
    """Solve CSTR (Continuous Stirred Tank Reactor) steady-state.

    Mass balance: 0 = F_in * c_in - F_out * c + V * r

    Args:
        feed_conc: Feed concentrations (mol/L)
        feed_flow: Volumetric flow rate (L/s)
        volume: Reactor volume (L)
        reactions: List of reactions
        temp: Temperature (K)

    Returns:
        outlet_conc: Outlet concentrations (mol/L)

    Determinism: repro (iterative solver)
    """
    # Residence time
    tau = volume / feed_flow

    species_list = list(feed_conc.keys())

    def residuals(c_vec):
        c = dict(zip(species_list, c_vec))
        rates = reaction_rates(c, temp, reactions)
        return [
            c[s] - feed_conc.get(s, 0.0) - tau * rates.get(s, 0.0)
            for s in species_list
        ]

    c0 = np.array([feed_conc[s] for s in species_list])
    sol = root(residuals, c0, method='hybr', tol=1e-10)
    c_sol = np.maximum(sol.x, 0.0)  # concentrations cannot be negative

    return dict(zip(species_list, c_sol))


@operator(
    domain="kinetics",
    category=OpCategory.INTEGRATE,
    signature="(feed_conc: Dict[str, float], reactions: List[Reaction], length: float, area: float, flow_velocity: float, temp: float, n_points: int) -> Dict[str, np.ndarray]",
    deterministic=True,
    doc="Simulate Plug Flow Reactor (PFR) spatial profile"
)
def pfr(
    feed_conc: Dict[str, float],
    reactions: List[Reaction],
    length: float,
    area: float,
    flow_velocity: float,
    temp: float,
    n_points: int = 100
) -> Dict[str, np.ndarray]:
    """Simulate Plug Flow Reactor (PFR) spatial profile.

    Args:
        feed_conc: Feed concentrations (mol/L)
        reactions: List of reactions
        length: Reactor length (m)
        area: Cross-sectional area (m²)
        flow_velocity: Axial velocity (m/s)
        temp: Temperature (K)
        n_points: Number of spatial points

    Returns:
        conc_profile: Concentration vs position for each species

    Determinism: strict
    """
    # PFR is equivalent to batch reactor in time domain
    # Space time = length / velocity
    space_time = length / flow_velocity

    # Convert to time-domain ODE
    z_points = np.linspace(0, length, n_points)
    t_points = z_points / flow_velocity

    conc_history = integrate_ode(feed_conc, reactions, temp, space_time, time_points=t_points)

    # Add position coordinate
    conc_profile = {**conc_history}
    conc_profile['position'] = z_points

    return conc_profile


# ============================================================================
# Non-Ideal Reactor Operators
# ============================================================================

@operator(
    domain="kinetics",
    category=OpCategory.QUERY,
    signature="(k_intrinsic: float, k_mass_transfer: float) -> float",
    deterministic=True,
    doc="Compute effective rate constant for mass-transfer-limited reaction"
)
def mass_transfer_limited(k_intrinsic: float, k_mass_transfer: float) -> float:
    """Compute effective rate constant for mass-transfer-limited reaction.

    1/k_eff = 1/k_intrinsic + 1/k_mass_transfer

    Args:
        k_intrinsic: Intrinsic reaction rate constant
        k_mass_transfer: Mass transfer coefficient

    Returns:
        k_eff: Effective rate constant

    Determinism: strict
    """
    k_eff = 1.0 / (1.0 / k_intrinsic + 1.0 / k_mass_transfer)
    return k_eff


@operator(
    domain="kinetics",
    category=OpCategory.INTEGRATE,
    signature="(feed_conc: Dict[str, float], reactions: List[Reaction], length: float, velocity: float, dispersion_coeff: float, temp: float, n_points: int) -> Dict[str, np.ndarray]",
    deterministic=True,
    doc="Simulate PFR with axial dispersion"
)
def pfr_with_dispersion(
    feed_conc: Dict[str, float],
    reactions: List[Reaction],
    length: float,
    velocity: float,
    dispersion_coeff: float,
    temp: float,
    n_points: int = 100
) -> Dict[str, np.ndarray]:
    """Simulate PFR with axial dispersion.

    PDE: dc/dt = -v * dc/dz + D * d²c/dz² + r(c)

    Args:
        feed_conc: Feed concentrations (mol/L)
        reactions: List of reactions
        length: Reactor length (m)
        velocity: Flow velocity (m/s)
        dispersion_coeff: Axial dispersion coefficient (m²/s)
        temp: Temperature (K)
        n_points: Number of spatial points

    Returns:
        conc_profile: Steady-state concentration profile

    Determinism: repro
    """
    # Discretize in space
    z = np.linspace(0, length, n_points)
    dz = z[1] - z[0]

    all_species = list(feed_conc.keys())
    n_species = len(all_species)

    # Initial guess (feed concentration everywhere)
    conc_field = np.zeros((n_species, n_points))
    for i, species in enumerate(all_species):
        conc_field[i, :] = feed_conc[species]

    # Iteratively solve steady-state PDE
    for iteration in range(1000):
        conc_field_new = conc_field.copy()

        for iz in range(1, n_points - 1):
            # Build concentration dict for this point
            conc_dict = {species: conc_field[i, iz] for i, species in enumerate(all_species)}

            # Compute reaction rates
            rates = reaction_rates(conc_dict, temp, reactions)

            for i, species in enumerate(all_species):
                # Finite differences
                dc_dz = (conc_field[i, iz + 1] - conc_field[i, iz - 1]) / (2 * dz)
                d2c_dz2 = (conc_field[i, iz + 1] - 2 * conc_field[i, iz] + conc_field[i, iz - 1]) / dz**2

                # Steady state: 0 = -v * dc/dz + D * d²c/dz² + r
                # Solve for c at interior points
                r = rates[species]
                conc_field_new[i, iz] = conc_field[i, iz] + 0.01 * (
                    -velocity * dc_dz + dispersion_coeff * d2c_dz2 + r
                )

        # Boundary conditions
        for i in range(n_species):
            conc_field_new[i, 0] = feed_conc[all_species[i]]  # Inlet
            conc_field_new[i, -1] = conc_field_new[i, -2]  # Outlet (zero gradient)

        # Check convergence
        error = np.max(np.abs(conc_field_new - conc_field))
        if error < 1e-6:
            break

        conc_field = conc_field_new

    # Convert to dictionary
    conc_profile = {species: conc_field[i] for i, species in enumerate(all_species)}
    conc_profile['position'] = z

    return conc_profile


# ============================================================================
# Utility Functions
# ============================================================================

@operator(
    domain="kinetics",
    category=OpCategory.CONSTRUCT,
    signature="(reactants: Dict[str, float], products: Dict[str, float], A: float, Ea: float, reversible: bool) -> Reaction",
    deterministic=True,
    doc="Create a simple Arrhenius reaction"
)
def create_reaction(
    reactants: Dict[str, float],
    products: Dict[str, float],
    A: float,
    Ea: float,
    reversible: bool = False
) -> Reaction:
    """Create a simple Arrhenius reaction.

    Args:
        reactants: Reactant stoichiometry
        products: Product stoichiometry
        A: Pre-exponential factor
        Ea: Activation energy (J/mol)
        reversible: Is reaction reversible

    Returns:
        reaction: Reaction object

    Determinism: strict
    """
    rate_law = RateLaw(type=RateLawType.ARRHENIUS, A=A, Ea=Ea)
    return Reaction(reactants, products, rate_law, reversible)


# ============================================================================
# Domain Registration
# ============================================================================

class KineticsOperations:
    """Reaction kinetics domain operations."""

    # Rate Laws
    arrhenius = staticmethod(arrhenius)
    modified_arrhenius = staticmethod(modified_arrhenius)
    vant_hoff = staticmethod(vant_hoff)

    # Reaction Networks
    reaction_rates = staticmethod(reaction_rates)
    integrate_ode = staticmethod(integrate_ode)

    # Ideal Reactors
    batch_reactor = staticmethod(batch_reactor)
    cstr = staticmethod(cstr)
    pfr = staticmethod(pfr)

    # Non-Ideal Reactors
    mass_transfer_limited = staticmethod(mass_transfer_limited)
    pfr_with_dispersion = staticmethod(pfr_with_dispersion)

    # Utilities
    create_reaction = staticmethod(create_reaction)


# Create domain instance
kinetics = KineticsOperations()


__all__ = [
    'RateLawType', 'RateLaw', 'Reaction', 'ReactorType', 'Reactor',
    'arrhenius', 'modified_arrhenius', 'vant_hoff',
    'reaction_rates', 'integrate_ode',
    'batch_reactor', 'cstr', 'pfr',
    'mass_transfer_limited', 'pfr_with_dispersion',
    'create_reaction', 'kinetics', 'KineticsOperations'
]
