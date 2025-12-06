"""MolecularDomain - Molecular structure, mechanics, and dynamics.

This module implements molecular representation, force field calculations, geometry
optimization, conformer generation, molecular dynamics simulations, and trajectory
analysis. Essential for drug design, materials science, and chemical simulations.

Specification: docs/specifications/chemistry.md

Note: For production use, this module can be extended with RDKit for enhanced
molecular informatics capabilities. Current implementation provides core functionality
with NumPy/SciPy.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Union
from enum import Enum
import warnings


# ============================================================================
# Core Types
# ============================================================================

@dataclass
class Atom:
    """Atomic structure and properties."""
    element: str  # Element symbol (e.g., "C", "H", "O")
    atomic_number: int
    mass: float  # amu
    charge: float = 0.0  # e (elementary charge)

    @staticmethod
    def from_element(element: str, charge: float = 0.0) -> 'Atom':
        """Create atom from element symbol."""
        # Periodic table data (partial)
        periodic_table = {
            'H': (1, 1.008), 'C': (6, 12.011), 'N': (7, 14.007),
            'O': (8, 15.999), 'F': (9, 18.998), 'Na': (11, 22.990),
            'P': (15, 30.974), 'S': (16, 32.06), 'Cl': (17, 35.45),
            'Br': (35, 79.904)
        }
        if element not in periodic_table:
            raise ValueError(f"Unknown element: {element}")

        atomic_number, mass = periodic_table[element]
        return Atom(element, atomic_number, mass, charge)


@dataclass
class Bond:
    """Chemical bond between two atoms."""
    atom1: int  # Index of first atom
    atom2: int  # Index of second atom
    order: float  # 1.0=single, 2.0=double, 1.5=aromatic, 3.0=triple


@dataclass
class Molecule:
    """Molecular structure with atoms, bonds, and coordinates."""
    atoms: List[Atom]
    bonds: List[Bond]
    positions: np.ndarray  # Shape (n_atoms, 3), Angstrom
    velocities: Optional[np.ndarray] = None  # Shape (n_atoms, 3), Angstrom/fs
    forces: Optional[np.ndarray] = None  # Shape (n_atoms, 3), kcal/(mol·Angstrom)

    def __post_init__(self):
        """Initialize arrays if needed."""
        if self.velocities is None:
            self.velocities = np.zeros_like(self.positions)
        if self.forces is None:
            self.forces = np.zeros_like(self.positions)

    @property
    def n_atoms(self) -> int:
        """Number of atoms in molecule."""
        return len(self.atoms)

    @property
    def masses(self) -> np.ndarray:
        """Array of atomic masses (amu)."""
        return np.array([atom.mass for atom in self.atoms])

    @property
    def charges(self) -> np.ndarray:
        """Array of atomic charges (e)."""
        return np.array([atom.charge for atom in self.atoms])

    # Convenience methods that wrap module-level functions
    def num_atoms(self) -> int:
        """Get number of atoms (alias for n_atoms property)."""
        return self.n_atoms

    def total_mass(self) -> float:
        """Calculate total molecular mass."""
        return molecular_weight(self)

    def center_of_mass(self) -> np.ndarray:
        """Calculate center of mass."""
        return center_of_mass(self)

    def moment_of_inertia(self) -> np.ndarray:
        """Calculate moment of inertia tensor."""
        return moment_of_inertia(self)


@dataclass
class Trajectory:
    """Molecular dynamics trajectory."""
    frames: List[Molecule]
    times: np.ndarray  # fs
    energies: np.ndarray  # kcal/mol

    @property
    def n_frames(self) -> int:
        return len(self.frames)


@dataclass
class ForceField:
    """Force field specification."""
    name: str  # "amber", "charmm", "uff", "gaff", "oplsaa"
    parameters: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# Loading & Conversion Operators
# ============================================================================

def load_smiles(smiles: str, generate_3d: bool = True) -> Molecule:
    """Load molecule from SMILES string.

    Note: This is a stub implementation. For production, use RDKit:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        AllChem.EmbedMolecule(mol)

    Args:
        smiles: SMILES string (e.g., "CCO" for ethanol)
        generate_3d: Generate 3D coordinates

    Returns:
        molecule: Molecular structure

    Determinism: strict (with fixed seed for 3D generation)
    """
    warnings.warn("load_smiles is a stub. Use RDKit for full implementation.")

    # Stub: Create simple ethanol-like molecule
    atoms = [
        Atom.from_element('C'),
        Atom.from_element('C'),
        Atom.from_element('O'),
        Atom.from_element('H'),
        Atom.from_element('H'),
    ]
    bonds = [
        Bond(0, 1, 1.0),
        Bond(1, 2, 1.0),
    ]
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.54, 0.0, 0.0],
        [2.2, 1.2, 0.0],
        [-0.5, 0.9, 0.0],
        [-0.5, -0.9, 0.0],
    ])

    return Molecule(atoms, bonds, positions)


def load_xyz(filepath: str) -> Molecule:
    """Load molecule from XYZ file.

    Args:
        filepath: Path to XYZ file

    Returns:
        molecule: Molecular structure

    Determinism: strict
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    n_atoms = int(lines[0].strip())
    # Comment line is lines[1]

    atoms = []
    positions = []
    for i in range(2, 2 + n_atoms):
        parts = lines[i].split()
        element = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])

        atoms.append(Atom.from_element(element))
        positions.append([x, y, z])

    positions = np.array(positions)
    bonds = []  # XYZ doesn't store bonds

    return Molecule(atoms, bonds, positions)


def load_pdb(filepath: str) -> Molecule:
    """Load molecule from PDB file.

    Args:
        filepath: Path to PDB file

    Returns:
        molecule: Molecular structure

    Determinism: strict
    """
    atoms = []
    positions = []

    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                element = line[76:78].strip()
                if not element:
                    element = line[12:14].strip()[0]

                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                atoms.append(Atom.from_element(element))
                positions.append([x, y, z])

    positions = np.array(positions)
    bonds = []  # PDB bond info requires CONECT records

    return Molecule(atoms, bonds, positions)


def to_smiles(molecule: Molecule) -> str:
    """Convert molecule to SMILES string.

    Note: Stub implementation. Use RDKit for production.

    Args:
        molecule: Molecular structure

    Returns:
        smiles: SMILES string

    Determinism: strict
    """
    warnings.warn("to_smiles is a stub. Use RDKit for full implementation.")
    return "CCO"  # Placeholder


def to_xyz(molecule: Molecule) -> str:
    """Convert molecule to XYZ format string.

    Args:
        molecule: Molecular structure

    Returns:
        xyz: XYZ format string

    Determinism: strict
    """
    lines = [str(molecule.n_atoms), "Generated by Morphogen"]

    for atom, pos in zip(molecule.atoms, molecule.positions):
        line = f"{atom.element:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}"
        lines.append(line)

    return '\n'.join(lines)


def generate_3d(
    molecule: Molecule,
    force_field: str = "uff",
    seed: int = 42
) -> Molecule:
    """Generate 3D coordinates from 2D/SMILES.

    Args:
        molecule: Molecular structure
        force_field: Force field for optimization ("uff", "mmff")
        seed: Random seed for reproducibility

    Returns:
        molecule_3d: Molecule with 3D coordinates

    Determinism: strict (with fixed seed)
    """
    np.random.seed(seed)

    # Simple 3D embedding using random perturbation + relaxation
    positions_3d = molecule.positions.copy()
    if positions_3d.shape[1] == 2:
        # Add Z dimension
        z = np.random.randn(molecule.n_atoms) * 0.1
        positions_3d = np.column_stack([positions_3d, z])

    molecule_3d = Molecule(
        molecule.atoms,
        molecule.bonds,
        positions_3d,
        molecule.velocities,
        molecule.forces
    )

    # Quick geometry optimization
    return optimize_geometry(molecule_3d, force_field, max_iterations=100)


# ============================================================================
# Molecular Properties
# ============================================================================

def molecular_weight(molecule: Molecule) -> float:
    """Compute molecular weight.

    Args:
        molecule: Molecular structure

    Returns:
        mw: Molecular weight (g/mol)

    Determinism: strict
    """
    return np.sum(molecule.masses)


def molecular_formula(molecule: Molecule) -> str:
    """Compute molecular formula.

    Args:
        molecule: Molecular structure

    Returns:
        formula: Molecular formula (e.g., "C2H6O")

    Determinism: strict
    """
    element_counts = {}
    for atom in molecule.atoms:
        element_counts[atom.element] = element_counts.get(atom.element, 0) + 1

    # Order: C, H, then alphabetical
    formula_parts = []
    for element in ['C', 'H']:
        if element in element_counts:
            count = element_counts.pop(element)
            formula_parts.append(f"{element}{count if count > 1 else ''}")

    for element in sorted(element_counts.keys()):
        count = element_counts[element]
        formula_parts.append(f"{element}{count if count > 1 else ''}")

    return ''.join(formula_parts)


def center_of_mass(molecule: Molecule) -> np.ndarray:
    """Compute center of mass.

    Args:
        molecule: Molecular structure

    Returns:
        com: Center of mass (Angstrom), shape (3,)

    Determinism: strict
    """
    masses = molecule.masses
    total_mass = np.sum(masses)
    com = np.sum(molecule.positions * masses[:, np.newaxis], axis=0) / total_mass
    return com


def moment_of_inertia(molecule: Molecule) -> np.ndarray:
    """Compute moment of inertia tensor.

    Args:
        molecule: Molecular structure

    Returns:
        I: Moment of inertia tensor (amu·Angstrom²), shape (3, 3)

    Determinism: strict
    """
    com = center_of_mass(molecule)
    positions_centered = molecule.positions - com
    masses = molecule.masses

    I = np.zeros((3, 3))
    for i in range(molecule.n_atoms):
        r = positions_centered[i]
        m = masses[i]

        I[0, 0] += m * (r[1]**2 + r[2]**2)
        I[1, 1] += m * (r[0]**2 + r[2]**2)
        I[2, 2] += m * (r[0]**2 + r[1]**2)

        I[0, 1] -= m * r[0] * r[1]
        I[0, 2] -= m * r[0] * r[2]
        I[1, 2] -= m * r[1] * r[2]

    I[1, 0] = I[0, 1]
    I[2, 0] = I[0, 2]
    I[2, 1] = I[1, 2]

    return I


def dipole_moment(molecule: Molecule) -> np.ndarray:
    """Compute electric dipole moment.

    Args:
        molecule: Molecular structure

    Returns:
        dipole: Dipole moment (Debye), shape (3,)

    Determinism: strict
    """
    # Dipole = sum(q_i * r_i)
    charges = molecule.charges
    dipole_au = np.sum(charges[:, np.newaxis] * molecule.positions, axis=0)

    # Convert from e·Angstrom to Debye (1 Debye = 0.20819 e·Angstrom)
    dipole_debye = dipole_au / 0.20819

    return dipole_debye


def find_rings(molecule: Molecule) -> List[List[int]]:
    """Find ring systems in molecule.

    Args:
        molecule: Molecular structure

    Returns:
        rings: List of atom indices forming rings

    Determinism: strict
    """
    # Build adjacency list
    adj = [set() for _ in range(molecule.n_atoms)]
    for bond in molecule.bonds:
        adj[bond.atom1].add(bond.atom2)
        adj[bond.atom2].add(bond.atom1)

    # Simple cycle detection using DFS
    rings = []
    visited = set()

    def dfs(node, parent, path):
        if node in visited:
            if node in path:
                # Found a cycle
                cycle_start = path.index(node)
                ring = path[cycle_start:]
                if len(ring) >= 3:
                    rings.append(sorted(ring))
            return

        visited.add(node)
        path.append(node)

        for neighbor in adj[node]:
            if neighbor != parent:
                dfs(neighbor, node, path.copy())

    for start_node in range(molecule.n_atoms):
        if start_node not in visited:
            dfs(start_node, -1, [])

    # Remove duplicates
    unique_rings = []
    for ring in rings:
        if ring not in unique_rings:
            unique_rings.append(ring)

    return unique_rings


# ============================================================================
# Force Field Calculations
# ============================================================================

def lennard_jones_potential(r: np.ndarray, epsilon: float, sigma: float) -> np.ndarray:
    """Calculate Lennard-Jones potential energy.

    Args:
        r: Distance(s) between particles (Angstroms)
        epsilon: Depth of potential well (kJ/mol)
        sigma: Distance at which potential is zero (Angstroms)

    Returns:
        Potential energy (kJ/mol)
    """
    return 4.0 * epsilon * ((sigma / r)**12 - (sigma / r)**6)


def harmonic_bond_energy(r: float, r_eq: float, k: float) -> float:
    """Calculate harmonic bond energy.

    Args:
        r: Current bond length (Angstroms)
        r_eq: Equilibrium bond length (Angstroms)
        k: Force constant (kcal/mol/Angstrom^2)

    Returns:
        Bond energy (kcal/mol)
    """
    return 0.5 * k * (r - r_eq)**2


def coulomb_energy(q1: float, q2: float, r: float) -> float:
    """Calculate Coulomb electrostatic energy.

    Args:
        q1: Charge on first particle (elementary charge)
        q2: Charge on second particle (elementary charge)
        r: Distance between particles (Angstroms)

    Returns:
        Electrostatic energy (kcal/mol)
    """
    # Coulomb constant in kcal*Angstrom/(mol*e^2)
    k_e = 332.0637
    return k_e * q1 * q2 / r


def compute_energy(
    molecule: Molecule,
    force_field: str = "uff",
    include_terms: List[str] = None
) -> float:
    """Compute molecular energy using force field.

    Args:
        molecule: Molecular structure
        force_field: Force field name
        include_terms: Energy terms to include

    Returns:
        energy: Total energy (kcal/mol)

    Determinism: strict
    """
    if include_terms is None:
        include_terms = ["bond", "angle", "dihedral", "vdw", "electrostatic"]

    energy = 0.0

    if "bond" in include_terms:
        energy += bond_energy(molecule, force_field)
    if "angle" in include_terms:
        energy += angle_energy(molecule, force_field)
    if "vdw" in include_terms:
        energy += vdw_energy(molecule, force_field)
    if "electrostatic" in include_terms:
        energy += electrostatic_energy(molecule, force_field)

    return energy


def bond_energy(molecule: Molecule, force_field: str = "uff") -> float:
    """Compute bond stretching energy.

    E = sum_bonds k_b * (r - r0)^2

    Args:
        molecule: Molecular structure
        force_field: Force field name

    Returns:
        energy: Bond energy (kcal/mol)

    Determinism: strict
    """
    # Typical force constants and equilibrium distances
    bond_params = {
        1.0: (340.0, 1.54),  # Single bond (k, r0)
        2.0: (680.0, 1.34),  # Double bond
        3.0: (1020.0, 1.20),  # Triple bond
    }

    energy = 0.0
    for bond in molecule.bonds:
        i, j = bond.atom1, bond.atom2
        r = np.linalg.norm(molecule.positions[j] - molecule.positions[i])

        k, r0 = bond_params.get(bond.order, (340.0, 1.54))
        energy += k * (r - r0)**2

    return energy


def angle_energy(molecule: Molecule, force_field: str = "uff") -> float:
    """Compute angle bending energy.

    Args:
        molecule: Molecular structure
        force_field: Force field name

    Returns:
        energy: Angle energy (kcal/mol)

    Determinism: strict
    """
    # Typical angle force constant
    k_theta = 50.0  # kcal/(mol·rad²)
    theta0 = 109.5 * np.pi / 180.0  # Tetrahedral angle

    # Build adjacency for finding angles
    adj = [[] for _ in range(molecule.n_atoms)]
    for bond in molecule.bonds:
        adj[bond.atom1].append(bond.atom2)
        adj[bond.atom2].append(bond.atom1)

    energy = 0.0
    for i in range(molecule.n_atoms):
        neighbors = adj[i]
        if len(neighbors) < 2:
            continue

        # All pairs of neighbors form angles
        for j in range(len(neighbors)):
            for k in range(j + 1, len(neighbors)):
                atom_j = neighbors[j]
                atom_k = neighbors[k]

                # Compute angle j-i-k
                v1 = molecule.positions[atom_j] - molecule.positions[i]
                v2 = molecule.positions[atom_k] - molecule.positions[i]

                cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                theta = np.arccos(cos_theta)

                energy += k_theta * (theta - theta0)**2

    return energy


def vdw_energy(molecule: Molecule, force_field: str = "uff") -> float:
    """Compute van der Waals energy (Lennard-Jones).

    E = sum_pairs 4*epsilon * [(sigma/r)^12 - (sigma/r)^6]

    Args:
        molecule: Molecular structure
        force_field: Force field name

    Returns:
        energy: VdW energy (kcal/mol)

    Determinism: strict
    """
    # UFF parameters (epsilon in kcal/mol, sigma in Angstrom)
    vdw_params = {
        'H': (0.044, 2.571),
        'C': (0.105, 3.431),
        'N': (0.069, 3.261),
        'O': (0.060, 3.118),
    }

    energy = 0.0
    for i in range(molecule.n_atoms):
        for j in range(i + 1, molecule.n_atoms):
            r = np.linalg.norm(molecule.positions[j] - molecule.positions[i])

            elem_i = molecule.atoms[i].element
            elem_j = molecule.atoms[j].element

            eps_i, sig_i = vdw_params.get(elem_i, (0.05, 3.0))
            eps_j, sig_j = vdw_params.get(elem_j, (0.05, 3.0))

            # Lorentz-Berthelot combining rules
            epsilon = np.sqrt(eps_i * eps_j)
            sigma = (sig_i + sig_j) / 2.0

            # Lennard-Jones
            if r > 0.1:  # Avoid singularity
                sr = sigma / r
                energy += 4 * epsilon * (sr**12 - sr**6)

    return energy


def electrostatic_energy(molecule: Molecule, force_field: str = "uff") -> float:
    """Compute electrostatic energy (Coulomb).

    E = sum_pairs k_e * q_i * q_j / r

    Args:
        molecule: Molecular structure
        force_field: Force field name

    Returns:
        energy: Electrostatic energy (kcal/mol)

    Determinism: strict
    """
    # Coulomb constant in kcal·Angstrom/(mol·e²)
    k_e = 332.0637

    energy = 0.0
    charges = molecule.charges

    for i in range(molecule.n_atoms):
        for j in range(i + 1, molecule.n_atoms):
            r = np.linalg.norm(molecule.positions[j] - molecule.positions[i])

            if r > 0.1:  # Avoid singularity
                energy += k_e * charges[i] * charges[j] / r

    return energy


def compute_forces(molecule: Molecule, force_field: str = "uff") -> np.ndarray:
    """Compute forces on all atoms.

    Args:
        molecule: Molecular structure
        force_field: Force field name

    Returns:
        forces: Atomic forces (kcal/(mol·Angstrom)), shape (n_atoms, 3)

    Determinism: strict
    """
    forces = np.zeros_like(molecule.positions)

    # Numerical gradient (finite differences)
    epsilon = 1e-5
    for i in range(molecule.n_atoms):
        for dim in range(3):
            # Forward
            molecule.positions[i, dim] += epsilon
            E_plus = compute_energy(molecule, force_field)

            # Backward
            molecule.positions[i, dim] -= 2 * epsilon
            E_minus = compute_energy(molecule, force_field)

            # Restore
            molecule.positions[i, dim] += epsilon

            # Force = -dE/dx
            forces[i, dim] = -(E_plus - E_minus) / (2 * epsilon)

    return forces


# ============================================================================
# Geometry Optimization
# ============================================================================

def optimize_geometry(
    molecule: Molecule,
    force_field: str = "uff",
    method: str = "bfgs",
    max_iterations: int = 1000,
    convergence: float = 1e-6
) -> Molecule:
    """Minimize molecular energy.

    Args:
        molecule: Initial structure
        force_field: Force field name
        method: Optimization method
        max_iterations: Maximum iterations
        convergence: Convergence criterion (kcal/(mol·Angstrom))

    Returns:
        molecule_opt: Optimized structure

    Determinism: strict
    """
    positions = molecule.positions.copy()

    for iteration in range(max_iterations):
        # Compute forces
        mol_temp = Molecule(molecule.atoms, molecule.bonds, positions)
        forces = compute_forces(mol_temp, force_field)

        # Check convergence
        force_rms = np.sqrt(np.mean(forces**2))
        if force_rms < convergence:
            break

        # Steepest descent step
        step_size = 0.01
        positions += step_size * forces

    return Molecule(molecule.atoms, molecule.bonds, positions)


def optimize_constrained(
    molecule: Molecule,
    force_field: str = "uff",
    constraints: List = None
) -> Molecule:
    """Optimize geometry with constraints.

    Args:
        molecule: Initial structure
        force_field: Force field name
        constraints: List of constraint functions

    Returns:
        molecule_opt: Optimized structure

    Determinism: strict
    """
    # Simplified: just optimize without constraints for now
    return optimize_geometry(molecule, force_field)


# ============================================================================
# Conformer Generation
# ============================================================================

def generate_conformers(
    molecule: Molecule,
    n: int = 100,
    method: str = "random",
    energy_window: float = 10.0,
    rms_threshold: float = 0.5,
    seed: int = 42
) -> List[Molecule]:
    """Generate conformers (different 3D structures).

    Args:
        molecule: Base structure
        n: Number of conformers to generate
        method: Generation method
        energy_window: Max energy above minimum (kcal/mol)
        rms_threshold: Min RMSD between conformers (Angstrom)
        seed: Random seed

    Returns:
        conformers: List of conformer structures

    Determinism: strict (with fixed seed)
    """
    np.random.seed(seed)
    conformers = []

    for i in range(n):
        # Random perturbation + optimization
        positions = molecule.positions + np.random.randn(*molecule.positions.shape) * 0.5
        conf = Molecule(molecule.atoms, molecule.bonds, positions)
        conf = optimize_geometry(conf, max_iterations=100)

        # Check energy window
        energy = compute_energy(conf)
        if i == 0:
            min_energy = energy

        if energy - min_energy <= energy_window:
            # Check RMSD with existing conformers
            unique = True
            for existing in conformers:
                if rmsd(conf, existing, align=False) < rms_threshold:
                    unique = False
                    break

            if unique:
                conformers.append(conf)

    return conformers


def cluster_conformers(
    conformers: List[Molecule],
    method: str = "rmsd",
    threshold: float = 1.0
) -> List[List[int]]:
    """Cluster conformers by similarity.

    Args:
        conformers: List of conformers
        method: Clustering method
        threshold: Clustering threshold (Angstrom for RMSD)

    Returns:
        clusters: List of conformer indices per cluster

    Determinism: strict
    """
    n = len(conformers)

    # Compute distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = rmsd(conformers[i], conformers[j], align=False)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    # Simple hierarchical clustering
    clusters = [[i] for i in range(n)]

    while len(clusters) > 1:
        # Find closest pair
        min_dist = float('inf')
        merge_i, merge_j = 0, 1

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Average distance between clusters
                dist_sum = 0
                count = 0
                for ci in clusters[i]:
                    for cj in clusters[j]:
                        dist_sum += dist_matrix[ci, cj]
                        count += 1
                avg_dist = dist_sum / count

                if avg_dist < min_dist:
                    min_dist = avg_dist
                    merge_i, merge_j = i, j

        if min_dist > threshold:
            break

        # Merge clusters
        clusters[merge_i].extend(clusters[merge_j])
        clusters.pop(merge_j)

    return clusters


# ============================================================================
# Molecular Dynamics
# ============================================================================

def velocity_verlet(
    positions: np.ndarray,
    velocities: np.ndarray,
    forces: np.ndarray,
    masses: np.ndarray,
    dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Velocity Verlet integration step.

    Args:
        positions: Current positions (Angstrom)
        velocities: Current velocities (Angstrom/fs)
        forces: Current forces (kcal/(mol·Angstrom))
        masses: Atomic masses (amu)
        dt: Time step (fs)

    Returns:
        positions_new: Updated positions
        velocities_new: Updated velocities

    Determinism: strict
    """
    # Convert forces to accelerations
    # 1 kcal/(mol·amu·Angstrom) = 418.4 Angstrom²/fs²
    conversion = 418.4
    accelerations = forces / masses[:, np.newaxis] * conversion

    # Update positions
    positions_new = positions + velocities * dt + 0.5 * accelerations * dt**2

    # Update velocities (need new forces, but use current for now)
    velocities_new = velocities + accelerations * dt

    return positions_new, velocities_new


def langevin_integrator(
    positions: np.ndarray,
    velocities: np.ndarray,
    forces: np.ndarray,
    masses: np.ndarray,
    temp: float,
    friction: float,
    dt: float,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Langevin dynamics integration step.

    Args:
        positions: Current positions (Angstrom)
        velocities: Current velocities (Angstrom/fs)
        forces: Current forces (kcal/(mol·Angstrom))
        masses: Atomic masses (amu)
        temp: Temperature (K)
        friction: Friction coefficient (1/ps)
        dt: Time step (fs)
        seed: Random seed

    Returns:
        positions_new: Updated positions
        velocities_new: Updated velocities

    Determinism: strict (with fixed seed)
    """
    np.random.seed(seed)

    # Boltzmann constant in kcal/(mol·K)
    k_B = 0.001987

    # Convert forces to accelerations
    conversion = 418.4
    accelerations = forces / masses[:, np.newaxis] * conversion

    # Langevin equation: dv = a*dt - gamma*v*dt + sqrt(2*gamma*k_B*T/m)*dW
    gamma = friction * 1e-3  # Convert to 1/fs

    random_force = np.sqrt(2 * gamma * k_B * temp / masses[:, np.newaxis])
    random_force *= np.random.randn(*velocities.shape) / np.sqrt(dt)

    # Update
    velocities_new = velocities + (accelerations - gamma * velocities + random_force) * dt
    positions_new = positions + velocities_new * dt

    return positions_new, velocities_new


def md_simulate(
    molecule: Molecule,
    force_field: str = "uff",
    temp: float = 300.0,
    pressure: Optional[float] = None,
    time: float = 10000.0,
    dt: float = 1.0,
    ensemble: str = "nvt",
    save_interval: int = 100
) -> Trajectory:
    """Run molecular dynamics simulation.

    Args:
        molecule: Initial structure
        force_field: Force field name
        temp: Temperature (K)
        pressure: Pressure (atm), for NPT
        time: Simulation time (fs)
        dt: Time step (fs)
        ensemble: Ensemble type ("nve", "nvt", "npt")
        save_interval: Save frequency (steps)

    Returns:
        trajectory: MD trajectory

    Determinism: repro (stochastic thermostats)
    """
    # Initialize velocities from Maxwell-Boltzmann
    k_B = 0.001987  # kcal/(mol·K)
    masses = molecule.masses

    # v_rms = sqrt(k_B * T / m)
    velocities = np.random.randn(molecule.n_atoms, 3)
    for i in range(molecule.n_atoms):
        v_scale = np.sqrt(k_B * temp / masses[i])
        velocities[i] *= v_scale * np.sqrt(418.4)  # Convert to Angstrom/fs

    # Remove net momentum
    velocities -= np.mean(velocities, axis=0)

    # Simulation
    positions = molecule.positions.copy()
    n_steps = int(time / dt)

    frames = []
    times = []
    energies = []

    for step in range(n_steps):
        # Compute forces
        mol_temp = Molecule(molecule.atoms, molecule.bonds, positions, velocities)
        forces = compute_forces(mol_temp, force_field)

        # Integration
        if ensemble == "nvt":
            positions, velocities = langevin_integrator(
                positions, velocities, forces, masses, temp, friction=1.0, dt=dt
            )
        else:
            positions, velocities = velocity_verlet(
                positions, velocities, forces, masses, dt
            )

        # Save frame
        if step % save_interval == 0:
            mol_frame = Molecule(molecule.atoms, molecule.bonds, positions.copy())
            frames.append(mol_frame)
            times.append(step * dt)
            energies.append(compute_energy(mol_frame, force_field))

    return Trajectory(frames, np.array(times), np.array(energies))


# ============================================================================
# Trajectory Analysis
# ============================================================================

def rmsd(
    molecule1: Molecule,
    molecule2: Molecule,
    align: bool = True
) -> float:
    """Compute root-mean-square deviation between two structures.

    Args:
        molecule1: First structure
        molecule2: Second structure
        align: Whether to align structures first

    Returns:
        rmsd_value: RMSD (Angstrom)

    Determinism: strict
    """
    pos1 = molecule1.positions - center_of_mass(molecule1)
    pos2 = molecule2.positions - center_of_mass(molecule2)

    if align:
        # Kabsch algorithm for optimal alignment
        H = pos1.T @ pos2
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure right-handed coordinate system
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        pos1_aligned = pos1 @ R
    else:
        pos1_aligned = pos1

    diff = pos1_aligned - pos2
    rmsd_value = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    return rmsd_value


def rmsf(trajectory: Trajectory) -> np.ndarray:
    """Compute root-mean-square fluctuation per atom.

    Args:
        trajectory: MD trajectory

    Returns:
        rmsf_values: RMSF per atom (Angstrom)

    Determinism: strict
    """
    n_atoms = trajectory.frames[0].n_atoms
    n_frames = trajectory.n_frames

    # Compute average positions
    avg_positions = np.zeros((n_atoms, 3))
    for frame in trajectory.frames:
        avg_positions += frame.positions
    avg_positions /= n_frames

    # Compute fluctuations
    fluctuations = np.zeros(n_atoms)
    for frame in trajectory.frames:
        diff = frame.positions - avg_positions
        fluctuations += np.sum(diff**2, axis=1)

    rmsf_values = np.sqrt(fluctuations / n_frames)

    return rmsf_values


def diffusion_coefficient(trajectory: Trajectory) -> float:
    """Compute diffusion coefficient from mean squared displacement.

    D = lim_{t->inf} <|r(t) - r(0)|^2> / (6*t)

    Args:
        trajectory: MD trajectory

    Returns:
        D: Diffusion coefficient (cm²/s)

    Determinism: strict
    """
    # Compute center of mass for each frame
    com_trajectory = []
    for frame in trajectory.frames:
        com_trajectory.append(center_of_mass(frame))
    com_trajectory = np.array(com_trajectory)

    # Mean squared displacement
    msd = np.sum((com_trajectory - com_trajectory[0])**2, axis=1)

    # Linear fit to get slope
    times_ps = trajectory.times / 1000.0  # Convert fs to ps
    if len(times_ps) > 1:
        slope = np.polyfit(times_ps, msd, 1)[0]
        D = slope / 6.0

        # Convert from Angstrom²/ps to cm²/s
        D *= 1e-8 / 1e-12  # = 1e4
        return D
    else:
        return 0.0


def rdf(
    trajectory: Trajectory,
    atom_type_1: str = "O",
    atom_type_2: str = "H",
    r_max: float = 10.0,
    n_bins: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute radial distribution function.

    Args:
        trajectory: MD trajectory
        atom_type_1: First atom type
        atom_type_2: Second atom type
        r_max: Maximum distance (Angstrom)
        n_bins: Number of bins

    Returns:
        r: Distance array (Angstrom)
        g_r: Radial distribution function

    Determinism: strict
    """
    # Find atoms of each type
    frame0 = trajectory.frames[0]
    indices_1 = [i for i, atom in enumerate(frame0.atoms) if atom.element == atom_type_1]
    indices_2 = [i for i, atom in enumerate(frame0.atoms) if atom.element == atom_type_2]

    # Initialize histogram
    hist = np.zeros(n_bins)
    r_bins = np.linspace(0, r_max, n_bins + 1)
    dr = r_bins[1] - r_bins[0]

    # Accumulate over trajectory
    for frame in trajectory.frames:
        for i in indices_1:
            for j in indices_2:
                if i != j:
                    r = np.linalg.norm(frame.positions[j] - frame.positions[i])
                    if r < r_max:
                        bin_idx = int(r / dr)
                        if bin_idx < n_bins:
                            hist[bin_idx] += 1

    # Normalize
    n_frames = trajectory.n_frames
    n_pairs = len(indices_1) * len(indices_2)

    r = (r_bins[:-1] + r_bins[1:]) / 2
    volume_shell = 4 * np.pi * r**2 * dr

    # Density (assuming box volume)
    box_volume = 1000.0  # Placeholder
    density = n_pairs / box_volume

    g_r = hist / (n_frames * n_pairs * density * volume_shell)

    return r, g_r


def hydrogen_bonds(
    trajectory: Trajectory,
    donor_acceptor_distance: float = 3.5,
    angle_cutoff: float = 30.0
) -> List[Tuple[int, int, int]]:
    """Find hydrogen bonds in trajectory.

    Args:
        trajectory: MD trajectory
        donor_acceptor_distance: Max D-A distance (Angstrom)
        angle_cutoff: Max D-H-A angle deviation from 180° (degrees)

    Returns:
        hbonds: List of (donor, hydrogen, acceptor) indices

    Determinism: strict
    """
    # Simplified: find O-H...O patterns
    hbonds = []

    frame = trajectory.frames[0]

    # Find O and H atoms
    O_indices = [i for i, atom in enumerate(frame.atoms) if atom.element == 'O']
    H_indices = [i for i, atom in enumerate(frame.atoms) if atom.element == 'H']

    for donor in O_indices:
        for hydrogen in H_indices:
            for acceptor in O_indices:
                if donor != acceptor:
                    d_h = np.linalg.norm(frame.positions[hydrogen] - frame.positions[donor])
                    h_a = np.linalg.norm(frame.positions[acceptor] - frame.positions[hydrogen])
                    d_a = np.linalg.norm(frame.positions[acceptor] - frame.positions[donor])

                    if d_h < 1.2 and d_a < donor_acceptor_distance:
                        # Check angle
                        v1 = frame.positions[hydrogen] - frame.positions[donor]
                        v2 = frame.positions[acceptor] - frame.positions[hydrogen]

                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi

                        if abs(angle - 180) < angle_cutoff:
                            hbonds.append((donor, hydrogen, acceptor))

    return hbonds


# ============================================================================
# Domain Registration
# ============================================================================

class MolecularOperations:
    """Molecular domain operations."""

    # Loading & Conversion
    load_smiles = staticmethod(load_smiles)
    load_xyz = staticmethod(load_xyz)
    load_pdb = staticmethod(load_pdb)
    to_smiles = staticmethod(to_smiles)
    to_xyz = staticmethod(to_xyz)
    generate_3d = staticmethod(generate_3d)

    # Molecular Properties
    molecular_weight = staticmethod(molecular_weight)
    molecular_formula = staticmethod(molecular_formula)
    center_of_mass = staticmethod(center_of_mass)
    moment_of_inertia = staticmethod(moment_of_inertia)
    dipole_moment = staticmethod(dipole_moment)
    find_rings = staticmethod(find_rings)

    # Force Field Calculations
    compute_energy = staticmethod(compute_energy)
    compute_forces = staticmethod(compute_forces)
    bond_energy = staticmethod(bond_energy)
    angle_energy = staticmethod(angle_energy)
    vdw_energy = staticmethod(vdw_energy)
    electrostatic_energy = staticmethod(electrostatic_energy)

    # Geometry Optimization
    optimize_geometry = staticmethod(optimize_geometry)
    optimize_constrained = staticmethod(optimize_constrained)

    # Conformer Generation
    generate_conformers = staticmethod(generate_conformers)
    cluster_conformers = staticmethod(cluster_conformers)

    # Molecular Dynamics
    velocity_verlet = staticmethod(velocity_verlet)
    langevin_integrator = staticmethod(langevin_integrator)
    md_simulate = staticmethod(md_simulate)

    # Trajectory Analysis
    rmsd = staticmethod(rmsd)
    rmsf = staticmethod(rmsf)
    diffusion_coefficient = staticmethod(diffusion_coefficient)
    rdf = staticmethod(rdf)
    hydrogen_bonds = staticmethod(hydrogen_bonds)


# Create domain instance
molecular = MolecularOperations()


__all__ = [
    'Atom', 'Bond', 'Molecule', 'Trajectory', 'ForceField',
    'load_smiles', 'load_xyz', 'load_pdb', 'to_smiles', 'to_xyz', 'generate_3d',
    'molecular_weight', 'molecular_formula', 'center_of_mass', 'moment_of_inertia',
    'dipole_moment', 'find_rings',
    'compute_energy', 'compute_forces', 'bond_energy', 'angle_energy',
    'vdw_energy', 'electrostatic_energy',
    'optimize_geometry', 'optimize_constrained',
    'generate_conformers', 'cluster_conformers',
    'velocity_verlet', 'langevin_integrator', 'md_simulate',
    'rmsd', 'rmsf', 'diffusion_coefficient', 'rdf', 'hydrogen_bonds',
    'molecular', 'MolecularOperations'
]
