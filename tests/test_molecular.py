"""Tests for molecular domain - molecular dynamics, force fields, and simulations."""

import pytest
import numpy as np
from morphogen.stdlib import molecular


class TestAtomCreation:
    """Test atom creation and basic properties."""

    def test_atom_from_element(self):
        """Test creating atoms from element symbols."""
        h = molecular.Atom.from_element('H')
        assert h.element == 'H'
        assert h.atomic_number == 1
        assert abs(h.mass - 1.008) < 0.01
        assert h.charge == 0.0

        c = molecular.Atom.from_element('C')
        assert c.element == 'C'
        assert c.atomic_number == 6
        assert abs(c.mass - 12.011) < 0.01

    def test_atom_with_charge(self):
        """Test creating atoms with charges."""
        o_minus = molecular.Atom.from_element('O', charge=-1.0)
        assert o_minus.charge == -1.0

        na_plus = molecular.Atom.from_element('Na', charge=1.0)
        assert na_plus.charge == 1.0

    def test_invalid_element(self):
        """Test that invalid element symbols raise errors."""
        with pytest.raises(ValueError, match="Unknown element"):
            molecular.Atom.from_element('Xx')


class TestMoleculeCreation:
    """Test molecule creation and structure."""

    def test_water_molecule(self):
        """Test creating a water molecule (H2O)."""
        # Create atoms
        o = molecular.Atom.from_element('O')
        h1 = molecular.Atom.from_element('H')
        h2 = molecular.Atom.from_element('H')
        atoms = [o, h1, h2]

        # Create bonds (O-H bonds)
        bonds = [
            molecular.Bond(atom1=0, atom2=1, order=1),
            molecular.Bond(atom1=0, atom2=2, order=1)
        ]

        # Create positions
        positions = np.array([
            [0.0, 0.0, 0.0],     # O
            [0.96, 0.0, 0.0],    # H1
            [-0.24, 0.93, 0.0]   # H2
        ])

        mol = molecular.Molecule(atoms=atoms, bonds=bonds, positions=positions)

        assert mol.num_atoms() == 3
        assert len(mol.bonds) == 2
        assert mol.positions.shape == (3, 3)

    def test_methane_molecule(self):
        """Test creating methane (CH4)."""
        atoms = [molecular.Atom.from_element('C')] + [molecular.Atom.from_element('H') for _ in range(4)]

        # Tetrahedral bonds
        bonds = [
            molecular.Bond(0, 1, order=1),
            molecular.Bond(0, 2, order=1),
            molecular.Bond(0, 3, order=1),
            molecular.Bond(0, 4, order=1)
        ]

        # Tetrahedral geometry (approximate)
        positions = np.array([
            [0.0, 0.0, 0.0],      # C
            [0.63, 0.63, 0.63],   # H
            [-0.63, -0.63, 0.63], # H
            [0.63, -0.63, -0.63], # H
            [-0.63, 0.63, -0.63]  # H
        ])

        mol = molecular.Molecule(atoms=atoms, bonds=bonds, positions=positions)
        assert mol.num_atoms() == 5
        assert len(mol.bonds) == 4

    def test_double_bond(self):
        """Test creating molecule with double bond (ethylene)."""
        atoms = [molecular.Atom.from_element('C'), molecular.Atom.from_element('C')]
        bonds = [molecular.Bond(0, 1, order=2)]  # C=C double bond
        positions = np.array([[0.0, 0.0, 0.0], [1.33, 0.0, 0.0]])

        mol = molecular.Molecule(atoms=atoms, bonds=bonds, positions=positions)
        assert mol.bonds[0].order == 2


class TestMolecularProperties:
    """Test molecular property calculations."""

    def test_molecular_mass(self):
        """Test calculating molecular mass."""
        # Water: H2O = 2*1.008 + 15.999 ≈ 18.015
        atoms = [
            molecular.Atom.from_element('O'),
            molecular.Atom.from_element('H'),
            molecular.Atom.from_element('H')
        ]
        positions = np.zeros((3, 3))
        mol = molecular.Molecule(atoms=atoms, bonds=[], positions=positions)

        mass = mol.total_mass()
        assert abs(mass - 18.015) < 0.1

    def test_center_of_mass(self):
        """Test center of mass calculation."""
        # Simple linear molecule with equal masses
        atoms = [molecular.Atom.from_element('C'), molecular.Atom.from_element('C')]
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        mol = molecular.Molecule(atoms=atoms, bonds=[], positions=positions)

        com = mol.center_of_mass()
        assert np.allclose(com, [1.0, 0.0, 0.0], atol=0.01)

    def test_moment_of_inertia(self):
        """Test moment of inertia calculation."""
        # Diatomic molecule along x-axis
        atoms = [molecular.Atom.from_element('O'), molecular.Atom.from_element('O')]
        positions = np.array([[-0.6, 0.0, 0.0], [0.6, 0.0, 0.0]])
        mol = molecular.Molecule(atoms=atoms, bonds=[], positions=positions)

        inertia = mol.moment_of_inertia()
        # For linear molecule along x, Ixx should be ~0, Iyy = Izz > 0
        assert inertia[0, 0] < 0.1  # Ixx ≈ 0
        assert inertia[1, 1] > 0    # Iyy > 0
        assert np.allclose(inertia[1, 1], inertia[2, 2], rtol=0.01)


class TestForceFields:
    """Test force field calculations."""

    def test_lennard_jones_potential(self):
        """Test Lennard-Jones potential calculation."""
        epsilon = 0.997  # kJ/mol
        sigma = 3.4      # Angstroms
        r = np.array([2**(1/6) * sigma])  # Minimum of LJ potential at r = 2^(1/6)*sigma

        energy = molecular.lennard_jones_potential(r, epsilon, sigma)

        # At r = sigma, energy ≈ 0
        # At r = 2^(1/6) * sigma ≈ 3.82, energy = -epsilon
        assert len(energy) == 1
        assert energy[0] < 0  # Attractive at this distance (minimum)
        assert np.isclose(energy[0], -epsilon, rtol=1e-5)  # Should be at minimum

    def test_harmonic_bond_energy(self):
        """Test harmonic bond energy calculation."""
        # Equilibrium bond length
        r_eq = 1.0
        k = 500.0  # Force constant

        # At equilibrium: energy = 0
        energy_eq = molecular.harmonic_bond_energy(r_eq, r_eq, k)
        assert abs(energy_eq) < 1e-10

        # Stretched bond: energy > 0
        r_stretched = 1.1
        energy_stretched = molecular.harmonic_bond_energy(r_stretched, r_eq, k)
        assert energy_stretched > 0

        # Expected: E = 0.5 * k * (1.1 - 1.0)^2 = 0.5 * 500 * 0.01 = 2.5
        assert abs(energy_stretched - 2.5) < 0.01

    def test_coulomb_energy(self):
        """Test Coulomb electrostatic energy."""
        q1 = 1.0   # +1 charge
        q2 = -1.0  # -1 charge
        r = 3.0    # Angstroms

        energy = molecular.coulomb_energy(q1, q2, r)

        # Opposite charges = attractive (negative energy)
        assert energy < 0

        # Same charges = repulsive
        energy_repulsive = molecular.coulomb_energy(1.0, 1.0, r)
        assert energy_repulsive > 0


class TestGeometryOptimization:
    """Test molecular geometry optimization."""

    def test_optimize_diatomic(self):
        """Test optimization of simple diatomic molecule."""
        # Start with non-equilibrium H2 distance
        atoms = [molecular.Atom.from_element('H'), molecular.Atom.from_element('H')]
        bonds = [molecular.Bond(0, 1, order=1)]
        positions = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])  # Too long

        mol = molecular.Molecule(atoms=atoms, bonds=bonds, positions=positions)

        # Optimize geometry
        optimized = molecular.optimize_geometry(mol, method='steepest_descent',
                                                max_iter=100, tol=1e-4)

        # Check that H-H distance is closer to equilibrium (~0.74 Angstroms)
        final_distance = np.linalg.norm(optimized.positions[1] - optimized.positions[0])
        assert 0.6 < final_distance < 1.0  # Should be closer to ~0.74
        assert final_distance < 1.5  # Should have moved from initial position

    def test_optimize_water_geometry(self):
        """Test water molecule geometry optimization."""
        atoms = [
            molecular.Atom.from_element('O'),
            molecular.Atom.from_element('H'),
            molecular.Atom.from_element('H')
        ]
        bonds = [
            molecular.Bond(0, 1, order=1),
            molecular.Bond(0, 2, order=1)
        ]
        # Start with linear configuration (non-equilibrium)
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0]
        ])

        mol = molecular.Molecule(atoms=atoms, bonds=bonds, positions=positions)
        optimized = molecular.optimize_geometry(mol, max_iter=200, tol=1e-4)

        # Water should bend (not remain linear)
        # Check that H atoms are not collinear with O
        v1 = optimized.positions[1] - optimized.positions[0]
        v2 = optimized.positions[2] - optimized.positions[0]
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angle_deg = np.degrees(angle)

        # Water angle should be ~104-105 degrees, not 180
        assert angle_deg < 170  # Should have bent from linear


class TestMolecularDynamics:
    """Test molecular dynamics simulation."""

    def test_md_integration_conserves_atoms(self):
        """Test that MD simulation doesn't change number of atoms."""
        atoms = [molecular.Atom.from_element('H'), molecular.Atom.from_element('H')]
        bonds = [molecular.Bond(0, 1, order=1)]
        positions = np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
        velocities = np.zeros((2, 3))

        mol = molecular.Molecule(atoms=atoms, bonds=bonds, positions=positions,
                                 velocities=velocities)

        # Run short MD simulation
        trajectory = molecular.run_md(mol, dt=0.5, steps=10, ensemble='nve')

        assert len(trajectory) == 10
        for frame in trajectory:
            assert frame.num_atoms() == 2

    def test_md_velocity_verlet(self):
        """Test MD with Velocity Verlet integrator."""
        # Single particle in harmonic potential
        atoms = [molecular.Atom.from_element('C')]
        positions = np.array([[1.0, 0.0, 0.0]])
        velocities = np.array([[0.0, 0.0, 0.0]])

        mol = molecular.Molecule(atoms=atoms, bonds=[], positions=positions,
                                 velocities=velocities)

        # Run MD (will just move under any applied forces)
        trajectory = molecular.run_md(mol, dt=0.1, steps=5, ensemble='nve')

        assert len(trajectory) >= 5

    def test_md_temperature_control(self):
        """Test NVT ensemble with temperature control."""
        # Water molecule
        atoms = [
            molecular.Atom.from_element('O'),
            molecular.Atom.from_element('H'),
            molecular.Atom.from_element('H')
        ]
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0]
        ])
        velocities = np.random.randn(3, 3) * 0.1
        bonds = [molecular.Bond(0, 1, 1), molecular.Bond(0, 2, 1)]

        mol = molecular.Molecule(atoms=atoms, bonds=bonds, positions=positions,
                                 velocities=velocities)

        # Run NVT MD at 300K
        trajectory = molecular.run_md(mol, dt=0.5, steps=20, ensemble='nvt',
                                      temperature=300.0)

        # Check temperature is approximately maintained
        temps = [molecular.calculate_temperature(frame) for frame in trajectory]
        mean_temp = np.mean(temps)

        # Temperature should be roughly 300K (allow wide range for short simulation)
        assert 100 < mean_temp < 500


class TestTrajectoryAnalysis:
    """Test trajectory analysis functions."""

    def test_rmsd_calculation(self):
        """Test RMSD (root mean square deviation) calculation."""
        # Two identical structures
        atoms = [molecular.Atom.from_element('C') for _ in range(3)]
        pos1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        pos2 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)

        mol1 = molecular.Molecule(atoms=atoms, bonds=[], positions=pos1)
        mol2 = molecular.Molecule(atoms=atoms, bonds=[], positions=pos2)

        rmsd = molecular.calculate_rmsd(mol1, mol2)
        assert abs(rmsd) < 1e-10  # Should be zero for identical structures

        # Translated structure
        pos3 = pos1 + np.array([1.0, 0.0, 0.0])
        mol3 = molecular.Molecule(atoms=atoms, bonds=[], positions=pos3)

        rmsd_translated = molecular.calculate_rmsd(mol1, mol3)
        assert rmsd_translated > 0  # Should be non-zero

    def test_radius_of_gyration(self):
        """Test radius of gyration calculation."""
        # Linear molecule
        atoms = [molecular.Atom.from_element('C') for _ in range(3)]
        positions = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        mol = molecular.Molecule(atoms=atoms, bonds=[], positions=positions)

        rg = molecular.radius_of_gyration(mol)
        assert rg > 0  # Should be positive

        # Compact structure should have smaller Rg
        positions_compact = np.array([[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]], dtype=float)
        mol_compact = molecular.Molecule(atoms=atoms, bonds=[], positions=positions_compact)
        rg_compact = molecular.radius_of_gyration(mol_compact)

        assert rg_compact < rg  # Compact structure has smaller Rg


class TestConformers:
    """Test conformer generation and searching."""

    def test_generate_conformers(self):
        """Test generation of multiple conformers."""
        # Ethane (C2H6) - can have different rotational conformers
        atoms = [
            molecular.Atom.from_element('C'),
            molecular.Atom.from_element('C')
        ] + [molecular.Atom.from_element('H') for _ in range(6)]

        bonds = [
            molecular.Bond(0, 1, order=1),  # C-C
        ] + [molecular.Bond(0, i, order=1) for i in range(2, 5)] + \
            [molecular.Bond(1, i, order=1) for i in range(5, 8)]

        # Initial geometry
        positions = np.random.randn(8, 3)
        mol = molecular.Molecule(atoms=atoms, bonds=bonds, positions=positions)

        # Generate conformers
        conformers = molecular.generate_conformers(mol, num_conformers=5, seed=42)

        assert len(conformers) == 5
        # All should have same number of atoms
        assert all(c.num_atoms() == 8 for c in conformers)

        # Conformers should have different geometries
        rmsd = molecular.calculate_rmsd(conformers[0], conformers[1])
        assert rmsd > 0.01  # Should be different


class TestDeterminism:
    """Test deterministic behavior."""

    def test_md_determinism(self):
        """Test that MD simulation with same seed gives identical results."""
        atoms = [molecular.Atom.from_element('H'), molecular.Atom.from_element('H')]
        bonds = [molecular.Bond(0, 1, order=1)]
        positions = np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
        velocities = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])

        mol = molecular.Molecule(atoms=atoms, bonds=bonds, positions=positions.copy(),
                                 velocities=velocities.copy())

        # Run twice with same parameters
        traj1 = molecular.run_md(mol, dt=0.5, steps=10, ensemble='nve', seed=42)

        mol2 = molecular.Molecule(atoms=atoms, bonds=bonds, positions=positions.copy(),
                                  velocities=velocities.copy())
        traj2 = molecular.run_md(mol2, dt=0.5, steps=10, ensemble='nve', seed=42)

        # Trajectories should be identical
        for frame1, frame2 in zip(traj1, traj2):
            assert np.allclose(frame1.positions, frame2.positions, atol=1e-12)

    def test_conformer_determinism(self):
        """Test that conformer generation with same seed is deterministic."""
        atoms = [molecular.Atom.from_element('C') for _ in range(4)]
        bonds = [molecular.Bond(i, i+1, order=1) for i in range(3)]
        positions = np.random.randn(4, 3)
        mol = molecular.Molecule(atoms=atoms, bonds=bonds, positions=positions.copy())

        conf1 = molecular.generate_conformers(mol, num_conformers=3, seed=123)

        mol2 = molecular.Molecule(atoms=atoms, bonds=bonds, positions=positions.copy())
        conf2 = molecular.generate_conformers(mol2, num_conformers=3, seed=123)

        # Should generate identical conformers
        for c1, c2 in zip(conf1, conf2):
            assert np.allclose(c1.positions, c2.positions, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
