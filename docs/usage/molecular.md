---
title: "Molecular Domain — Narrative Guide"
type: guide
beth_topics:
  - morphogen
  - molecular
  - molecular-dynamics
  - force-field
  - trajectory
  - chemistry
---

# Molecular Domain

The `molecular` domain provides a self-contained molecular mechanics engine:
force field evaluation (UFF), geometry optimization, molecular dynamics
(NVE and NVT/Langevin), and trajectory analysis. No external chemistry toolkit
required for the core workflow.

> **RDKit note**: `load_smiles` with 3D coordinate generation is a stub in the
> base installation. For SMILES → 3D conformer generation, install RDKit:
> `conda install -c conda-forge rdkit`. The MD engine, energy evaluation,
> and analysis operators work fully without it.

## Quick start

```python
import numpy as np
from morphogen.stdlib.molecular import (
    load_smiles, optimize_geometry, run_md,
    calculate_temperature, calculate_rmsd, radius_of_gyration,
    molecular_weight, molecular_formula,
)
```

---

## Recipe 1 — Load and inspect a molecule

```python
# Load from SMILES string
mol = load_smiles("CCO")   # ethanol

print(f"Formula:  {molecular_formula(mol)}")
print(f"MW:       {molecular_weight(mol):.2f} g/mol")
print(f"Atoms:    {mol.n_atoms}")
print(f"Positions shape: {mol.positions.shape}")   # (n_atoms, 3), Å
```

For files, use `load_xyz` or `load_pdb`:

```python
from morphogen.stdlib.molecular import load_xyz, load_pdb

mol = load_xyz("molecule.xyz")   # standard XYZ format
mol = load_pdb("protein.pdb")    # PDB format
```

---

## Recipe 2 — Geometry optimisation

`optimize_geometry` minimises the UFF force field energy using steepest descent
(default) or L-BFGS-B.

```python
from morphogen.stdlib.molecular import optimize_geometry, compute_energy

mol = load_smiles("c1ccccc1")   # benzene

e_initial = compute_energy(mol, force_field="uff")
mol_opt = optimize_geometry(
    mol,
    force_field="uff",
    method="lbfgsb",       # "steepest_descent" | "lbfgsb"
    max_iterations=1000,
    convergence=1e-6,
)
e_final = compute_energy(mol_opt, force_field="uff")
print(f"Energy: {e_initial:.4f} → {e_final:.4f} kcal/mol")
print(f"RMSD from initial: {calculate_rmsd(mol, mol_opt):.4f} Å")
```

Energy terms can be inspected individually:

```python
from morphogen.stdlib.molecular import bond_energy, angle_energy, vdw_energy

print(f"bond: {bond_energy(mol_opt):.4f} kcal/mol")
print(f"angle: {angle_energy(mol_opt):.4f} kcal/mol")
print(f"vdW: {vdw_energy(mol_opt):.4f} kcal/mol")
```

---

## Recipe 3 — Molecular dynamics

Run NVE (constant energy) or NVT (constant temperature via Langevin) dynamics.
`run_md` returns a list of `Molecule` frames — one per step.

```python
mol_opt = optimize_geometry(load_smiles("CCO"), force_field="uff")

# NVT: Langevin thermostat at 300 K
frames = run_md(
    mol_opt,
    dt=1.0,             # femtoseconds
    steps=1000,
    ensemble="nvt",
    temperature=300.0,  # Kelvin
    seed=42,
)

print(f"frames: {len(frames)}")

# Sample observables
T_final = calculate_temperature(frames[-1])
Rg_final = radius_of_gyration(frames[-1])
print(f"T(final):  {T_final:.1f} K  (target 300 K)")
print(f"Rg(final): {Rg_final:.3f} Å")
```

NVE (microcanonical) — no thermostat, energy conserved:

```python
frames_nve = run_md(mol_opt, dt=0.5, steps=500, ensemble="nve")
```

---

## Recipe 4 — Trajectory analysis

```python
# Temperature trajectory
temps = [calculate_temperature(f) for f in frames]
print(f"T mean: {np.mean(temps):.1f} K  std: {np.std(temps):.1f} K")

# RMSD drift from start
rmsds = [calculate_rmsd(frames[0], f) for f in frames]
print(f"RMSD at end: {rmsds[-1]:.3f} Å")

# Radius of gyration over time
Rg_traj = [radius_of_gyration(f) for f in frames]
print(f"Rg range: {min(Rg_traj):.3f} – {max(Rg_traj):.3f} Å")
```

For longer simulations, `md_simulate` runs in a single call and stores
every `save_interval`-th frame in a `Trajectory` object with built-in
analysis methods:

```python
from morphogen.stdlib.molecular import md_simulate, rmsf, diffusion_coefficient

traj = md_simulate(
    mol_opt,
    force_field="uff",
    temp=300.0,
    time=10000.0,       # fs
    dt=1.0,
    ensemble="nvt",
    save_interval=10,
)

rmsf_per_atom = rmsf(traj)       # per-atom fluctuation, shape (n_atoms,)
D = diffusion_coefficient(traj)  # Å²/fs
print(f"most mobile atom: {np.argmax(rmsf_per_atom)}, RMSF={rmsf_per_atom.max():.3f} Å")
print(f"diffusion coefficient: {D:.6f} Å²/fs")
```

---

## Recipe 5 — Coupling to thermo and kinetics

Molecular geometry feeds directly into thermodynamic calculations:

```python
from morphogen.stdlib.thermo import enthalpy_of_reaction, gibbs_free_energy, equilibrium_constant
from morphogen.stdlib.kinetics import arrhenius, integrate_ode, create_reaction

# Use molecular weight and energy to estimate reaction parameters
mw = molecular_weight(mol_opt)
e = compute_energy(mol_opt, force_field="uff")

# Estimate activation energy from geometry strain (toy model)
Ea = abs(e) * 4184  # kcal/mol → J/mol (rough)
k_500K = arrhenius(temp=500.0, A=1e12, Ea=Ea)
print(f"Rate constant at 500 K: {k_500K:.3e} s⁻¹")

# Full chemistry pipeline — see docs/usage/chemistry_pipeline.md
```

See [`docs/usage/chemistry_pipeline.md`](chemistry_pipeline.md) for a worked
molecular → thermo → kinetics pipeline.

---

## Full operator reference

| Operator | Returns | Notes |
|----------|---------|-------|
| `load_smiles(smiles, generate_3d)` | `Molecule` | Full 3D generation requires RDKit |
| `load_xyz(filepath)` | `Molecule` | Standard XYZ file |
| `load_pdb(filepath)` | `Molecule` | PDB file |
| `to_smiles(molecule)` | `str` | Canonical SMILES |
| `to_xyz(molecule)` | `str` | XYZ string |
| `molecular_weight(molecule)` | `float` | g/mol |
| `molecular_formula(molecule)` | `str` | e.g. `"C6H6"` |
| `center_of_mass(molecule)` | `np.ndarray (3,)` | Å |
| `moment_of_inertia(molecule)` | `np.ndarray (3,3)` | amu·Å² |
| `compute_energy(molecule, force_field, include_terms)` | `float` | kcal/mol |
| `compute_forces(molecule, force_field)` | `np.ndarray (n,3)` | kcal/mol·Å |
| `optimize_geometry(molecule, force_field, method, max_iterations, convergence)` | `Molecule` | Methods: `"steepest_descent"`, `"lbfgsb"` |
| `run_md(molecule, dt, steps, ensemble, temperature, seed)` | `List[Molecule]` | Ensembles: `"nve"`, `"nvt"` |
| `md_simulate(molecule, ...)` | `Trajectory` | High-level interface with save_interval |
| `calculate_temperature(molecule)` | `float` | K, from kinetic energy |
| `calculate_rmsd(molecule1, molecule2)` | `float` | Å, with optional alignment |
| `radius_of_gyration(molecule)` | `float` | Å |
| `rmsf(trajectory)` | `np.ndarray (n_atoms,)` | Per-atom fluctuation |
| `diffusion_coefficient(trajectory)` | `float` | Å²/fs |
| `bond_energy`, `angle_energy`, `vdw_energy`, `electrostatic_energy` | `float` | kcal/mol, UFF terms |

`Molecule` fields: `atoms` (list of `Atom`), `positions` (Å, shape n×3),
`bonds` (list of `Bond`).

---

## See also

- [`docs/usage/chemistry_pipeline.md`](chemistry_pipeline.md) — molecular → thermo → kinetics worked example
- [`examples/showcase_molecular_reactor.py`](/examples/showcase_molecular_reactor.py) — reactor simulation
- [`docs/specifications/chemistry.md`](/docs/specifications/chemistry.md) — force field spec
- [`morphogen/stdlib/molecular.py`](/morphogen/stdlib/molecular.py) — source with full docstrings
