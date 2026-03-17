---
title: "Chemistry Pipeline — Molecular → Thermo → Kinetics"
type: guide
beth_topics:
  - morphogen
  - chemistry
  - molecular
  - thermo
  - kinetics
  - reaction-engineering
  - pipeline
---

# Chemistry Pipeline

This guide walks a single worked example through three connected domains:

1. **`molecular`** — compute molecular geometry and energy
2. **`thermo`** — derive thermodynamic state (enthalpy, Gibbs free energy, equilibrium)
3. **`kinetics`** — simulate reaction rates and concentration evolution

The domains pass plain numeric values — there is no magic coupling object.
Each output is a number or array that the next domain accepts directly.

---

## The system: ethylene hydrogenation

C₂H₄ + H₂ → C₂H₆  (ethylene + hydrogen → ethane)

This is a textbook heterogeneous catalysis reaction: fast at 300 K with a
Pd catalyst, used industrially to remove traces of ethylene from polyethylene.
We will estimate the rate constant, run a batch reactor, and plot the
concentration curves.

---

## Step 1 — Molecular: geometry and baseline energy

```python
import numpy as np
from morphogen.stdlib.molecular import (
    load_smiles, optimize_geometry, compute_energy, molecular_formula,
)

# Load reactants and product
ethylene = optimize_geometry(load_smiles("C=C"),  force_field="uff")
hydrogen = optimize_geometry(load_smiles("[H][H]"), force_field="uff")
ethane   = optimize_geometry(load_smiles("CC"),   force_field="uff")

# Force field energies (kcal/mol) — relative, not absolute heats of formation
E_react = compute_energy(ethylene) + compute_energy(hydrogen)
E_prod  = compute_energy(ethane)
dE_uff  = E_prod - E_react

print("Reactants:", molecular_formula(ethylene), "+", molecular_formula(hydrogen))
print("Product:  ", molecular_formula(ethane))
print(f"ΔE (UFF): {dE_uff:.3f} kcal/mol  (negative = exothermic)")
```

> UFF energies are force-field estimates, not thermochemical values. Step 2
> uses literature data for real ΔH. The molecular step here demonstrates how
> geometry feeds into the pipeline — in a production workflow you would use
> `qchem.compute_energy` (semi-empirical/DFT) for ΔE, or pass RDKit-computed
> enthalpies directly.

---

## Step 2 — Thermo: equilibrium and driving force

```python
from morphogen.stdlib.thermo import (
    enthalpy_of_reaction, gibbs_free_energy, equilibrium_constant,
    heat_capacity,
)

T = 350.0      # K — mild hydrogenation temperature
p = 1e5        # Pa (1 atm)

# Standard enthalpy of hydrogenation at 298 K: −136.6 kJ/mol
dH = -136_600.0     # J/mol

# Standard entropy change (C2H6 formation from elements, rough estimate)
# ΔS ≈ S°(ethane) - S°(ethylene) - S°(H2) ≈ -120 J/(mol·K)
dS = -120.0         # J/(mol·K)

dG = gibbs_free_energy(T, enthalpy=dH, entropy=dS)
K_eq = equilibrium_constant(dG, T)

print(f"T = {T} K")
print(f"ΔH = {dH/1000:.1f} kJ/mol  (exothermic)")
print(f"ΔG = {dG/1000:.1f} kJ/mol")
print(f"K_eq = {K_eq:.2e}  (>>1 = strongly favours product)")

# Heat capacity of ethane (to estimate heat of reaction at 350 K)
Cp_ethane = heat_capacity("C2H6", T)
print(f"Cp(C2H6, {T} K) ≈ {Cp_ethane:.1f} J/(mol·K)")
```

---

## Step 3 — Kinetics: rate constant and reactor simulation

With the thermodynamic picture established, estimate the kinetic parameters
and simulate a batch reactor.

```python
from morphogen.stdlib.kinetics import (
    arrhenius, integrate_ode, batch_reactor,
    create_reaction, cstr,
)

# Arrhenius parameters for Pd-catalysed hydrogenation (literature estimates)
# A ≈ 10^10 s⁻¹, Ea ≈ 45 kJ/mol
A_factor = 1.0e10    # pre-exponential (s⁻¹, first-order in this simplified model)
Ea       = 45_000.0  # J/mol

k_350K = arrhenius(temp=350.0, A=A_factor, Ea=Ea)
k_400K = arrhenius(temp=400.0, A=A_factor, Ea=Ea)
print(f"k(350 K) = {k_350K:.3e} s⁻¹")
print(f"k(400 K) = {k_400K:.3e} s⁻¹  ({k_400K/k_350K:.1f}× faster)")
```

### Batch reactor — concentration profiles

```python
# Define the reaction: C2H4 → C2H6  (H2 excess, pseudo-first-order)
rxn = create_reaction(
    reactants={"C2H4": 1},
    products={"C2H6": 1},
    A=A_factor,
    Ea=Ea,
    reversible=False,   # K_eq >> 1, so treat as irreversible
)

# Initial concentrations: 0.1 mol/L ethylene, H2 excess
c0 = {"C2H4": 0.1, "C2H6": 0.0}

# Integrate 60 seconds
result = integrate_ode(
    conc_initial=c0,
    reactions=[rxn],
    temp=350.0,
    time=60.0,
    time_points=np.linspace(0, 60, 600),
)

# result: dict of species → np.ndarray over time
t_half_idx = np.argmin(np.abs(result["C2H4"] - 0.05))   # 50% conversion
t_90_idx   = np.argmin(np.abs(result["C2H4"] - 0.01))   # 90% conversion

print(f"\nt₁/₂ ≈ {result['t'][t_half_idx]:.2f} s")
print(f"t₉₀ ≈  {result['t'][t_90_idx]:.2f} s")
print(f"[C2H4] final: {result['C2H4'][-1]:.5f} mol/L")
print(f"[C2H6] final: {result['C2H6'][-1]:.5f} mol/L")
```

### CSTR — steady-state conversion

```python
# Continuous stirred-tank reactor: 1 L/min feed, 5 L vessel
feed = {"C2H4": 0.1, "C2H6": 0.0}   # mol/L
steady_state = cstr(
    feed_conc=feed,
    feed_flow=1/60,    # L/s
    volume=5.0,        # L
    reactions=[rxn],
    temp=350.0,
)

conversion = 1 - steady_state["C2H4"] / feed["C2H4"]
print(f"\nCSTR steady-state conversion: {conversion*100:.1f}%")
```

---

## Full pipeline summary

```
load_smiles("C=C")
    → optimize_geometry()     [molecular: geometry, energy]
        → compute_energy()    [molecular: ΔE estimate]

enthalpy_of_reaction()        [thermo: ΔH, literature values]
gibbs_free_energy()           [thermo: ΔG]
equilibrium_constant()        [thermo: K_eq — is conversion feasible?]
heat_capacity()               [thermo: Cp for energy balance]

arrhenius()                   [kinetics: k(T)]
create_reaction()             [kinetics: define stoichiometry + rate law]
integrate_ode()               [kinetics: batch concentration profiles]
cstr()                        [kinetics: steady-state design]
```

Each step feeds numbers into the next. No Morphogen-specific coupling objects —
just Python values and NumPy arrays passing between domains.

---

## See also

- [`docs/usage/molecular.md`](molecular.md) — full molecular domain reference
- [`examples/showcase_molecular_reactor.py`](/examples/showcase_molecular_reactor.py) — extended reactor demo
- [`docs/use-cases/chemistry-unified-framework.md`](/docs/use-cases/chemistry-unified-framework.md) — broader chemistry use-case overview
- [`morphogen/stdlib/thermo.py`](/morphogen/stdlib/thermo.py) — thermodynamics source
- [`morphogen/stdlib/kinetics.py`](/morphogen/stdlib/kinetics.py) — kinetics source
