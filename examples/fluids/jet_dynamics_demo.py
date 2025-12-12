"""Fluid Jet Dynamics Demo

This example demonstrates the fluid_jet domain capabilities:
- Creating jets from tubes
- Computing Reynolds numbers
- Calculating centerline velocity decay
- Visualizing jet spreading and field distributions

The fluid_jet domain models turbulent free jets, which are important in:
- HVAC systems (air diffusers, ventilation)
- Combustion (fuel injection)
- Aeroacoustics (jet noise)
- Industrial mixing processes
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from morphogen.stdlib import fluid_jet


def demo_single_jet_analysis():
    """Analyze a single turbulent jet from a circular tube."""
    print("=" * 60)
    print("SINGLE JET ANALYSIS")
    print("=" * 60)
    print()

    # Create a jet: 50mm diameter tube, 0.1 kg/s air flow, 300K
    jet = fluid_jet.jet_from_tube(
        tube_diameter=0.05,           # 50mm diameter
        tube_position=(0.0, 0.0, 0.0),  # Origin
        tube_direction=(1.0, 0.0, 0.0),  # Along x-axis
        m_dot=0.1,                    # 0.1 kg/s mass flow
        T_out=300.0,                  # 300K temperature
        rho=1.2                       # 1.2 kg/m³ air density
    )

    print(f"Jet created:")
    print(f"  Diameter: {jet.diameter:.3f} m")
    print(f"  Exit velocity: {jet.velocity:.2f} m/s")
    print(f"  Momentum flux: {jet.momentum_flux:.2f} kg·m/s²")
    print()

    # Calculate Reynolds number (turbulent flow indicator)
    Re = fluid_jet.jet_reynolds(jet, mu=1.8e-5)  # Air viscosity at 300K
    print(f"Reynolds number: {Re:.0f}")
    if Re > 4000:
        print("  → Turbulent flow (Re > 4000)")
    elif Re > 2300:
        print("  → Transitional flow (2300 < Re < 4000)")
    else:
        print("  → Laminar flow (Re < 2300)")
    print()

    # Analyze velocity decay along centerline
    print("Centerline velocity decay:")
    distances = [0.5, 1.0, 2.0, 5.0, 10.0]  # Meters downstream

    for dist in distances:
        vel = fluid_jet.jet_centerline_velocity(jet, distance=dist)
        width = fluid_jet.jet_spreading_width(jet, distance=dist)
        decay_percent = (1.0 - vel / jet.velocity) * 100

        print(f"  x = {dist:5.1f}m: v = {vel:6.2f} m/s "
              f"(decay: {decay_percent:4.1f}%), "
              f"width = {width:.3f}m")
    print()

    # Plot velocity profile
    distances_fine = np.linspace(0.1, 10.0, 100)
    velocities = [fluid_jet.jet_centerline_velocity(jet, d)
                  for d in distances_fine]
    widths = [fluid_jet.jet_spreading_width(jet, d)
              for d in distances_fine]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Velocity decay
    ax1.plot(distances_fine, velocities, 'b-', linewidth=2)
    ax1.axhline(jet.velocity, color='r', linestyle='--',
                label='Exit velocity')
    ax1.set_xlabel('Distance from exit (m)')
    ax1.set_ylabel('Centerline velocity (m/s)')
    ax1.set_title('Jet Velocity Decay')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Jet spreading
    ax2.plot(distances_fine, widths, 'g-', linewidth=2)
    ax2.axhline(jet.diameter, color='r', linestyle='--',
                label='Exit diameter')
    ax2.set_xlabel('Distance from exit (m)')
    ax2.set_ylabel('Jet width (m)')
    ax2.set_title('Jet Spreading (Half-Width)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('jet_dynamics_single.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: jet_dynamics_single.png")
    print()

    return jet


def demo_radial_jet_array():
    """Demonstrate multiple jets in a radial configuration (like a burner)."""
    print("=" * 60)
    print("RADIAL JET ARRAY")
    print("=" * 60)
    print()

    # Create 8 jets in a circle (like a gas burner)
    jet_array = fluid_jet.create_jet_array_radial(
        n_jets=8,                  # 8 jets around circle
        radius=0.2,                # 20cm radius
        jet_diameter=0.01,         # 10mm diameter each
        m_dot_per_jet=0.01,        # 0.01 kg/s per jet
        temperature=400.0,         # 400K (hot combustion air)
        height=0.0,                # At z=0 plane
        angle_inward=15.0          # 15° angled inward
    )

    print(f"Created radial jet array:")
    print(f"  Number of jets: {jet_array.count}")
    print(f"  Total mass flow: {jet_array.total_flow:.4f} kg/s")
    print(f"  Pattern radius: 0.2 m")
    print(f"  Inward angle: 15°")
    print()

    # Generate 2D field visualization (top-down view)
    field_vectors = fluid_jet.jet_field_2d(
        jet_array=jet_array,
        grid_size=(200, 200),
        grid_bounds=(-0.5, 0.5, -0.5, 0.5),  # 1m x 1m region
        decay=0.15  # Decay rate for visualization
    )

    # Compute velocity magnitude for visualization
    field_magnitude = np.sqrt(field_vectors[:, :, 0]**2 + field_vectors[:, :, 1]**2)

    print("Generated 2D jet field (200x200 grid)")
    print(f"  Max velocity: {field_magnitude.max():.4f} m/s")
    print(f"  Mean velocity: {field_magnitude.mean():.4f} m/s")
    print()

    # Visualize the radial pattern
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Field intensity (velocity magnitude)
    im1 = ax1.imshow(field_magnitude, extent=[-0.5, 0.5, -0.5, 0.5],
                     origin='lower', cmap='hot', interpolation='bilinear')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Jet Velocity Magnitude (Top View)')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1, label='Velocity (m/s)')

    # Mark jet positions
    for jet in jet_array.jets:
        ax1.plot(jet.position[0], jet.position[1], 'wo',
                markersize=8, markeredgecolor='blue', markeredgewidth=2)

    # Contour plot with velocity vectors
    contours = ax2.contourf(field_magnitude, levels=20, extent=[-0.5, 0.5, -0.5, 0.5],
                            cmap='viridis', origin='lower')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Jet Field Contours')
    ax2.set_aspect('equal')
    plt.colorbar(contours, ax=ax2, label='Intensity')

    # Mark jet positions
    for jet in jet_array.jets:
        ax2.plot(jet.position[0], jet.position[1], 'ro',
                markersize=8, markeredgecolor='white', markeredgewidth=2)

    plt.tight_layout()
    plt.savefig('jet_dynamics_radial.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: jet_dynamics_radial.png")
    print()


def demo_jet_entrainment():
    """Calculate entrainment rate (surrounding fluid drawn into jet)."""
    print("=" * 60)
    print("JET ENTRAINMENT ANALYSIS")
    print("=" * 60)
    print()

    # Create a high-velocity jet
    jet = fluid_jet.jet_from_tube(
        tube_diameter=0.03,           # 30mm
        tube_position=(0.0, 0.0, 0.0),
        tube_direction=(0.0, 0.0, 1.0),  # Vertical
        m_dot=0.05,                   # 0.05 kg/s
        T_out=300.0,
        rho=1.2
    )

    print(f"Jet: {jet.diameter*1000:.1f}mm diameter, {jet.velocity:.1f} m/s velocity")
    print()

    # Calculate entrainment into a plume above the jet
    plume_velocity = 5.0   # m/s upward flow
    plume_density = 1.1    # kg/m³ (slightly lighter due to heating)

    entrainment = fluid_jet.jet_entrainment(
        jet=jet,
        plume_velocity=plume_velocity,
        plume_density=plume_density,
        model="empirical"
    )

    print(f"Plume conditions:")
    print(f"  Velocity: {plume_velocity} m/s")
    print(f"  Density: {plume_density} kg/m³")
    print()
    print(f"Entrainment rate: {entrainment:.6f} kg/s")
    print(f"  (Rate at which surrounding air is drawn into plume)")
    print()

    # Compare to jet mass flow
    entrainment_ratio = entrainment / jet.velocity
    print(f"Entrainment coefficient: {entrainment_ratio:.4f}")
    print()


def main():
    """Run all fluid jet demonstrations."""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  FLUID JET DYNAMICS DEMO".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    print("Demonstrating the fluid_jet domain:")
    print("  • Turbulent jet analysis")
    print("  • Reynolds number calculation")
    print("  • Velocity decay and spreading")
    print("  • Radial jet arrays")
    print("  • Entrainment effects")
    print()

    # Run demonstrations
    jet = demo_single_jet_analysis()
    demo_radial_jet_array()
    demo_jet_entrainment()

    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print()
    print("Applications demonstrated:")
    print("  ✓ HVAC: Air diffuser analysis")
    print("  ✓ Combustion: Radial burner pattern")
    print("  ✓ Mixing: Entrainment calculations")
    print()
    print("The fluid_jet domain enables:")
    print("  • Turbulent jet modeling")
    print("  • Multi-jet configurations")
    print("  • Flow visualization")
    print("  • Engineering design calculations")
    print()
    print("Generated visualizations:")
    print("  • jet_dynamics_single.png")
    print("  • jet_dynamics_radial.png")
    print()


if __name__ == "__main__":
    main()
