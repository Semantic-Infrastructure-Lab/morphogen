"""Molecular Visualization Demo

Demonstrates Morphogen's 3D molecular visualization capabilities:
- Ball-and-stick representation
- Space-filling (CPK) representation
- Element-based coloring
- Charge-based coloring
- Orbital isosurface visualization
- Camera animation around molecules

Requirements:
- PyVista for 3D rendering
- NumPy for array operations
"""

import numpy as np
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def create_water_molecule():
    """Create a water molecule (H2O)."""
    from morphogen.stdlib.molecular import Atom, Bond, Molecule

    atoms = [
        Atom.from_element('O'),
        Atom.from_element('H'),
        Atom.from_element('H'),
    ]
    # Water geometry (approximate, bond length ~0.96 A, angle ~104.5 degrees)
    positions = np.array([
        [0.0, 0.0, 0.0],      # O at origin
        [0.96, 0.0, 0.0],     # H1 along x-axis
        [-0.24, 0.93, 0.0],   # H2 at ~104.5 degrees
    ])
    bonds = [Bond(0, 1, 1.0), Bond(0, 2, 1.0)]
    return Molecule(atoms=atoms, bonds=bonds, positions=positions)


def create_methane_molecule():
    """Create a methane molecule (CH4) with tetrahedral geometry."""
    from morphogen.stdlib.molecular import Atom, Bond, Molecule

    atoms = [
        Atom.from_element('C'),
        Atom.from_element('H'),
        Atom.from_element('H'),
        Atom.from_element('H'),
        Atom.from_element('H'),
    ]
    # Tetrahedral geometry
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.09, 0.0, 0.0],
        [-0.36, 1.03, 0.0],
        [-0.36, -0.51, 0.89],
        [-0.36, -0.51, -0.89],
    ])
    bonds = [Bond(0, i, 1.0) for i in range(1, 5)]
    return Molecule(atoms=atoms, bonds=bonds, positions=positions)


def create_ethanol_molecule():
    """Create an ethanol molecule (C2H5OH)."""
    from morphogen.stdlib.molecular import Atom, Bond, Molecule

    atoms = [
        Atom.from_element('C'),   # 0: CH3 carbon
        Atom.from_element('C'),   # 1: CH2 carbon
        Atom.from_element('O'),   # 2: hydroxyl oxygen
        Atom.from_element('H'),   # 3-5: CH3 hydrogens
        Atom.from_element('H'),
        Atom.from_element('H'),
        Atom.from_element('H'),   # 6-7: CH2 hydrogens
        Atom.from_element('H'),
        Atom.from_element('H'),   # 8: OH hydrogen
    ]
    # Approximate ethanol geometry
    positions = np.array([
        [0.0, 0.0, 0.0],          # C1 (CH3)
        [1.54, 0.0, 0.0],         # C2 (CH2)
        [2.08, 1.42, 0.0],        # O (OH)
        [-0.37, -0.51, 0.89],     # H (CH3)
        [-0.37, -0.51, -0.89],    # H (CH3)
        [-0.37, 1.02, 0.0],       # H (CH3)
        [1.91, -0.51, 0.89],      # H (CH2)
        [1.91, -0.51, -0.89],     # H (CH2)
        [3.05, 1.42, 0.0],        # H (OH)
    ])
    bonds = [
        Bond(0, 1, 1.0),  # C-C
        Bond(1, 2, 1.0),  # C-O
        Bond(0, 3, 1.0), Bond(0, 4, 1.0), Bond(0, 5, 1.0),  # CH3
        Bond(1, 6, 1.0), Bond(1, 7, 1.0),  # CH2
        Bond(2, 8, 1.0),  # OH
    ]
    return Molecule(atoms=atoms, bonds=bonds, positions=positions)


def demo_ball_and_stick():
    """Demo: Ball-and-stick molecular visualization."""
    from morphogen.stdlib import visual3d

    print("Rendering ball-and-stick molecules...")

    # Create molecules
    water = create_water_molecule()
    methane = create_methane_molecule()
    ethanol = create_ethanol_molecule()

    # Render each molecule
    water_vis = visual3d.molecule(water, style='ball_and_stick')
    methane_vis = visual3d.molecule(methane, style='ball_and_stick')
    ethanol_vis = visual3d.molecule(ethanol, style='ball_and_stick')

    # Camera looking at water
    camera = visual3d.camera(position=(3, 3, 2), focal_point=(0, 0, 0))

    # Render water
    image = visual3d.render_3d(
        water_vis,
        camera=camera,
        background=(0.1, 0.1, 0.15),
        width=800,
        height=600
    )

    # Save using visual output
    from morphogen.stdlib import visual
    visual.output(image, str(OUTPUT_DIR / "water_ball_stick.png"))
    print(f"  Saved: {OUTPUT_DIR / 'water_ball_stick.png'}")

    # Render ethanol with adjusted camera
    camera_ethanol = visual3d.camera(position=(5, 4, 3), focal_point=(1.0, 0.5, 0))
    image_ethanol = visual3d.render_3d(
        ethanol_vis,
        camera=camera_ethanol,
        background=(0.1, 0.1, 0.15),
        width=800,
        height=600
    )
    visual.output(image_ethanol, str(OUTPUT_DIR / "ethanol_ball_stick.png"))
    print(f"  Saved: {OUTPUT_DIR / 'ethanol_ball_stick.png'}")


def demo_spacefill():
    """Demo: Space-filling (CPK) molecular visualization."""
    from morphogen.stdlib import visual3d, visual

    print("Rendering space-filling molecules...")

    ethanol = create_ethanol_molecule()
    vis = visual3d.molecule(ethanol, style='spacefill')

    camera = visual3d.camera(position=(6, 5, 4), focal_point=(1.0, 0.5, 0))
    image = visual3d.render_3d(
        vis,
        camera=camera,
        background=(0.05, 0.05, 0.1),
        width=800,
        height=600
    )

    visual.output(image, str(OUTPUT_DIR / "ethanol_spacefill.png"))
    print(f"  Saved: {OUTPUT_DIR / 'ethanol_spacefill.png'}")


def demo_orbital_visualization():
    """Demo: Molecular orbital isosurface visualization."""
    from morphogen.stdlib import visual3d, visual

    print("Rendering orbital isosurfaces...")

    # Create 2p-like orbital field (simplified)
    # Real orbitals would come from quantum chemistry calculations
    x, y, z = np.mgrid[-4:4:40j, -4:4:40j, -4:4:40j]
    r = np.sqrt(x**2 + y**2 + z**2)

    # 2p_x orbital: x * exp(-r/2)
    orbital_2px = x * np.exp(-r / 2)
    orbital_2px /= np.max(np.abs(orbital_2px))  # Normalize

    # Visualize orbital
    vis = visual3d.orbital(
        orbital_2px,
        isovalues=(0.15, -0.15),
        positive_color=(0.0, 0.3, 0.8),  # Blue positive lobe
        negative_color=(0.8, 0.3, 0.0),  # Orange negative lobe
        opacity=0.8,
        spacing=(0.2, 0.2, 0.2),
        origin=(-4.0, -4.0, -4.0)
    )

    camera = visual3d.camera(position=(10, 10, 8), focal_point=(0, 0, 0))
    image = visual3d.render_3d(
        vis,
        camera=camera,
        background=(0.02, 0.02, 0.05),
        width=800,
        height=600
    )

    visual.output(image, str(OUTPUT_DIR / "orbital_2px.png"))
    print(f"  Saved: {OUTPUT_DIR / 'orbital_2px.png'}")


def demo_rotating_molecule():
    """Demo: Animated rotation around a molecule."""
    from morphogen.stdlib import visual3d, visual

    print("Rendering rotating molecule animation...")

    ethanol = create_ethanol_molecule()
    vis = visual3d.molecule(ethanol, style='ball_and_stick', atom_scale=0.4)

    # Create orbital camera path
    camera_path = visual3d.orbit_camera(
        center=(1.0, 0.5, 0.0),
        radius=6.0,
        elevation=25.0,
        frames=60,
        orbits=1.0
    )

    # Render video
    output_path = str(OUTPUT_DIR / "ethanol_rotation.mp4")
    visual3d.video_3d(
        vis,
        camera_path=camera_path,
        path=output_path,
        fps=30,
        width=640,
        height=480,
        background=(0.1, 0.1, 0.15)
    )
    print(f"  Saved: {output_path}")


def demo_fly_through():
    """Demo: Fly-through camera path."""
    from morphogen.stdlib import visual3d, visual

    print("Rendering fly-through animation...")

    # Create multiple molecules at different positions
    water = create_water_molecule()

    # Position molecules in a line
    molecules = []
    for i in range(3):
        vis = visual3d.molecule(water, style='ball_and_stick')
        # Offset each molecule (would need transformation support)
        molecules.append(vis)

    # For now, just show single molecule with fly-through path
    vis = visual3d.molecule(water, style='ball_and_stick')

    # Create fly-through camera path
    waypoints = [
        (5, 0, 2),
        (3, 3, 1),
        (0, 5, 2),
        (-3, 3, 1),
        (-5, 0, 2),
    ]

    camera_path = visual3d.fly_path(
        waypoints,
        frames=90,
        look_at=(0, 0, 0),
        interpolation="spline"
    )

    output_path = str(OUTPUT_DIR / "water_flythrough.mp4")
    visual3d.video_3d(
        vis,
        camera_path=camera_path,
        path=output_path,
        fps=30,
        width=640,
        height=480,
        background=(0.05, 0.05, 0.1)
    )
    print(f"  Saved: {output_path}")


def main():
    """Run all molecular visualization demos."""
    print("=" * 60)
    print("Morphogen 3D Molecular Visualization Demo")
    print("=" * 60)
    print()

    demo_ball_and_stick()
    print()

    demo_spacefill()
    print()

    demo_orbital_visualization()
    print()

    demo_rotating_molecule()
    print()

    demo_fly_through()
    print()

    print("=" * 60)
    print("All demos complete!")
    print(f"Output files saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
