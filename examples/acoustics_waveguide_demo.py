"""Acoustics Domain Demo: 1D Waveguide Simulation

This example demonstrates the Phase 1 acoustics implementation:
1. Creating pipe geometries (simple pipes and expansion chambers)
2. Building digital waveguides for acoustic simulation
3. Computing reflection coefficients
4. Simulating wave propagation
5. Computing frequency response and resonances

Use cases demonstrated:
- Simple open pipe resonance
- Expansion chamber (muffler) acoustics
- Helmholtz resonator calculations
"""

import numpy as np
import matplotlib.pyplot as plt
from morphogen.stdlib.acoustics import (
    acoustics, create_pipe, create_expansion_chamber,
    SPEED_OF_SOUND
)


def demo_simple_pipe_resonance():
    """Demo 1: Simple open pipe showing resonances."""
    print("=" * 70)
    print("DEMO 1: Simple Open Pipe Resonance")
    print("=" * 70)

    # Create a 1m pipe (open at both ends)
    pipe = create_pipe(diameter=0.025, length=1.0)
    print(f"\nPipe: {pipe}")

    # Build waveguide
    wg = acoustics.waveguide_from_geometry(
        pipe,
        discretization=0.02,  # 2cm segments
        sample_rate=44100
    )
    print(f"Waveguide: {wg}")
    print(f"Total delay: {wg.delay_samples} samples")

    # Compute reflection coefficients (open end)
    reflections = acoustics.reflection_coefficients(wg, end_condition="open")
    print(f"\nReflection coefficients: {len(reflections)}")
    for r in reflections:
        print(f"  {r}")

    # Compute transfer function
    print("\nComputing transfer function...")
    response = acoustics.transfer_function(
        wg, reflections,
        freq_range=(50.0, 1000.0),
        resolution=10.0
    )
    print(f"Transfer function: {response}")

    # Find resonant frequencies
    resonances = acoustics.resonant_frequencies(response, threshold_db=-10.0)
    print(f"\nResonant frequencies (peaks > -10dB):")
    for i, f in enumerate(resonances):
        print(f"  f{i+1} = {f:.1f} Hz")

    # Theoretical resonances for open-open pipe: f_n = n * c / (2L)
    print(f"\nTheoretical resonances (f = n * c / 2L):")
    for n in range(1, 6):
        f_theory = n * SPEED_OF_SOUND / (2 * pipe.length)
        print(f"  f{n} = {f_theory:.1f} Hz")

    # Plot frequency response
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(response.frequencies, response.magnitude)
    plt.axhline(-10, color='r', linestyle='--', label='Threshold (-10 dB)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Transfer Function: Magnitude')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(response.frequencies, response.phase)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (radians)')
    plt.title('Transfer Function: Phase')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('examples/output_acoustics_pipe_resonance.png', dpi=150)
    print("\nSaved plot: examples/output_acoustics_pipe_resonance.png")


def demo_expansion_chamber():
    """Demo 2: Expansion chamber (muffler) acoustics."""
    print("\n" + "=" * 70)
    print("DEMO 2: Expansion Chamber (Muffler)")
    print("=" * 70)

    # Create expansion chamber (muffler geometry)
    chamber = create_expansion_chamber(
        inlet_diameter=0.04,   # 40mm inlet
        belly_diameter=0.12,   # 120mm belly
        outlet_diameter=0.05,  # 50mm outlet
        total_length=1.0       # 1m total
    )
    print(f"\nExpansion chamber: {chamber}")

    # Build waveguide
    wg = acoustics.waveguide_from_geometry(chamber, discretization=0.02)
    print(f"Waveguide: {wg}")

    # Compute reflection coefficients
    reflections = acoustics.reflection_coefficients(wg, end_condition="open")
    print(f"\nReflection coefficients: {len(reflections)}")
    internal_reflections = [r for r in reflections if r.position < wg.num_segments - 1]
    print(f"Internal reflections (at area changes): {len(internal_reflections)}")

    # Simulate impulse response
    print("\nSimulating impulse response...")
    p_fwd = np.zeros(wg.num_segments)
    p_bwd = np.zeros(wg.num_segments)

    num_steps = 500
    output = np.zeros(num_steps)

    for t in range(num_steps):
        # Impulse at t=0
        if t == 0:
            excitation = np.array([1.0])
        else:
            excitation = None

        p_fwd, p_bwd = acoustics.waveguide_step(
            p_fwd, p_bwd, wg, reflections,
            excitation=excitation, excitation_pos=0
        )

        # Record output at end
        p_total = acoustics.total_pressure(p_fwd, p_bwd)
        output[t] = p_total[-1]

    # Plot impulse response
    plt.figure(figsize=(12, 8))

    # Impulse response
    plt.subplot(2, 2, 1)
    time = np.arange(num_steps) / wg.sample_rate * 1000  # ms
    plt.plot(time, output)
    plt.xlabel('Time (ms)')
    plt.ylabel('Pressure (Pa)')
    plt.title('Impulse Response')
    plt.grid(True)

    # Waveguide geometry
    plt.subplot(2, 2, 2)
    positions = np.arange(wg.num_segments) * wg.segment_length
    plt.plot(positions, wg.diameters * 1000)  # Convert to mm
    plt.xlabel('Position (m)')
    plt.ylabel('Diameter (mm)')
    plt.title('Pipe Geometry')
    plt.grid(True)

    # Frequency response
    print("\nComputing frequency response...")
    response = acoustics.transfer_function(
        wg, reflections,
        freq_range=(50.0, 1000.0),
        resolution=20.0
    )

    plt.subplot(2, 2, 3)
    plt.plot(response.frequencies, response.magnitude)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Frequency Response')
    plt.grid(True)

    # Find resonances
    resonances = acoustics.resonant_frequencies(response, threshold_db=-5.0)
    print(f"\nResonant frequencies:")
    for f in resonances:
        print(f"  {f:.1f} Hz")

    # Energy over time
    plt.subplot(2, 2, 4)
    energy = output ** 2
    plt.plot(time, energy)
    plt.xlabel('Time (ms)')
    plt.ylabel('Energy (Pa²)')
    plt.title('Energy Decay')
    plt.grid(True)
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig('examples/output_acoustics_expansion_chamber.png', dpi=150)
    print("\nSaved plot: examples/output_acoustics_expansion_chamber.png")


def demo_helmholtz_resonator():
    """Demo 3: Helmholtz resonator calculations."""
    print("\n" + "=" * 70)
    print("DEMO 3: Helmholtz Resonator")
    print("=" * 70)

    # Example: Quarter-wave resonator for muffler
    volume = 500e-6        # 500 cm³
    neck_length = 0.05     # 50 mm
    neck_area = 20e-4      # 20 cm²

    print(f"\nResonator parameters:")
    print(f"  Volume: {volume*1e6:.0f} cm³")
    print(f"  Neck length: {neck_length*1000:.0f} mm")
    print(f"  Neck area: {neck_area*1e4:.0f} cm²")

    # Compute resonant frequency
    f_res = acoustics.helmholtz_frequency(volume, neck_length, neck_area)
    print(f"\nResonant frequency: {f_res:.1f} Hz")

    # Compute impedance over frequency range
    frequencies = np.linspace(20, 2000, 200)
    impedances = []

    for f in frequencies:
        Z = acoustics.helmholtz_impedance(
            frequency=f,
            volume=volume,
            neck_length=neck_length,
            neck_area=neck_area,
            damping=0.1
        )
        impedances.append(Z)

    impedances = np.array(impedances)

    # Plot impedance
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(frequencies, np.abs(impedances))
    plt.axvline(f_res, color='r', linestyle='--', label=f'f_res = {f_res:.1f} Hz')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('|Z| (Pa·s/m³)')
    plt.title('Impedance Magnitude')
    plt.grid(True)
    plt.legend()
    plt.yscale('log')

    plt.subplot(1, 3, 2)
    plt.plot(frequencies, impedances.real, label='Resistance')
    plt.plot(frequencies, impedances.imag, label='Reactance')
    plt.axvline(f_res, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Z (Pa·s/m³)')
    plt.title('Impedance Components')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(frequencies, np.angle(impedances))
    plt.axvline(f_res, color='r', linestyle='--')
    plt.axhline(0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (radians)')
    plt.title('Impedance Phase')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('examples/output_acoustics_helmholtz.png', dpi=150)
    print("\nSaved plot: examples/output_acoustics_helmholtz.png")


def demo_radiation_impedance():
    """Demo 4: Radiation impedance calculations."""
    print("\n" + "=" * 70)
    print("DEMO 4: Radiation Impedance")
    print("=" * 70)

    # Test different pipe diameters
    diameters = [0.025, 0.05, 0.10]  # 25mm, 50mm, 100mm
    frequencies = np.logspace(1, 4, 100)  # 10 Hz to 10 kHz

    plt.figure(figsize=(12, 4))

    for d in diameters:
        Z_rad = []
        for f in frequencies:
            Z = acoustics.radiation_impedance_unflanged(d, f)
            Z_rad.append(Z)
        Z_rad = np.array(Z_rad)

        plt.subplot(1, 2, 1)
        plt.semilogx(frequencies, Z_rad.real, label=f'd = {d*1000:.0f} mm')

        plt.subplot(1, 2, 2)
        plt.semilogx(frequencies, Z_rad.imag, label=f'd = {d*1000:.0f} mm')

    plt.subplot(1, 2, 1)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Radiation Resistance (Pa·s/m³)')
    plt.title('Radiation Impedance: Real Part')
    plt.grid(True, which='both')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Radiation Reactance (Pa·s/m³)')
    plt.title('Radiation Impedance: Imaginary Part')
    plt.grid(True, which='both')
    plt.legend()

    plt.tight_layout()
    plt.savefig('examples/output_acoustics_radiation.png', dpi=150)
    print("\nSaved plot: examples/output_acoustics_radiation.png")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MORPHOGEN ACOUSTICS DOMAIN - Phase 1 Demonstration")
    print("1D Waveguide Acoustics")
    print("=" * 70)

    # Run all demos
    demo_simple_pipe_resonance()
    demo_expansion_chamber()
    demo_helmholtz_resonator()
    demo_radiation_impedance()

    print("\n" + "=" * 70)
    print("All demos completed successfully!")
    print("=" * 70)
    print("\nGenerated plots:")
    print("  - output_acoustics_pipe_resonance.png")
    print("  - output_acoustics_expansion_chamber.png")
    print("  - output_acoustics_helmholtz.png")
    print("  - output_acoustics_radiation.png")
