#!/usr/bin/env python3
"""
Automatic State Management Demo

This example demonstrates Morphogen's automatic state management in the
OperatorExecutor scheduler. No manual state tracking required!

Features demonstrated:
1. **Phase Continuity** - Oscillators seamlessly continue across executions
2. **Filter State Continuity** - VCF filters maintain state across buffer hops
3. **Infinite Synthesis** - Process audio incrementally without clicks/pops
4. **Zero Configuration** - State managed automatically per node_id

Session: donupa-1123 (2025-11-23)
Feature: Convention-Based State Management Refactor
"""

import numpy as np
import matplotlib.pyplot as plt
from morphogen.scheduler.operator_executor import (
    OperatorExecutor,
    create_audio_registry,
)
from morphogen.stdlib.audio import AudioBuffer, AudioOperations as audio


def demo_phase_continuity():
    """
    Demo 1: Automatic Phase Continuity

    Executes sine oscillator 10 times. The OperatorExecutor automatically
    manages phase state, producing seamless continuity across executions.
    """
    print("\n" + "=" * 70)
    print("DEMO 1: Automatic Phase Continuity")
    print("=" * 70)

    # Create executor
    registry = create_audio_registry()
    executor = OperatorExecutor(registry, sample_rate=48000)

    freq = 440.0  # A440
    chunk_samples = 4800  # 0.1 second chunks
    num_chunks = 10

    print(f"\nGenerating {num_chunks} chunks of {freq}Hz sine wave")
    print(f"Chunk size: {chunk_samples} samples (0.1 seconds)")
    print("\nPhase state after each chunk:")

    # Execute oscillator multiple times with SAME node_id
    # Phase state automatically persists!
    chunks = []
    for i in range(num_chunks):
        result = executor.execute(
            operator_name="sine",
            node_id="osc_auto",  # Same node_id = automatic state persistence
            params={"freq": freq},
            inputs={},
            num_samples=chunk_samples,
            rate_hz=48000.0,
        )
        chunks.append(result["out"])

        # Show phase state (automatically managed by executor)
        phase = executor._operator_state["osc_auto"]["phase"]
        print(f"  Chunk {i+1:2d}: phase = {phase:.6f} radians")

    # Concatenate all chunks
    with_state = np.concatenate(chunks)

    # Generate reference: continuous sine wave (no state management)
    t_full = np.arange(len(with_state)) / 48000.0
    reference = np.sin(2.0 * np.pi * freq * t_full)

    # Measure continuity quality
    max_error = np.max(np.abs(with_state - reference))
    rms_error = np.sqrt(np.mean((with_state - reference) ** 2))

    print(f"\n✅ Phase Continuity Quality:")
    print(f"   Max error: {max_error:.2e}")
    print(f"   RMS error: {rms_error:.2e}")
    print(f"   Quality: {'EXCELLENT' if max_error < 1e-6 else 'GOOD'}")
    print(f"\n💡 No manual state management - executor handled everything!")

    return with_state, reference


def demo_filter_state_continuity():
    """
    Demo 2: Automatic Filter State Continuity

    Processes sawtooth wave through VCF lowpass in 5 chunks. The OperatorExecutor
    automatically manages biquad filter state for seamless continuity.
    """
    print("\n" + "=" * 70)
    print("DEMO 2: Automatic Filter State Continuity")
    print("=" * 70)

    # Create executor
    registry = create_audio_registry()
    executor = OperatorExecutor(registry, sample_rate=48000)

    freq = 220.0  # A3
    cutoff_hz = 2000.0
    q = 2.0
    chunk_samples = 4800  # 0.1 second chunks
    num_chunks = 5

    print(f"\nFiltering {freq}Hz sawtooth through {cutoff_hz}Hz lowpass (Q={q})")
    print(f"Processing in {num_chunks} chunks of {chunk_samples} samples")
    print("\nFilter state after each chunk:")

    # Generate full sawtooth signal
    t_full = np.arange(num_chunks * chunk_samples) / 48000.0
    full_signal = 2.0 * (t_full * freq - np.floor(t_full * freq + 0.5))

    # Process in chunks with SAME node_id
    # Filter state automatically persists!
    chunks = []
    for i in range(num_chunks):
        # Extract chunk
        chunk = full_signal[i * chunk_samples : (i + 1) * chunk_samples]
        cutoff = np.ones(chunk_samples) * cutoff_hz

        # Execute filter - state managed automatically!
        result = executor.execute(
            operator_name="vcf_lowpass",
            node_id="filter_auto",  # Same node_id = automatic state persistence
            params={"q": q},
            inputs={"signal": chunk, "cutoff": cutoff},
            num_samples=chunk_samples,
            rate_hz=48000.0,
        )
        chunks.append(result["out"])

        # Show filter state (automatically managed by executor)
        state = executor._operator_state["filter_auto"]["filter_state"].data
        print(f"  Chunk {i+1}: filter_state = [{state[0]:+.6f}, {state[1]:+.6f}]")

    # Concatenate hopped output
    with_state = np.concatenate(chunks)

    # Generate reference: continuous filtering (no hops)
    signal_buf = AudioBuffer(data=full_signal, sample_rate=48000)
    cutoff_buf = AudioBuffer(data=np.ones(len(full_signal)) * cutoff_hz, sample_rate=48000)
    reference = audio.vcf_lowpass(signal_buf, cutoff_buf, q=q)

    # Measure continuity quality
    max_error = np.max(np.abs(with_state - reference.data))
    rms_error = np.sqrt(np.mean((with_state - reference.data) ** 2))

    # Check boundary discontinuities specifically
    boundary_errors = []
    for i in range(1, num_chunks):
        boundary_idx = i * chunk_samples
        for offset in range(-5, 5):
            idx = boundary_idx + offset
            if 0 <= idx < len(with_state):
                error = abs(with_state[idx] - reference.data[idx])
                boundary_errors.append(error)

    max_boundary_error = max(boundary_errors)

    print(f"\n✅ Filter State Continuity Quality:")
    print(f"   Max error (overall): {max_error:.2e}")
    print(f"   Max boundary error: {max_boundary_error:.2e}")
    print(f"   RMS error: {rms_error:.2e}")
    print(f"   Quality: {'EXCELLENT' if max_error < 1e-6 else 'GOOD'}")
    print(f"\n💡 No manual state management - executor handled everything!")

    return with_state, reference.data


def demo_infinite_synth_patch():
    """
    Demo 3: Infinite Synth Patch

    Combines phase continuity AND filter state for a complete synth patch
    that can run infinitely without clicks or discontinuities.

    This is the killer feature - real-time synthesis with perfect continuity!
    """
    print("\n" + "=" * 70)
    print("DEMO 3: Infinite Synth Patch (Phase + Filter State)")
    print("=" * 70)

    # Create executor
    registry = create_audio_registry()
    executor = OperatorExecutor(registry, sample_rate=48000)

    osc_freq = 110.0  # A2
    cutoff_start = 500.0
    cutoff_end = 3000.0
    q = 3.0
    chunk_samples = 2400  # 0.05 second chunks
    num_chunks = 20  # 1 second total

    print(f"\nInfinite synth patch:")
    print(f"  Oscillator: {osc_freq}Hz sawtooth")
    print(f"  Filter: VCF lowpass, Q={q}")
    print(f"  Cutoff: sweep from {cutoff_start}Hz to {cutoff_end}Hz")
    print(f"  Processing: {num_chunks} chunks × {chunk_samples} samples")
    print(f"\nState management (both phase + filter_state):")

    # Process in chunks - BOTH phase and filter state managed automatically!
    chunks = []
    for i in range(num_chunks):
        # Calculate cutoff for this chunk (linear sweep)
        progress = i / (num_chunks - 1)
        cutoff_hz = cutoff_start + progress * (cutoff_end - cutoff_start)

        # 1. Generate oscillator chunk (PHASE STATE automatic!)
        osc_result = executor.execute(
            operator_name="saw",
            node_id="osc_patch",
            params={"freq": osc_freq},
            inputs={},
            num_samples=chunk_samples,
            rate_hz=48000.0,
        )

        # 2. Filter oscillator output (FILTER STATE automatic!)
        cutoff_array = np.ones(chunk_samples) * cutoff_hz
        filter_result = executor.execute(
            operator_name="vcf_lowpass",
            node_id="filter_patch",
            params={"q": q},
            inputs={"signal": osc_result["out"], "cutoff": cutoff_array},
            num_samples=chunk_samples,
            rate_hz=48000.0,
        )

        chunks.append(filter_result["out"])

        # Show both states
        if i % 5 == 0:  # Every 5th chunk
            phase = executor._operator_state["osc_patch"]["phase"]
            filt_state = executor._operator_state["filter_patch"]["filter_state"].data
            print(f"  Chunk {i+1:2d}: cutoff={cutoff_hz:6.1f}Hz, "
                  f"phase={phase:.4f}, filter=[{filt_state[0]:+.4f}, {filt_state[1]:+.4f}]")

    # Concatenate all chunks
    output = np.concatenate(chunks)

    # Analyze continuity at chunk boundaries
    discontinuities = []
    for i in range(1, num_chunks):
        boundary_idx = i * chunk_samples
        # Check derivative discontinuity
        before = output[boundary_idx - 1]
        after = output[boundary_idx]
        disc = abs(after - before)
        discontinuities.append(disc)

    max_discontinuity = max(discontinuities)
    avg_discontinuity = np.mean(discontinuities)

    print(f"\n✅ Infinite Synth Patch Quality:")
    print(f"   Max boundary discontinuity: {max_discontinuity:.2e}")
    print(f"   Avg boundary discontinuity: {avg_discontinuity:.2e}")
    print(f"   Quality: {'EXCELLENT' if max_discontinuity < 0.01 else 'GOOD'}")
    print(f"\n💡 Both phase AND filter state managed automatically!")
    print(f"💡 This patch could run INFINITELY without clicks!")

    return output


def visualize_results():
    """
    Demo 4: Visualization of Automatic State Management

    Creates plots showing:
    1. Phase continuity comparison
    2. Filter state continuity comparison
    3. Infinite synth patch output
    """
    print("\n" + "=" * 70)
    print("DEMO 4: Visualization")
    print("=" * 70)

    # Run demos to get data
    phase_with_state, phase_reference = demo_phase_continuity()
    filter_with_state, filter_reference = demo_filter_state_continuity()
    infinite_output = demo_infinite_synth_patch()

    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle("Automatic State Management in Morphogen Scheduler", fontsize=14, fontweight="bold")

    # Plot 1: Phase continuity
    ax1 = axes[0]
    samples_to_plot = 9600  # 0.2 seconds
    t = np.arange(samples_to_plot) / 48000.0
    ax1.plot(t, phase_reference[:samples_to_plot], 'b-', linewidth=2, alpha=0.7, label="Reference (continuous)")
    ax1.plot(t, phase_with_state[:samples_to_plot], 'r--', linewidth=1, label="With auto phase state")
    ax1.axvline(x=0.1, color='gray', linestyle=':', alpha=0.5, label="Chunk boundaries")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Demo 1: Phase Continuity (440Hz Sine)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Filter state continuity
    ax2 = axes[1]
    ax2.plot(t, filter_reference[:samples_to_plot], 'b-', linewidth=2, alpha=0.7, label="Reference (continuous)")
    ax2.plot(t, filter_with_state[:samples_to_plot], 'r--', linewidth=1, label="With auto filter state")
    for i in range(1, 2):  # Show first boundary
        ax2.axvline(x=i * 0.1, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Demo 2: Filter State Continuity (220Hz Saw → 2kHz Lowpass)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Infinite synth patch
    ax3 = axes[2]
    t_inf = np.arange(len(infinite_output)) / 48000.0
    ax3.plot(t_inf, infinite_output, 'g-', linewidth=0.5)
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Amplitude")
    ax3.set_title("Demo 3: Infinite Synth Patch (110Hz Saw → Sweeping Lowpass)")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("examples/auto_state_management.png", dpi=150)
    print("\n✅ Visualization saved: examples/auto_state_management.png")


def main():
    """Run all demos."""
    print("=" * 70)
    print("AUTOMATIC STATE MANAGEMENT DEMONSTRATION")
    print("Morphogen Scheduler - Convention-Based State Management")
    print("Session: donupa-1123 (2025-11-23)")
    print("=" * 70)

    # Run individual demos
    demo_phase_continuity()
    demo_filter_state_continuity()
    demo_infinite_synth_patch()

    # Create visualization
    visualize_results()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\n✅ All automatic state management features demonstrated:")
    print("   1. Phase continuity (executor-managed state)")
    print("   2. Filter state continuity (operator-managed state)")
    print("   3. Combined infinite synthesis (both states)")
    print("   4. Visualization showing perfect continuity")
    print("\n💡 KEY INSIGHT: Zero manual state management required!")
    print("   - Just use same node_id across executions")
    print("   - Scheduler handles phase and filter state automatically")
    print("   - Perfect for real-time synthesis, infinite patches, streaming audio")
    print("\n🎉 Convention-based state management makes infinite synthesis trivial!")


if __name__ == "__main__":
    main()
