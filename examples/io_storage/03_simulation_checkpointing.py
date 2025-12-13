"""Example: Simulation checkpointing and resume.

Demonstrates checkpoint/resume workflow for long-running simulations.
"""

import numpy as np
import sys
from pathlib import Path
import time

# Add morphogen to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from morphogen.stdlib import io_storage as io


def heat_diffusion_step(field, dt, diffusivity=0.1):
    """Perform one heat diffusion step (simple explicit method)."""
    # 5-point stencil Laplacian
    laplacian = (
        np.roll(field, 1, axis=0) +
        np.roll(field, -1, axis=0) +
        np.roll(field, 1, axis=1) +
        np.roll(field, -1, axis=1) -
        4 * field
    )
    return field + dt * diffusivity * laplacian


def example_1_basic_checkpoint():
    """Basic checkpointing example."""
    print("=" * 60)
    print("Example 1: Basic checkpoint save/load")
    print("=" * 60)

    # Create simulation state
    size = 64
    field = np.random.rand(size, size).astype(np.float32)

    # Add hot spot in center
    center = size // 2
    field[center-5:center+5, center-5:center+5] = 1.0

    # Simulation parameters
    params = {
        "dt": 0.01,
        "diffusivity": 0.1,
        "grid_size": size
    }

    # Save initial checkpoint
    state = {"field": field, "parameters": params}
    metadata = {"iteration": 0, "time": 0.0}

    io.save_checkpoint("checkpoint_0.h5", state, metadata)
    print("  ✓ Saved initial checkpoint: checkpoint_0.h5")

    # Run simulation for a few steps
    for i in range(1, 6):
        field = heat_diffusion_step(field, params["dt"], params["diffusivity"])

        # Save checkpoint every step
        state["field"] = field
        metadata = {"iteration": i, "time": i * params["dt"]}
        io.save_checkpoint(f"checkpoint_{i}.h5", state, metadata)
        print(f"  ✓ Saved checkpoint_{i}.h5 (t={metadata['time']:.3f})")

    print()


def example_2_resume_simulation():
    """Resume simulation from checkpoint."""
    print("=" * 60)
    print("Example 2: Resume from checkpoint")
    print("=" * 60)

    # Load the last checkpoint
    checkpoint_file = "checkpoint_5.h5"
    state, metadata = io.load_checkpoint(checkpoint_file)

    print(f"  Loaded checkpoint: {checkpoint_file}")
    print(f"    Iteration: {metadata['iteration']}")
    print(f"    Time: {metadata['time']:.3f}")
    print(f"    Field shape: {state['field'].shape}")

    # Continue simulation
    field = state["field"]
    params = state["parameters"]
    iteration = metadata["iteration"]
    sim_time = metadata["time"]

    print("\n  Continuing simulation...")
    for i in range(5):
        iteration += 1
        sim_time += params["dt"]
        field = heat_diffusion_step(field, params["dt"], params["diffusivity"])

        # Save new checkpoints
        state["field"] = field
        new_metadata = {"iteration": iteration, "time": sim_time}
        io.save_checkpoint(f"checkpoint_{iteration}.h5", state, new_metadata)
        print(f"  ✓ Saved checkpoint_{iteration}.h5 (t={sim_time:.3f})")

    print()


def example_3_periodic_checkpointing():
    """Checkpoint periodically during long simulation."""
    print("=" * 60)
    print("Example 3: Periodic checkpointing")
    print("=" * 60)

    # Initialize
    size = 128
    field = np.random.rand(size, size).astype(np.float32) * 0.1
    field[size//2-10:size//2+10, size//2-10:size//2+10] = 1.0

    params = {"dt": 0.01, "diffusivity": 0.2, "grid_size": size}

    # Simulation parameters
    total_steps = 100
    checkpoint_interval = 20  # Save every 20 steps

    print(f"  Running {total_steps} steps, checkpointing every {checkpoint_interval} steps")

    start_time = time.time()

    for step in range(total_steps):
        field = heat_diffusion_step(field, params["dt"], params["diffusivity"])

        # Checkpoint periodically
        if (step + 1) % checkpoint_interval == 0 or step == 0:
            state = {"field": field, "parameters": params}
            metadata = {
                "iteration": step + 1,
                "time": (step + 1) * params["dt"],
                "checkpoint_interval": checkpoint_interval
            }
            checkpoint_name = f"periodic_checkpoint_{step+1:04d}.h5"
            io.save_checkpoint(checkpoint_name, state, metadata)
            print(f"  ✓ Checkpoint at step {step+1}: {checkpoint_name}")

    elapsed = time.time() - start_time
    print(f"\n  Simulation completed in {elapsed:.2f}s")
    print(f"  Final field mean: {field.mean():.4f}")
    print()


def example_4_checkpoint_with_hdf5_inspection():
    """Save checkpoint and demonstrate data inspection."""
    print("=" * 60)
    print("Example 4: Checkpoint with data inspection")
    print("=" * 60)

    # Create complex state with multiple fields
    size = 64
    state = {
        "velocity": np.random.rand(size, size, 2).astype(np.float32),
        "pressure": np.random.rand(size, size).astype(np.float32),
        "temperature": np.random.rand(size, size).astype(np.float32) * 100 + 300,
        "parameters": {
            "dt": 0.001,
            "viscosity": 0.01,
            "diffusivity": 0.02
        }
    }

    metadata = {
        "iteration": 1000,
        "time": 1.0,
        "version": "1.0",
        "description": "Fluid simulation with heat transfer"
    }

    # Save checkpoint
    checkpoint_file = "complex_checkpoint.h5"
    io.save_checkpoint(checkpoint_file, state, metadata)
    print(f"  ✓ Saved {checkpoint_file}")

    # Load and inspect
    loaded_state, loaded_metadata = io.load_checkpoint(checkpoint_file)

    print("\n  Checkpoint contents:")
    print("  State fields:")
    for key, value in loaded_state.items():
        if isinstance(value, np.ndarray):
            print(f"    - {key}: shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, dict):
            print(f"    - {key}: {value}")
        else:
            print(f"    - {key}: {value}")

    print("  Metadata:")
    for key, value in loaded_metadata.items():
        print(f"    - {key}: {value}")

    # Save visualization of temperature field
    temp = loaded_state["temperature"]
    temp_norm = (temp - temp.min()) / (temp.max() - temp.min())
    temp_rgb = np.stack([temp_norm, np.zeros_like(temp_norm), 1 - temp_norm], axis=-1)

    io.save_image("checkpoint_temperature_vis.png", temp_rgb)
    print("\n  ✓ Saved visualization: checkpoint_temperature_vis.png")

    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("SIMULATION CHECKPOINTING EXAMPLES")
    print("=" * 60)
    print()

    example_1_basic_checkpoint()
    example_2_resume_simulation()
    example_3_periodic_checkpointing()
    example_4_checkpoint_with_hdf5_inspection()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print("\nGenerated checkpoint files:")
    print("  - checkpoint_*.h5 (basic checkpoints)")
    print("  - periodic_checkpoint_*.h5 (periodic checkpoints)")
    print("  - complex_checkpoint.h5 (multi-field checkpoint)")
    print("  - checkpoint_temperature_vis.png (visualization)")
    print()


if __name__ == "__main__":
    main()
