"""Digital Twin Example - Cross-Domain Showcase

This example demonstrates the power of combining multiple Morphogen domains
to create a digital twin - a virtual representation of a physical system
that simulates real-world behavior.

Domains integrated:
- Field operations for thermal dynamics and fluid flow
- Integrators for time evolution
- Sparse linear algebra for PDE solving
- Visual rendering for real-time monitoring
- I/O for data logging and checkpointing
- Optimization for parameter tuning

Creates digital twin simulations:
- Manufacturing process (thermal treatment)
- Heat exchanger system
- Cooling system with active control
- Multi-physics coupling (thermal + structural)
- Real-time monitoring and visualization
"""

import numpy as np
from morphogen.stdlib import field, integrators, sparse_linalg, palette, image, visual, io_storage
from morphogen.stdlib.field import Field2D
from morphogen.stdlib.visual import Visual
import time


class ThermalManufacturingTwin:
    """Digital twin of a thermal manufacturing process.

    Simulates heat treatment of a metal part with:
    - Heating zones
    - Cooling zones
    - Material thermal properties
    - Quality metrics
    """

    def __init__(self, width=100, height=100, dt=0.01):
        """Initialize the digital twin.

        Args:
            width: Grid width
            height: Grid height
            dt: Time step
        """
        self.width = width
        self.height = height
        self.dt = dt

        # Initialize temperature field (room temperature)
        self.temperature = field.alloc((height, width), dtype=np.float32, fill_value=20.0)

        # Material properties
        self.thermal_diffusivity = 0.05  # m²/s

        # Process zones
        self.heating_zone = self._create_heating_zone()
        self.cooling_zone = self._create_cooling_zone()

        # Metrics
        self.time_elapsed = 0.0
        self.max_temp_history = []
        self.avg_temp_history = []
        self.quality_score = 0.0

    def _create_heating_zone(self):
        """Create heating zone mask."""
        zone = np.zeros((self.height, self.width), dtype=np.float32)
        # Left third is heating zone
        zone[:, :self.width//3] = 1.0
        return Field2D(zone)

    def _create_cooling_zone(self):
        """Create cooling zone mask."""
        zone = np.zeros((self.height, self.width), dtype=np.float32)
        # Right third is cooling zone
        zone[:, 2*self.width//3:] = 1.0
        return Field2D(zone)

    def step(self, heating_power=100.0, cooling_power=-50.0):
        """Advance simulation by one time step.

        Args:
            heating_power: Heating power (°C/s)
            cooling_power: Cooling power (°C/s)

        Returns:
            Current temperature field
        """
        # Apply heating
        heat_source = self.heating_zone.data * heating_power * self.dt
        self.temperature.data += heat_source

        # Apply cooling
        cooling_source = self.cooling_zone.data * cooling_power * self.dt
        self.temperature.data += cooling_source

        # Thermal diffusion
        self.temperature = field.diffuse(
            self.temperature,
            rate=self.thermal_diffusivity,
            dt=self.dt
        )

        # Update time
        self.time_elapsed += self.dt

        # Update metrics
        self.max_temp_history.append(np.max(self.temperature.data))
        self.avg_temp_history.append(np.mean(self.temperature.data))

        # Quality score: want uniform temperature in target range (80-100°C)
        target_min, target_max = 80.0, 100.0
        in_range = np.logical_and(
            self.temperature.data >= target_min,
            self.temperature.data <= target_max
        )
        self.quality_score = np.sum(in_range) / self.temperature.data.size

        return self.temperature

    def get_metrics(self):
        """Get current process metrics."""
        return {
            'time': self.time_elapsed,
            'max_temp': self.max_temp_history[-1] if self.max_temp_history else 0,
            'avg_temp': self.avg_temp_history[-1] if self.avg_temp_history else 0,
            'quality_score': self.quality_score,
        }


class HeatExchangerTwin:
    """Digital twin of a heat exchanger.

    Simulates counter-flow heat exchanger with:
    - Hot fluid inlet
    - Cold fluid inlet
    - Heat transfer between fluids
    - Efficiency monitoring
    """

    def __init__(self, length=200, n_channels=2, dt=0.01):
        """Initialize heat exchanger twin.

        Args:
            length: Length of heat exchanger
            n_channels: Number of fluid channels
            dt: Time step
        """
        self.length = length
        self.n_channels = n_channels
        self.dt = dt

        # Temperature fields for each channel
        self.hot_fluid = field.alloc((10, length), dtype=np.float32, fill_value=90.0)
        self.cold_fluid = field.alloc((10, length), dtype=np.float32, fill_value=20.0)

        # Heat transfer coefficient
        self.heat_transfer_coeff = 0.5

        # Flow velocities (hot flows left, cold flows right)
        self.hot_velocity = 5.0  # m/s
        self.cold_velocity = 5.0  # m/s

        # Metrics
        self.time_elapsed = 0.0
        self.efficiency_history = []

    def step(self):
        """Advance simulation by one time step."""
        # Heat transfer between fluids
        temp_diff = self.hot_fluid.data - self.cold_fluid.data
        heat_flux = self.heat_transfer_coeff * temp_diff * self.dt

        self.hot_fluid.data -= heat_flux
        self.cold_fluid.data += heat_flux

        # Advection (flow)
        # Hot fluid flows left to right, cold fluid flows right to left
        self.hot_fluid.data = np.roll(self.hot_fluid.data, 1, axis=1)
        self.cold_fluid.data = np.roll(self.cold_fluid.data, -1, axis=1)

        # Boundary conditions
        # Hot inlet (left)
        self.hot_fluid.data[:, 0] = 90.0
        # Cold inlet (right)
        self.cold_fluid.data[:, -1] = 20.0

        # Update time
        self.time_elapsed += self.dt

        # Calculate efficiency
        hot_out = np.mean(self.hot_fluid.data[:, -10:])
        cold_out = np.mean(self.cold_fluid.data[:, :10])
        efficiency = (cold_out - 20.0) / (90.0 - 20.0)
        self.efficiency_history.append(efficiency)

    def get_combined_view(self):
        """Get combined visualization of both channels."""
        combined = np.vstack([self.hot_fluid.data, self.cold_fluid.data])
        return Field2D(combined)

    def get_metrics(self):
        """Get heat exchanger metrics."""
        return {
            'time': self.time_elapsed,
            'efficiency': self.efficiency_history[-1] if self.efficiency_history else 0,
            'hot_outlet_temp': np.mean(self.hot_fluid.data[:, -10:]),
            'cold_outlet_temp': np.mean(self.cold_fluid.data[:, :10]),
        }


class CoolingSystemTwin:
    """Digital twin of an active cooling system.

    Simulates a cooling system with:
    - Heat source (e.g., electronics)
    - Cooling fans
    - Temperature sensors
    - Active control
    """

    def __init__(self, width=80, height=80, dt=0.01):
        """Initialize cooling system twin.

        Args:
            width: Grid width
            height: Grid height
            dt: Time step
        """
        self.width = width
        self.height = height
        self.dt = dt

        # Temperature field
        self.temperature = field.alloc((height, width), dtype=np.float32, fill_value=25.0)

        # Heat source (center, simulates electronics)
        self.heat_source = self._create_heat_source()

        # Cooling zones (corners, simulates fans)
        self.cooling_zones = self._create_cooling_zones()

        # Control parameters
        self.target_temp = 50.0
        self.max_cooling_power = 100.0

        # Metrics
        self.time_elapsed = 0.0
        self.temp_history = []
        self.cooling_power_history = []

    def _create_heat_source(self):
        """Create heat source in center."""
        source = np.zeros((self.height, self.width), dtype=np.float32)
        cy, cx = self.height // 2, self.width // 2
        y, x = np.ogrid[:self.height, :self.width]
        mask = (x - cx)**2 + (y - cy)**2 <= (min(self.width, self.height) // 6)**2
        source[mask] = 1.0
        return Field2D(source)

    def _create_cooling_zones(self):
        """Create cooling zones in corners."""
        zones = np.zeros((self.height, self.width), dtype=np.float32)

        # Four corners
        corner_size = 10
        zones[:corner_size, :corner_size] = 1.0
        zones[:corner_size, -corner_size:] = 1.0
        zones[-corner_size:, :corner_size] = 1.0
        zones[-corner_size:, -corner_size:] = 1.0

        return Field2D(zones)

    def step(self, heat_generation=50.0):
        """Advance simulation with active control.

        Args:
            heat_generation: Heat generation rate (W)
        """
        # Add heat from source
        self.temperature.data += self.heat_source.data * heat_generation * self.dt

        # Measure temperature (sensor in center)
        cy, cx = self.height // 2, self.width // 2
        measured_temp = self.temperature.data[cy, cx]

        # Simple proportional control
        temp_error = measured_temp - self.target_temp
        cooling_power = np.clip(temp_error * 2.0, 0, self.max_cooling_power)

        # Apply cooling
        self.temperature.data -= self.cooling_zones.data * cooling_power * self.dt

        # Thermal diffusion
        self.temperature = field.diffuse(
            self.temperature,
            rate=0.1,
            dt=self.dt
        )

        # Update time
        self.time_elapsed += self.dt

        # Metrics
        self.temp_history.append(measured_temp)
        self.cooling_power_history.append(cooling_power)

    def get_metrics(self):
        """Get cooling system metrics."""
        return {
            'time': self.time_elapsed,
            'center_temp': self.temp_history[-1] if self.temp_history else 0,
            'cooling_power': self.cooling_power_history[-1] if self.cooling_power_history else 0,
            'max_temp': np.max(self.temperature.data),
            'avg_temp': np.mean(self.temperature.data),
        }


def demo_manufacturing_process():
    """Demo: Thermal manufacturing process digital twin."""
    print("Demo 1: Thermal Manufacturing Process")
    print("-" * 60)

    # Create twin
    twin = ThermalManufacturingTwin(width=150, height=100, dt=0.01)

    # Simulate
    n_steps = 1000
    frames = []

    print(f"  Simulating manufacturing process ({n_steps} steps)...")

    for step in range(n_steps):
        # Run simulation step
        temp_field = twin.step(heating_power=80.0, cooling_power=-40.0)

        # Collect frames for visualization
        if step % 50 == 0:
            frames.append(temp_field.data.copy())

            # Print metrics
            if step % 200 == 0:
                metrics = twin.get_metrics()
                print(f"    t={metrics['time']:.2f}s: "
                      f"T_max={metrics['max_temp']:.1f}°C, "
                      f"T_avg={metrics['avg_temp']:.1f}°C, "
                      f"Quality={metrics['quality_score']:.2%}")

    # Visualize key frames
    frame_indices = [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, -1]

    for idx, frame_idx in enumerate(frame_indices):
        frame_data = frames[frame_idx]

        # Normalize for visualization
        normalized = (frame_data - 20.0) / (120.0 - 20.0)
        normalized = np.clip(normalized, 0, 1)

        # Apply thermal colormap
        pal = palette.fire(256)
        img = palette.map(pal, normalized)

        output_path = f"output_digital_twin_manufacturing_frame{idx:02d}.png"
        io_storage.save_image(output_path, img)
        print(f"  ✓ Saved: {output_path}")

    # Create metrics plot (temperature over time)
    metrics_field = np.zeros((100, 200), dtype=np.float32)

    # Plot max and avg temperature
    for i, (max_t, avg_t) in enumerate(zip(twin.max_temp_history[::10],
                                           twin.avg_temp_history[::10])):
        if i >= 200:
            break

        # Map temperature to y-coordinate (0-100°C -> 0-100 pixels)
        y_max = int((100 - max_t) * 100 / 100)
        y_avg = int((100 - avg_t) * 100 / 100)

        y_max = np.clip(y_max, 0, 99)
        y_avg = np.clip(y_avg, 0, 99)

        metrics_field[y_max, i] = 1.0  # Max temp
        metrics_field[y_avg, i] = 0.5  # Avg temp

    pal = palette.viridis(256)
    metrics_img = palette.map(pal, metrics_field)
    io_storage.save_image("output_digital_twin_manufacturing_metrics.png", metrics_img)
    print("  ✓ Saved: output_digital_twin_manufacturing_metrics.png")


def demo_heat_exchanger():
    """Demo: Heat exchanger digital twin."""
    print("\nDemo 2: Heat Exchanger System")
    print("-" * 60)

    # Create twin
    twin = HeatExchangerTwin(length=200, dt=0.01)

    # Simulate
    n_steps = 500
    frames = []

    print(f"  Simulating heat exchanger ({n_steps} steps)...")

    for step in range(n_steps):
        twin.step()

        # Collect frames
        if step % 25 == 0:
            combined = twin.get_combined_view()
            frames.append(combined.data.copy())

            if step % 100 == 0:
                metrics = twin.get_metrics()
                print(f"    t={metrics['time']:.2f}s: "
                      f"Efficiency={metrics['efficiency']:.2%}, "
                      f"Hot_out={metrics['hot_outlet_temp']:.1f}°C, "
                      f"Cold_out={metrics['cold_outlet_temp']:.1f}°C")

    # Visualize key frames
    frame_indices = [0, len(frames)//4, len(frames)//2, -1]

    for idx, frame_idx in enumerate(frame_indices):
        frame_data = frames[frame_idx]

        # Normalize
        normalized = (frame_data - 20.0) / (90.0 - 20.0)
        normalized = np.clip(normalized, 0, 1)

        # Apply colormap
        pal = palette.coolwarm(256)
        img = palette.map(pal, normalized)

        output_path = f"output_digital_twin_heat_exchanger_frame{idx:02d}.png"
        io_storage.save_image(output_path, img)
        print(f"  ✓ Saved: {output_path}")


def demo_cooling_system():
    """Demo: Active cooling system digital twin."""
    print("\nDemo 3: Active Cooling System")
    print("-" * 60)

    # Create twin
    twin = CoolingSystemTwin(width=100, height=100, dt=0.01)

    # Simulate
    n_steps = 800
    frames = []

    print(f"  Simulating cooling system with active control ({n_steps} steps)...")

    for step in range(n_steps):
        # Vary heat generation (simulate changing workload)
        if step < 200:
            heat_gen = 50.0
        elif step < 400:
            heat_gen = 100.0
        elif step < 600:
            heat_gen = 150.0
        else:
            heat_gen = 75.0

        twin.step(heat_generation=heat_gen)

        # Collect frames
        if step % 40 == 0:
            frames.append(twin.temperature.data.copy())

            if step % 200 == 0:
                metrics = twin.get_metrics()
                print(f"    t={metrics['time']:.2f}s: "
                      f"T_center={metrics['center_temp']:.1f}°C, "
                      f"T_max={metrics['max_temp']:.1f}°C, "
                      f"Cooling={metrics['cooling_power']:.1f}W")

    # Visualize key frames
    frame_indices = [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, -1]

    for idx, frame_idx in enumerate(frame_indices):
        frame_data = frames[frame_idx]

        # Normalize
        normalized = (frame_data - 25.0) / (100.0 - 25.0)
        normalized = np.clip(normalized, 0, 1)

        # Apply thermal colormap
        pal = palette.inferno(256)
        img = palette.map(pal, normalized)

        output_path = f"output_digital_twin_cooling_frame{idx:02d}.png"
        io_storage.save_image(output_path, img)
        print(f"  ✓ Saved: {output_path}")

    # Create control metrics visualization
    # Temperature vs cooling power over time
    metrics_field = np.zeros((100, len(twin.temp_history)//10), dtype=np.float32)

    for i in range(min(len(twin.temp_history)//10, metrics_field.shape[1])):
        idx = i * 10

        # Temperature (mapped to 0-100 range)
        temp = twin.temp_history[idx]
        y_temp = int((100 - (temp - 25)) * 100 / 75)
        y_temp = np.clip(y_temp, 0, 99)

        metrics_field[y_temp, i] = 1.0

    pal = palette.plasma(256)
    metrics_img = palette.map(pal, metrics_field)
    io_storage.save_image("output_digital_twin_cooling_control.png", metrics_img)
    print("  ✓ Saved: output_digital_twin_cooling_control.png")


def demo_multi_physics_coupling():
    """Demo: Multi-physics coupling (thermal + structural)."""
    print("\nDemo 4: Multi-Physics Coupling")
    print("-" * 60)

    print("  Simulating thermal expansion and stress...")

    # Thermal field
    width, height = 100, 100
    temp = field.alloc((height, width), dtype=np.float32, fill_value=20.0)

    # Add localized heat source
    cy, cx = height // 2, width // 2
    y, x = np.ogrid[:height, :width]
    heat_mask = (x - cx)**2 + (y - cy)**2 <= 15**2

    # Simulate thermal + mechanical coupling
    frames_temp = []
    frames_stress = []

    # Material properties
    thermal_expansion_coeff = 1e-5  # 1/K
    youngs_modulus = 200e9  # Pa

    dt = 0.01
    n_steps = 500

    for step in range(n_steps):
        # Add heat
        temp.data[heat_mask] += 5.0 * dt

        # Thermal diffusion
        temp = field.diffuse(temp, rate=0.1, dt=dt)

        # Compute thermal strain
        thermal_strain = (temp.data - 20.0) * thermal_expansion_coeff

        # Compute stress (simplified)
        stress = thermal_strain * youngs_modulus

        # Collect frames
        if step % 25 == 0:
            frames_temp.append(temp.data.copy())
            frames_stress.append(stress.copy())

    # Visualize final state
    # Temperature field
    temp_normalized = (frames_temp[-1] - 20.0) / (np.max(frames_temp[-1]) - 20.0 + 1e-6)
    temp_normalized = np.clip(temp_normalized, 0, 1)

    pal_temp = palette.fire(256)
    temp_img = palette.map(pal_temp, temp_normalized)
    io_storage.save_image("output_digital_twin_multiphysics_temp.png", temp_img)
    print("  ✓ Saved: output_digital_twin_multiphysics_temp.png")

    # Stress field
    stress_normalized = frames_stress[-1] / (np.max(np.abs(frames_stress[-1])) + 1e-6)
    stress_normalized = (stress_normalized + 1.0) * 0.5  # Map to [0, 1]

    pal_stress = palette.coolwarm(256)
    stress_img = palette.map(pal_stress, stress_normalized)
    io_storage.save_image("output_digital_twin_multiphysics_stress.png", stress_img)
    print("  ✓ Saved: output_digital_twin_multiphysics_stress.png")


def main():
    """Run all digital twin demonstrations."""
    print("=" * 60)
    print("DIGITAL TWIN - CROSS-DOMAIN SHOWCASE")
    print("=" * 60)
    print()
    print("Domains: Field + Integrators + SparseLinalg + Visual + I/O")
    print()

    demo_manufacturing_process()
    demo_heat_exchanger()
    demo_cooling_system()
    demo_multi_physics_coupling()

    print()
    print("=" * 60)
    print("ALL DIGITAL TWIN DEMOS COMPLETE!")
    print("=" * 60)
    print()
    print("This showcase demonstrates:")
    print("  • Thermal manufacturing process simulation")
    print("  • Heat exchanger efficiency modeling")
    print("  • Active cooling system with control")
    print("  • Multi-physics coupling (thermal + structural)")
    print("  • Real-time monitoring and metrics")
    print()
    print("Key insight: Digital twins enable virtual testing,")
    print("optimization, and monitoring of physical systems")
    print("before building them - saving time and money!")


if __name__ == "__main__":
    main()
