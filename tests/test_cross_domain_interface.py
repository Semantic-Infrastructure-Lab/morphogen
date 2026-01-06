"""
Tests for cross-domain interface transformations.

Tests the DomainInterface base class and concrete implementations for:
- Field → Agent (sample field at agent positions)
- Agent → Field (deposit agent properties to field)
- Physics → Audio (sonification of physical events)
- Fluid → Acoustics (pressure coupling to acoustic wave propagation)
- Acoustics → Audio (acoustic field sampling to audio synthesis)
"""

import numpy as np
from morphogen.cross_domain import (
    FieldToAgentInterface,
    AgentToFieldInterface,
    PhysicsToAudioInterface,
    FluidToAcousticsInterface,
    AcousticsToAudioInterface,
    DomainInterface,
)
from morphogen.cross_domain.registry import CrossDomainRegistry
from morphogen.cross_domain.validators import (
    validate_cross_domain_flow,
    validate_field_data,
    validate_agent_positions,
    check_dimensional_compatibility,
)


def test_field_to_agent_basic():
    """Test basic Field → Agent transform (sample field at positions)."""
    # Create a simple field (10x10 grid with gradient)
    field = np.arange(100, dtype=np.float32).reshape(10, 10)

    # Agent positions (5 agents in the field)
    positions = np.array([
        [0.0, 0.0],  # Top-left
        [9.0, 9.0],  # Bottom-right
        [5.0, 5.0],  # Center
        [2.0, 7.0],  # Other positions
        [7.0, 3.0],
    ], dtype=np.float32)

    # Create transform interface
    interface = FieldToAgentInterface(field, positions)

    # Validate
    assert interface.validate()

    # Transform (sample field at positions)
    sampled_values = interface.transform(field)

    # Check shape
    assert sampled_values.shape == (5,)

    # Check values are reasonable (should be between 0 and 99)
    assert np.all(sampled_values >= 0)
    assert np.all(sampled_values <= 99)

    print("✓ Field → Agent basic test passed")


def test_field_to_agent_vector_field():
    """Test Field → Agent with vector field."""
    # Create a 2D velocity field (10x10 grid, 2 components)
    field = np.zeros((10, 10, 2), dtype=np.float32)
    field[:, :, 0] = 1.0  # vx = 1.0
    field[:, :, 1] = -0.5  # vy = -0.5

    # Agent positions
    positions = np.array([
        [5.0, 5.0],
        [3.0, 7.0],
    ], dtype=np.float32)

    # Create and apply transform
    interface = FieldToAgentInterface(field, positions)
    sampled_values = interface(field)

    # Should return Nx2 array (velocity components)
    assert sampled_values.shape == (2, 2)

    # Check values
    assert np.allclose(sampled_values[:, 0], 1.0)
    assert np.allclose(sampled_values[:, 1], -0.5)

    print("✓ Field → Agent vector field test passed")


def test_agent_to_field_accumulate():
    """Test Agent → Field deposit (accumulate method)."""
    # Agent positions
    positions = np.array([
        [0, 0],
        [0, 0],  # Same position (should accumulate)
        [5, 5],
        [9, 9],
    ], dtype=np.float32)

    # Agent values to deposit
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    # Create interface
    interface = AgentToFieldInterface(
        positions,
        values,
        field_shape=(10, 10),
        method="accumulate"
    )

    # Validate
    assert interface.validate()

    # Transform
    field = interface((positions, values))

    # Check shape
    assert field.shape == (10, 10)

    # Check accumulation at (0, 0): should be 1.0 + 2.0 = 3.0
    assert field[0, 0] == 3.0

    # Check other positions
    assert field[5, 5] == 3.0
    assert field[9, 9] == 4.0

    # Most cells should be zero
    assert np.sum(field == 0) == (100 - 3)  # 3 unique positions

    print("✓ Agent → Field accumulate test passed")


def test_agent_to_field_average():
    """Test Agent → Field deposit (average method)."""
    positions = np.array([
        [5, 5],
        [5, 5],  # Same position
        [5, 5],  # Same position (3 agents at same spot)
    ], dtype=np.float32)

    values = np.array([3.0, 6.0, 9.0], dtype=np.float32)

    interface = AgentToFieldInterface(
        positions,
        values,
        field_shape=(10, 10),
        method="average"
    )

    field = interface.transform((positions, values))

    # Average of [3, 6, 9] = 6.0
    assert field[5, 5] == 6.0

    print("✓ Agent → Field average test passed")


def test_agent_to_field_max():
    """Test Agent → Field deposit (max method)."""
    positions = np.array([
        [2, 2],
        [2, 2],
        [2, 2],
    ], dtype=np.float32)

    values = np.array([1.0, 5.0, 3.0], dtype=np.float32)

    interface = AgentToFieldInterface(
        positions,
        values,
        field_shape=(10, 10),
        method="max"
    )

    field = interface.transform((positions, values))

    # Max of [1, 5, 3] = 5.0
    assert field[2, 2] == 5.0

    print("✓ Agent → Field max test passed")


def test_physics_to_audio_mapping():
    """Test Physics → Audio sonification."""
    # Mock physics events
    class MockCollisionEvent:
        def __init__(self, impulse, body_id, position, time):
            self.impulse = impulse
            self.body_id = body_id
            self.position = position
            self.time = time

    events = [
        MockCollisionEvent(impulse=50.0, body_id=0, position=(0, 0), time=0.0),
        MockCollisionEvent(impulse=100.0, body_id=1, position=(10, 0), time=0.1),
        MockCollisionEvent(impulse=25.0, body_id=2, position=(5, 5), time=0.2),
    ]

    # Create mapping
    mapping = {
        "impulse": "amplitude",
        "body_id": "pitch",
        "position": "pan",
    }

    interface = PhysicsToAudioInterface(events, mapping, sample_rate=48000)

    # Validate
    assert interface.validate()

    # Transform
    audio_params = interface.transform(events)

    # Check structure
    assert "triggers" in audio_params
    assert "amplitudes" in audio_params
    assert "frequencies" in audio_params
    assert "positions" in audio_params

    # Check values
    assert len(audio_params["triggers"]) == 3
    assert len(audio_params["amplitudes"]) == 3
    assert len(audio_params["frequencies"]) == 3

    # Check amplitude mapping (impulse → amplitude)
    # impulse=50 → amplitude=0.5, impulse=100 → amplitude=1.0
    assert audio_params["amplitudes"][0] == 0.5
    assert audio_params["amplitudes"][1] == 1.0

    # Check frequency mapping (body_id → pitch)
    # Should map to C major scale frequencies
    assert audio_params["frequencies"][0] >= 200  # Reasonable frequency
    assert audio_params["frequencies"][1] >= 200

    # Check trigger times (in samples)
    assert audio_params["triggers"][0] == 0
    assert audio_params["triggers"][1] == int(0.1 * 48000)
    assert audio_params["triggers"][2] == int(0.2 * 48000)

    print("✓ Physics → Audio mapping test passed")


def test_cross_domain_registry():
    """Test cross-domain registry lookup."""
    # Check built-in transforms are registered
    assert CrossDomainRegistry.has_transform("field", "agent")
    assert CrossDomainRegistry.has_transform("agent", "field")
    assert CrossDomainRegistry.has_transform("physics", "audio")

    # Get transform class
    field_to_agent = CrossDomainRegistry.get("field", "agent")
    assert field_to_agent == FieldToAgentInterface

    # List all transforms for field domain
    field_transforms = CrossDomainRegistry.list_transforms("field")
    assert ("field", "agent") in field_transforms

    print("✓ Cross-domain registry test passed")


def test_validators():
    """Test cross-domain validators."""
    # Test field validation
    field = np.zeros((10, 10), dtype=np.float32)
    assert validate_field_data(field, allow_vector=False)

    # Test agent position validation
    positions = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    assert validate_agent_positions(positions, ndim=2)

    # Test dimensional compatibility
    assert check_dimensional_compatibility((10, 10), positions)

    print("✓ Validators test passed")


def test_fluid_to_acoustics_coupling():
    """Test Fluid → Acoustics transform (pressure coupling)."""
    from morphogen.stdlib import field

    # Create fluid pressure fields (time series)
    grid_size = 32
    num_steps = 10

    pressure_fields = []
    for t in range(num_steps):
        # Create pressure field with some spatial variation
        pressure = field.alloc((grid_size, grid_size), fill_value=0.0)
        # Add a pressure pulse that evolves over time
        y, x = np.mgrid[0:grid_size, 0:grid_size]
        center_x, center_y = grid_size // 2, grid_size // 2
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        pressure.data = np.exp(-(r - t)**2 / 10.0)
        pressure_fields.append(pressure)

    # Create interface
    interface = FluidToAcousticsInterface(
        pressure_fields=pressure_fields,
        fluid_dt=0.01,
        speed_of_sound=5.0,
        coupling_strength=0.1
    )

    # Validate
    assert interface.validate()

    # Transform
    acoustic_fields = interface.transform(pressure_fields)

    # Check output
    assert len(acoustic_fields) == num_steps
    assert acoustic_fields[0].shape == (grid_size, grid_size)

    # Check that acoustic fields contain data
    for acoustic_field in acoustic_fields:
        assert isinstance(acoustic_field.data, np.ndarray)
        assert acoustic_field.data.shape == (grid_size, grid_size)

    print("✓ Fluid → Acoustics coupling test passed")


def test_acoustics_to_audio_sampling():
    """Test Acoustics → Audio transform (microphone sampling)."""
    from morphogen.stdlib import field

    # Create acoustic pressure fields (time series)
    grid_size = 32
    num_steps = 50  # 50 acoustic timesteps
    fluid_dt = 0.02  # 50 Hz acoustic simulation

    acoustic_fields = []
    for t in range(num_steps):
        # Create acoustic field with wave propagation
        acoustic = field.alloc((grid_size, grid_size), fill_value=0.0)
        y, x = np.mgrid[0:grid_size, 0:grid_size]
        # Traveling wave
        acoustic.data = np.sin(x * 0.2 + t * 0.5) * np.exp(-0.05 * t)
        acoustic_fields.append(acoustic)

    # Microphone positions
    mic_positions = [
        (grid_size // 4, grid_size // 2),      # Left mic
        (3 * grid_size // 4, grid_size // 2),  # Right mic
    ]

    # Create interface
    interface = AcousticsToAudioInterface(
        acoustic_fields=acoustic_fields,
        mic_positions=mic_positions,
        fluid_dt=fluid_dt,
        sample_rate=44100,
        add_turbulence_noise=False  # Disable noise for deterministic test
    )

    # Validate
    assert interface.validate()

    # Transform
    audio_buffer = interface.transform(acoustic_fields)

    # Check output
    assert hasattr(audio_buffer, 'data')
    assert hasattr(audio_buffer, 'sample_rate')
    assert audio_buffer.sample_rate == 44100

    # Should be stereo (2 channels)
    assert audio_buffer.is_stereo
    assert audio_buffer.data.shape[1] == 2

    # Check duration matches expected
    expected_duration = num_steps * fluid_dt
    assert abs(audio_buffer.duration - expected_duration) < 0.1

    # Check audio data is normalized (peak <= 1.0)
    peak = np.max(np.abs(audio_buffer.data))
    assert peak <= 1.0

    print("✓ Acoustics → Audio sampling test passed")


def test_fluid_acoustics_audio_pipeline():
    """Test complete 3-domain pipeline: Fluid → Acoustics → Audio."""
    from morphogen.stdlib import field

    # 1. Create fluid pressure fields
    grid_size = 16  # Small for fast test
    num_steps = 20

    pressure_fields = []
    for t in range(num_steps):
        pressure = field.alloc((grid_size, grid_size), fill_value=0.0)
        # Simple vortex-like pattern
        y, x = np.mgrid[0:grid_size, 0:grid_size]
        pressure.data = np.sin(x * 0.5) * np.cos(y * 0.5) * np.exp(-0.1 * t)
        pressure_fields.append(pressure)

    # 2. Transform Fluid → Acoustics
    fluid_to_acoustics = FluidToAcousticsInterface(
        pressure_fields=pressure_fields,
        fluid_dt=0.02,
        coupling_strength=0.15
    )
    acoustic_fields = fluid_to_acoustics.transform(pressure_fields)

    # 3. Transform Acoustics → Audio
    mic_positions = [(grid_size // 2, grid_size // 2)]  # Single mic
    acoustics_to_audio = AcousticsToAudioInterface(
        acoustic_fields=acoustic_fields,
        mic_positions=mic_positions,
        fluid_dt=0.02,
        sample_rate=22050,  # Lower sample rate for fast test
        add_turbulence_noise=False
    )
    audio_buffer = acoustics_to_audio.transform(acoustic_fields)

    # Verify complete pipeline
    assert len(acoustic_fields) == num_steps
    assert audio_buffer.sample_rate == 22050
    assert audio_buffer.data.shape[0] > 0

    # Should be mono (1 channel)
    assert not audio_buffer.is_stereo

    print("✓ Fluid → Acoustics → Audio pipeline test passed")


def test_cross_domain_registry_extended():
    """Test cross-domain registry includes new transforms."""
    # Check new transforms are registered
    assert CrossDomainRegistry.has_transform("fluid", "acoustics")
    assert CrossDomainRegistry.has_transform("acoustics", "audio")

    # Get transform classes
    fluid_to_acoustics = CrossDomainRegistry.get("fluid", "acoustics")
    assert fluid_to_acoustics == FluidToAcousticsInterface

    acoustics_to_audio = CrossDomainRegistry.get("acoustics", "audio")
    assert acoustics_to_audio == AcousticsToAudioInterface

    print("✓ Extended cross-domain registry test passed")


if __name__ == "__main__":
    test_field_to_agent_basic()
    test_field_to_agent_vector_field()
    test_agent_to_field_accumulate()
    test_agent_to_field_average()
    test_agent_to_field_max()
    test_physics_to_audio_mapping()
    test_cross_domain_registry()
    test_validators()

    # New tests for Phase 3 transforms
    test_fluid_to_acoustics_coupling()
    test_acoustics_to_audio_sampling()
    test_fluid_acoustics_audio_pipeline()
    test_cross_domain_registry_extended()

    print("\n✅ All cross-domain interface tests passed!")
