"""
Physics → Audio Domain Interfaces

Interfaces for sonification of physical events and fluid-acoustic coupling.
"""

from typing import Any, Dict, List, Optional, Tuple, Type
import numpy as np

from .base import DomainInterface


class PhysicsToAudioInterface(DomainInterface):
    """
    Physics → Audio: Sonification of physical events.

    Use cases:
    - Collision forces → percussion synthesis
    - Body velocities → pitch/volume
    - Contact points → spatial audio
    """

    source_domain = "physics"
    target_domain = "audio"

    def __init__(
        self,
        events,
        mapping: Dict[str, str],
        sample_rate: int = 48000
    ):
        """
        Args:
            events: Physical events (collisions, contacts, etc.)
            mapping: Dict mapping physics properties to audio parameters
                     e.g., {"impulse": "amplitude", "body_id": "pitch"}
            sample_rate: Audio sample rate
        """
        super().__init__(source_data=events)
        self.events = events
        self.mapping = mapping
        self.sample_rate = sample_rate

    def transform(self, source_data: Any) -> Dict[str, np.ndarray]:
        """
        Convert physics events to audio parameters.

        Returns:
            Dict with keys: 'triggers', 'amplitudes', 'frequencies', 'positions'
        """
        events = source_data if source_data is not None else self.events

        audio_params = {
            'triggers': [],
            'amplitudes': [],
            'frequencies': [],
            'positions': [],
        }

        for event in events:
            # Extract physics properties based on mapping
            if "impulse" in self.mapping:
                audio_param = self.mapping["impulse"]
                impulse = getattr(event, "impulse", 1.0)

                if audio_param == "amplitude":
                    # Map impulse magnitude to volume (0-1)
                    amplitude = np.clip(impulse / 100.0, 0.0, 1.0)
                    audio_params['amplitudes'].append(amplitude)

            if "body_id" in self.mapping:
                audio_param = self.mapping["body_id"]
                body_id = getattr(event, "body_id", 0)

                if audio_param == "pitch":
                    # Map body ID to frequency (C major scale)
                    frequencies = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]
                    freq = frequencies[body_id % len(frequencies)]
                    audio_params['frequencies'].append(freq)

            if "position" in self.mapping:
                pos = getattr(event, "position", (0, 0))
                audio_params['positions'].append(pos)

            # Trigger time (in samples)
            trigger_time = getattr(event, "time", 0.0)
            audio_params['triggers'].append(int(trigger_time * self.sample_rate))

        return audio_params

    def validate(self) -> bool:
        """Check events and mapping are valid."""
        if not self.events or not self.mapping:
            return False

        valid_physics_props = ["impulse", "body_id", "position", "velocity", "time"]
        valid_audio_params = ["amplitude", "pitch", "pan", "duration"]

        for phys_prop, audio_param in self.mapping.items():
            if phys_prop not in valid_physics_props:
                raise ValueError(f"Unknown physics property: {phys_prop}")
            if audio_param not in valid_audio_params:
                raise ValueError(f"Unknown audio parameter: {audio_param}")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {
            'events': List,
            'mapping': Dict[str, str],
        }

    def get_output_interface(self) -> Dict[str, Type]:
        return {
            'audio_params': Dict[str, np.ndarray],
        }


class FluidToAcousticsInterface(DomainInterface):
    """
    Fluid → Acoustics: Couple fluid pressure to acoustic wave propagation.

    Use cases:
    - CFD pressure fields → acoustic wave equation
    - Turbulent flow → aeroacoustic sound
    - Vortex shedding → acoustic radiation
    - Fluid-structure interaction → sound generation
    """

    source_domain = "fluid"
    target_domain = "acoustics"

    def __init__(
        self,
        pressure_fields: List[np.ndarray],
        fluid_dt: float = 0.01,
        speed_of_sound: float = 5.0,
        coupling_strength: float = 0.1,
        diffusion_coeff: Optional[float] = None
    ):
        """
        Args:
            pressure_fields: Time series of fluid pressure fields (List of 2D arrays)
            fluid_dt: Fluid simulation timestep
            speed_of_sound: Acoustic wave speed (grid units per timestep)
            coupling_strength: Strength of fluid→acoustic coupling
            diffusion_coeff: Diffusion coefficient for wave propagation (defaults to speed_of_sound)
        """
        super().__init__(source_data=pressure_fields)
        self.pressure_fields = pressure_fields
        self.fluid_dt = fluid_dt
        self.speed_of_sound = speed_of_sound
        self.coupling_strength = coupling_strength
        self.diffusion_coeff = diffusion_coeff or speed_of_sound

    def transform(self, source_data: Any) -> List[np.ndarray]:
        """
        Convert fluid pressure fields to acoustic pressure fields.

        The acoustic wave equation couples to fluid pressure gradients:
        d²p_acoustic/dt² = c² ∇²p_acoustic + S(p_fluid)

        Returns:
            List of acoustic pressure fields (2D numpy arrays)
        """
        from morphogen.stdlib import field

        pressure_fields = source_data if source_data is not None else self.pressure_fields

        if not pressure_fields:
            raise ValueError("No pressure fields provided")

        acoustic_fields = []

        # Initialize acoustic field with same shape as fluid field
        grid_shape = pressure_fields[0].data if hasattr(pressure_fields[0], 'data') else pressure_fields[0]
        grid_shape = grid_shape.shape
        acoustic = field.alloc(grid_shape, fill_value=0.0)

        # Propagate acoustic waves coupled to fluid pressure
        for i, pressure in enumerate(pressure_fields):
            # Extract data if Field2D object, otherwise use directly
            pressure_data = pressure.data if hasattr(pressure, 'data') else pressure

            # Couple fluid pressure to acoustic source term
            # Acoustic pressure responds to fluid pressure gradients
            source = pressure_data * self.coupling_strength

            # Wave equation: d²p/dt² = c² ∇²p + source
            # Simplified with diffusion approximation
            acoustic.data += source

            # Propagate (diffusion as wave approximation)
            acoustic = field.diffuse(
                acoustic,
                rate=self.diffusion_coeff,
                dt=self.fluid_dt
            )

            # Damping (acoustic energy dissipation)
            acoustic.data *= 0.98

            acoustic_fields.append(acoustic.copy())

        return acoustic_fields

    def validate(self) -> bool:
        """Check pressure fields are valid."""
        if not self.pressure_fields:
            return False

        if not isinstance(self.pressure_fields, list):
            raise TypeError("Pressure fields must be a list")

        # Check first field has valid shape
        first_field = self.pressure_fields[0]
        field_data = first_field.data if hasattr(first_field, 'data') else first_field

        if not isinstance(field_data, np.ndarray):
            raise TypeError("Pressure field must be numpy array or Field2D")

        if len(field_data.shape) != 2:
            raise ValueError("Pressure field must be 2D")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {
            'pressure_fields': List,
            'fluid_dt': float,
        }

    def get_output_interface(self) -> Dict[str, Type]:
        return {'acoustic_fields': List}


class AcousticsToAudioInterface(DomainInterface):
    """
    Acoustics → Audio: Sample acoustic field at microphones and synthesize audio.

    Use cases:
    - Acoustic pressure → audio waveform
    - Virtual microphone sampling
    - Spatial audio from acoustic fields
    - CFD aeroacoustics → audio rendering
    """

    source_domain = "acoustics"
    target_domain = "audio"

    def __init__(
        self,
        acoustic_fields: List[np.ndarray],
        mic_positions: List[Tuple[int, int]],
        fluid_dt: float = 0.01,
        sample_rate: int = 44100,
        add_turbulence_noise: bool = True,
        noise_level: float = 0.05
    ):
        """
        Args:
            acoustic_fields: Time series of acoustic pressure fields
            mic_positions: List of (y, x) microphone positions in grid coordinates
            fluid_dt: Acoustic simulation timestep
            sample_rate: Audio sample rate
            add_turbulence_noise: Whether to add turbulence detail
            noise_level: Level of turbulence noise (0.0 to 1.0)
        """
        super().__init__(source_data=acoustic_fields)
        self.acoustic_fields = acoustic_fields
        self.mic_positions = mic_positions
        self.fluid_dt = fluid_dt
        self.sample_rate = sample_rate
        self.add_turbulence_noise = add_turbulence_noise
        self.noise_level = noise_level

    def transform(self, source_data: Any) -> Any:
        """
        Convert acoustic pressure fields to audio waveform.

        Samples acoustic pressure at microphone positions over time,
        interpolates to audio sample rate, and creates audio buffer.

        Returns:
            AudioBuffer (mono if 1 mic, stereo if 2+ mics)
        """
        from morphogen.stdlib import audio

        acoustic_fields = source_data if source_data is not None else self.acoustic_fields

        if not acoustic_fields:
            raise ValueError("No acoustic fields provided")

        num_acoustic_samples = len(acoustic_fields)
        acoustic_duration = num_acoustic_samples * self.fluid_dt
        num_audio_samples = int(acoustic_duration * self.sample_rate)

        # Sample acoustic pressure at each microphone
        channels = []

        for mic_y, mic_x in self.mic_positions:
            # Sample pressure at this microphone over time
            mic_signal = []
            for acoustic_field in acoustic_fields:
                # Extract data if Field2D object
                field_data = acoustic_field.data if hasattr(acoustic_field, 'data') else acoustic_field

                # Sample at microphone position
                pressure_value = field_data[mic_y, mic_x]
                mic_signal.append(pressure_value)

            mic_signal = np.array(mic_signal, dtype=np.float32)

            # Interpolate to audio sample rate
            acoustic_time = np.arange(len(mic_signal)) * self.fluid_dt
            audio_time = np.arange(num_audio_samples) / self.sample_rate
            interpolated = np.interp(audio_time, acoustic_time, mic_signal)

            # Add turbulence noise for realism
            if self.add_turbulence_noise:
                envelope = np.abs(interpolated)
                noise = np.random.randn(len(interpolated)).astype(np.float32) * envelope * self.noise_level
                interpolated += noise

            channels.append(interpolated)

        # Create audio buffer (mono or multi-channel)
        if len(channels) == 1:
            audio_data = channels[0]
        else:
            audio_data = np.stack(channels, axis=1)

        # Normalize to prevent clipping
        peak = np.max(np.abs(audio_data))
        if peak > 0:
            audio_data = audio_data / peak * 0.7  # Leave headroom

        return audio.AudioBuffer(data=audio_data, sample_rate=self.sample_rate)

    def validate(self) -> bool:
        """Check acoustic fields and microphone positions are valid."""
        if not self.acoustic_fields:
            return False

        if not isinstance(self.acoustic_fields, list):
            raise TypeError("Acoustic fields must be a list")

        if not self.mic_positions:
            raise ValueError("At least one microphone position required")

        # Check microphone positions are valid
        first_field = self.acoustic_fields[0]
        field_data = first_field.data if hasattr(first_field, 'data') else first_field
        grid_shape = field_data.shape

        for mic_y, mic_x in self.mic_positions:
            if not (0 <= mic_y < grid_shape[0] and 0 <= mic_x < grid_shape[1]):
                raise ValueError(
                    f"Microphone position ({mic_y}, {mic_x}) out of bounds "
                    f"for grid shape {grid_shape}"
                )

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {
            'acoustic_fields': List,
            'mic_positions': List[Tuple[int, int]],
        }

    def get_output_interface(self) -> Dict[str, Type]:
        return {'audio_buffer': Any}  # AudioBuffer type
