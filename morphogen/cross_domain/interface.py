"""
Domain Interface Base Classes

Provides the foundational abstractions for cross-domain data flows in Morphogen.
Based on ADR-002: Cross-Domain Architectural Patterns.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Type
from dataclasses import dataclass
import numpy as np


@dataclass
class DomainMetadata:
    """Metadata describing a domain's capabilities and interfaces."""

    name: str
    version: str
    input_types: Set[str]  # What types this domain can accept
    output_types: Set[str]  # What types this domain can provide
    dependencies: List[str]  # Other domains this depends on
    description: str


class DomainInterface(ABC):
    """
    Base class for inter-domain data flows.

    Each domain pair (source → target) that supports composition must implement
    a DomainInterface subclass that handles:
    1. Type validation
    2. Data transformation
    3. Metadata propagation

    Example:
        class FieldToAgentInterface(DomainInterface):
            source_domain = "field"
            target_domain = "agent"

            def transform(self, field_data):
                # Sample field at agent positions
                return sampled_values

            def validate(self):
                # Check field dimensions, agent count, etc.
                return True
    """

    source_domain: str = None  # Set by subclass
    target_domain: str = None  # Set by subclass

    def __init__(self, source_data: Any = None, metadata: Optional[Dict] = None):
        self.source_data = source_data
        self.metadata = metadata or {}
        self._validated = False

    @abstractmethod
    def transform(self, source_data: Any) -> Any:
        """
        Convert source domain data to target domain format.

        Args:
            source_data: Data in source domain format

        Returns:
            Data in target domain format

        Raises:
            ValueError: If data cannot be transformed
            TypeError: If data types are incompatible
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement transform()"
        )

    @abstractmethod
    def validate(self) -> bool:
        """
        Ensure data types are compatible across domains.

        Returns:
            True if transformation is valid, False otherwise

        Raises:
            CrossDomainTypeError: If types are fundamentally incompatible
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement validate()"
        )

    def get_input_interface(self) -> Dict[str, Type]:
        """
        Describe what data this interface can accept.

        Returns:
            Dict mapping parameter names to their types
        """
        return {}

    def get_output_interface(self) -> Dict[str, Type]:
        """
        Describe what data this interface can provide.

        Returns:
            Dict mapping output names to their types
        """
        return {}

    def __call__(self, source_data: Any) -> Any:
        """
        Convenience method: validate and transform in one call.

        Args:
            source_data: Data to transform

        Returns:
            Transformed data
        """
        self.source_data = source_data
        if not self._validated:
            if not self.validate():
                raise ValueError(
                    f"Cross-domain flow {self.source_domain} → {self.target_domain} "
                    f"failed validation"
                )
            self._validated = True

        return self.transform(source_data)


class DomainTransform:
    """
    Decorator for registering cross-domain transform functions.

    Example:
        @DomainTransform(source="field", target="agent")
        def field_to_agent_force(field, agent_positions):
            '''Sample field values at agent positions.'''
            return sample_field(field, agent_positions)
    """

    def __init__(
        self,
        source: str,
        target: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        input_types: Optional[Dict[str, Type]] = None,
        output_type: Optional[Type] = None,
    ):
        self.source = source
        self.target = target
        self.name = name
        self.description = description
        self.input_types = input_types or {}
        self.output_type = output_type
        self.transform_fn = None

    def __call__(self, fn):
        """Register the decorated function as a transform."""
        self.transform_fn = fn
        self.name = self.name or fn.__name__
        self.description = self.description or fn.__doc__

        # Create a DomainInterface wrapper
        class TransformInterface(DomainInterface):
            source_domain = self.source
            target_domain = self.target

            def transform(iself, source_data: Any) -> Any:
                return fn(source_data)

            def validate(iself) -> bool:
                # Basic type checking if types specified
                if self.input_types:
                    # TODO: Implement type validation
                    pass
                return True

        # Store metadata
        TransformInterface.__name__ = f"{self.source}To{self.target.capitalize()}Transform"
        TransformInterface.__doc__ = self.description

        # Register in global registry (will be implemented)
        from .registry import CrossDomainRegistry
        CrossDomainRegistry.register(self.source, self.target, TransformInterface)

        return fn


# ============================================================================
# Concrete Domain Interfaces
# ============================================================================


class FieldToAgentInterface(DomainInterface):
    """
    Field → Agent: Sample field values at agent positions.

    Use cases:
    - Flow field → force on particles
    - Temperature field → agent color/behavior
    - Density field → agent sensing
    """

    source_domain = "field"
    target_domain = "agent"

    def __init__(self, field, positions, property_name="value"):
        super().__init__(source_data=field)
        self.field = field
        self.positions = positions
        self.property_name = property_name

    def transform(self, source_data: Any) -> np.ndarray:
        """Sample field at agent positions."""
        field = source_data if source_data is not None else self.field

        # Handle different field types
        if hasattr(field, 'data'):
            field_data = field.data
        elif isinstance(field, np.ndarray):
            field_data = field
        else:
            raise TypeError(f"Unknown field type: {type(field)}")

        # Sample using bilinear interpolation
        sampled = self._sample_field(field_data, self.positions)
        return sampled

    def validate(self) -> bool:
        """Check field and positions are compatible."""
        if self.field is None or self.positions is None:
            return False

        # Check positions are 2D (Nx2)
        if self.positions.ndim != 2 or self.positions.shape[1] != 2:
            raise ValueError(
                f"Agent positions must be Nx2, got shape {self.positions.shape}"
            )

        return True

    def _sample_field(self, field_data: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """
        Sample field at positions using bilinear interpolation.

        Args:
            field_data: 2D or 3D array (H, W) or (H, W, C)
            positions: Nx2 array of (y, x) coordinates

        Returns:
            N-length array of sampled values (or NxC for vector fields)
        """
        from scipy.ndimage import map_coordinates

        # Ensure field_data is a numpy array (not memoryview)
        field_data = np.asarray(field_data)

        # Normalize positions to grid coordinates
        h, w = field_data.shape[:2]
        coords = positions.copy()

        # Clamp to valid range
        coords[:, 0] = np.clip(coords[:, 0], 0, h - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, w - 1)

        # Sample using scipy map_coordinates
        if field_data.ndim == 2:
            # Scalar field
            sampled = map_coordinates(
                field_data,
                [coords[:, 0], coords[:, 1]],
                order=1,  # Bilinear
                mode='nearest'
            )
        else:
            # Vector field - sample each component
            sampled = np.zeros((len(positions), field_data.shape[2]), dtype=field_data.dtype)
            for c in range(field_data.shape[2]):
                component_data = np.asarray(field_data[:, :, c])
                sampled[:, c] = map_coordinates(
                    component_data,
                    [coords[:, 0], coords[:, 1]],
                    order=1,
                    mode='nearest'
                )

        return sampled

    def get_input_interface(self) -> Dict[str, Type]:
        return {
            'field': np.ndarray,
            'positions': np.ndarray,
        }

    def get_output_interface(self) -> Dict[str, Type]:
        return {
            'sampled_values': np.ndarray,
        }


class AgentToFieldInterface(DomainInterface):
    """
    Agent → Field: Deposit agent properties onto field grid.

    Use cases:
    - Particle positions → density field
    - Agent velocities → velocity field
    - Agent properties → heat sources
    """

    source_domain = "agent"
    target_domain = "field"

    def __init__(
        self,
        positions,
        values,
        field_shape: Tuple[int, int],
        method: str = "accumulate"
    ):
        super().__init__(source_data=(positions, values))
        self.positions = positions
        self.values = values
        self.field_shape = field_shape
        self.method = method  # "accumulate", "average", "max"

    def transform(self, source_data: Any) -> np.ndarray:
        """Deposit agent values onto field."""
        if source_data is not None:
            positions, values = source_data
        else:
            positions, values = self.positions, self.values

        field = np.zeros(self.field_shape, dtype=np.float32)

        # Convert positions to grid coordinates
        coords = positions.astype(int)

        # Clamp to valid range
        coords[:, 0] = np.clip(coords[:, 0], 0, self.field_shape[0] - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, self.field_shape[1] - 1)

        if self.method == "accumulate":
            # Sum all values at each grid cell
            for i, (y, x) in enumerate(coords):
                field[y, x] += values[i]

        elif self.method == "average":
            # Average values at each grid cell
            counts = np.zeros(self.field_shape, dtype=int)
            for i, (y, x) in enumerate(coords):
                field[y, x] += values[i]
                counts[y, x] += 1

            # Avoid division by zero
            mask = counts > 0
            field[mask] /= counts[mask]

        elif self.method == "max":
            # Take maximum value at each grid cell
            field.fill(-np.inf)
            for i, (y, x) in enumerate(coords):
                field[y, x] = max(field[y, x], values[i])
            field[field == -np.inf] = 0

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return field

    def validate(self) -> bool:
        """Check positions and values are compatible."""
        if self.positions is None or self.values is None:
            return False

        if len(self.positions) != len(self.values):
            raise ValueError(
                f"Positions ({len(self.positions)}) and values ({len(self.values)}) "
                f"must have same length"
            )

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {
            'positions': np.ndarray,
            'values': np.ndarray,
            'field_shape': Tuple[int, int],
        }

    def get_output_interface(self) -> Dict[str, Type]:
        return {
            'field': np.ndarray,
        }


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


class AudioToVisualInterface(DomainInterface):
    """
    Audio → Visual: Audio-reactive visual generation.

    Use cases:
    - FFT spectrum → color palette
    - Amplitude → particle emission
    - Beat detection → visual effects
    - Frequency analysis → color shifts
    """

    source_domain = "audio"
    target_domain = "visual"

    def __init__(
        self,
        audio_signal: np.ndarray,
        sample_rate: int = 44100,
        fft_size: int = 2048,
        mode: str = "spectrum"
    ):
        """
        Args:
            audio_signal: Audio samples (mono or stereo)
            sample_rate: Audio sample rate
            fft_size: FFT window size for spectral analysis
            mode: Analysis mode ("spectrum", "waveform", "energy", "beat")
        """
        super().__init__(source_data=audio_signal)
        self.audio_signal = audio_signal
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.mode = mode

    def transform(self, source_data: Any) -> Dict[str, np.ndarray]:
        """
        Convert audio to visual parameters.

        Returns:
            Dict with keys: 'colors', 'intensities', 'frequencies', 'energy'
        """
        audio = source_data if source_data is not None else self.audio_signal

        # Ensure mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        result = {}

        if self.mode == "spectrum":
            # FFT analysis
            fft = np.fft.rfft(audio[:self.fft_size])
            spectrum = np.abs(fft)

            # Normalize spectrum
            spectrum = spectrum / (np.max(spectrum) + 1e-10)

            result['spectrum'] = spectrum
            result['frequencies'] = np.fft.rfftfreq(self.fft_size, 1.0 / self.sample_rate)

            # Spectral centroid (brightness)
            spectral_centroid = np.sum(result['frequencies'] * spectrum) / (np.sum(spectrum) + 1e-10)
            result['brightness'] = spectral_centroid / (self.sample_rate / 2)  # Normalize

        elif self.mode == "waveform":
            # Raw waveform for oscilloscope-style visuals
            result['waveform'] = audio[:self.fft_size]
            result['amplitude'] = np.abs(audio[:self.fft_size])

        elif self.mode == "energy":
            # RMS energy
            energy = np.sqrt(np.mean(audio[:self.fft_size] ** 2))
            result['energy'] = energy
            result['intensity'] = np.clip(energy * 10.0, 0.0, 1.0)

        elif self.mode == "beat":
            # Simple beat detection (onset strength)
            hop_length = 512
            n_frames = len(audio) // hop_length

            onset_strength = []
            for i in range(n_frames):
                start = i * hop_length
                end = start + hop_length
                chunk = audio[start:end]
                energy = np.sqrt(np.mean(chunk ** 2))
                onset_strength.append(energy)

            onset_strength = np.array(onset_strength)

            # Detect peaks
            threshold = np.mean(onset_strength) + np.std(onset_strength)
            beats = onset_strength > threshold

            result['onset_strength'] = onset_strength
            result['beats'] = beats

        return result

    def validate(self) -> bool:
        """Check audio signal is valid."""
        if self.audio_signal is None:
            return False

        if not isinstance(self.audio_signal, np.ndarray):
            raise TypeError("Audio signal must be numpy array")

        if len(self.audio_signal) < self.fft_size:
            raise ValueError(f"Audio signal too short (need at least {self.fft_size} samples)")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {
            'audio_signal': np.ndarray,
            'sample_rate': int,
        }

    def get_output_interface(self) -> Dict[str, Type]:
        return {
            'visual_params': Dict[str, np.ndarray],
        }


class FieldToAudioInterface(DomainInterface):
    """
    Field → Audio: Field-driven audio synthesis.

    Use cases:
    - Temperature field → synthesis parameters
    - Vorticity → frequency modulation
    - Density patterns → rhythm generation
    - Field evolution → audio sequences
    """

    source_domain = "field"
    target_domain = "audio"

    def __init__(
        self,
        field: np.ndarray,
        mapping: Dict[str, str],
        sample_rate: int = 44100,
        duration: float = 1.0
    ):
        """
        Args:
            field: 2D field array
            mapping: Dict mapping field properties to audio parameters
                     e.g., {"mean": "frequency", "std": "amplitude"}
            sample_rate: Audio sample rate
            duration: Duration of generated audio (seconds)
        """
        super().__init__(source_data=field)
        self.field = field
        self.mapping = mapping
        self.sample_rate = sample_rate
        self.duration = duration

    def transform(self, source_data: Any) -> Dict[str, Any]:
        """
        Convert field to audio synthesis parameters.

        Returns:
            Dict with synthesis parameters
        """
        field = source_data if source_data is not None else self.field

        # Extract field statistics
        stats = {
            'mean': np.mean(field),
            'std': np.std(field),
            'min': np.min(field),
            'max': np.max(field),
            'range': np.ptp(field),
        }

        # Compute spatial statistics
        if field.ndim >= 2:
            # Gradient magnitude (activity/turbulence)
            gy, gx = np.gradient(field)
            gradient_mag = np.sqrt(gx**2 + gy**2)
            stats['gradient_mean'] = np.mean(gradient_mag)
            stats['gradient_max'] = np.max(gradient_mag)

        audio_params = {}

        # Map field properties to audio parameters
        for field_prop, audio_param in self.mapping.items():
            value = stats.get(field_prop, 0.0)

            if audio_param == "frequency":
                # Map to musical frequency range (100-1000 Hz)
                normalized = (value - stats['min']) / (stats['range'] + 1e-10)
                audio_params['frequency'] = 100.0 + normalized * 900.0

            elif audio_param == "amplitude":
                # Map to amplitude (0-1)
                normalized = (value - stats['min']) / (stats['range'] + 1e-10)
                audio_params['amplitude'] = np.clip(normalized, 0.0, 1.0)

            elif audio_param == "modulation":
                # Modulation depth
                audio_params['modulation_depth'] = np.clip(value / 10.0, 0.0, 1.0)

            elif audio_param == "filter_cutoff":
                # Filter cutoff frequency
                normalized = (value - stats['min']) / (stats['range'] + 1e-10)
                audio_params['filter_cutoff'] = 200.0 + normalized * 3800.0

        # Add timing info
        audio_params['sample_rate'] = self.sample_rate
        audio_params['duration'] = self.duration
        audio_params['n_samples'] = int(self.sample_rate * self.duration)

        return audio_params

    def validate(self) -> bool:
        """Check field and mapping are valid."""
        if self.field is None or self.mapping is None:
            return False

        valid_field_props = ['mean', 'std', 'min', 'max', 'range', 'gradient_mean', 'gradient_max']
        valid_audio_params = ['frequency', 'amplitude', 'modulation', 'filter_cutoff']

        for field_prop, audio_param in self.mapping.items():
            if field_prop not in valid_field_props:
                raise ValueError(f"Unknown field property: {field_prop}")
            if audio_param not in valid_audio_params:
                raise ValueError(f"Unknown audio parameter: {audio_param}")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {
            'field': np.ndarray,
            'mapping': Dict[str, str],
        }

    def get_output_interface(self) -> Dict[str, Type]:
        return {
            'audio_params': Dict[str, Any],
        }


class TerrainToFieldInterface(DomainInterface):
    """
    Terrain → Field: Convert terrain heightmap to scalar field.

    Use cases:
    - Heightmap → diffusion initial conditions
    - Elevation → potential field
    - Terrain features → field patterns
    """

    source_domain = "terrain"
    target_domain = "field"

    def __init__(self, heightmap: np.ndarray, normalize: bool = True):
        """
        Args:
            heightmap: 2D terrain heightmap
            normalize: If True, normalize to [0, 1] range
        """
        super().__init__(source_data=heightmap)
        self.heightmap = heightmap
        self.normalize = normalize

    def transform(self, source_data: Any) -> np.ndarray:
        """Convert heightmap to field."""
        heightmap = source_data if source_data is not None else self.heightmap

        # Extract height data if wrapped in object
        if hasattr(heightmap, 'data'):
            field_data = heightmap.data.copy()
        else:
            field_data = heightmap.copy()

        if self.normalize:
            # Normalize to [0, 1]
            field_min = field_data.min()
            field_max = field_data.max()
            if field_max > field_min:
                field_data = (field_data - field_min) / (field_max - field_min)

        return field_data

    def validate(self) -> bool:
        """Check heightmap is valid."""
        if self.heightmap is None:
            return False

        # Extract array
        if hasattr(self.heightmap, 'data'):
            arr = self.heightmap.data
        else:
            arr = self.heightmap

        if not isinstance(arr, np.ndarray):
            raise TypeError("Heightmap must be numpy array")

        if arr.ndim != 2:
            raise ValueError(f"Heightmap must be 2D, got shape {arr.shape}")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {'heightmap': np.ndarray}

    def get_output_interface(self) -> Dict[str, Type]:
        return {'field': np.ndarray}


class FieldToTerrainInterface(DomainInterface):
    """
    Field → Terrain: Convert scalar field to terrain heightmap.

    Use cases:
    - Procedural field → terrain generation
    - Simulation result → landscape
    """

    source_domain = "field"
    target_domain = "terrain"

    def __init__(self, field: np.ndarray, height_scale: float = 100.0):
        """
        Args:
            field: 2D scalar field
            height_scale: Scaling factor for height values
        """
        super().__init__(source_data=field)
        self.field = field
        self.height_scale = height_scale

    def transform(self, source_data: Any) -> Dict[str, np.ndarray]:
        """Convert field to heightmap."""
        field = source_data if source_data is not None else self.field

        # Normalize field to [0, 1]
        field_min = field.min()
        field_max = field.max()
        normalized = (field - field_min) / (field_max - field_min + 1e-10)

        # Scale to height range
        heightmap = normalized * self.height_scale

        return {
            'heightmap': heightmap,
            'min_height': 0.0,
            'max_height': self.height_scale,
        }

    def validate(self) -> bool:
        """Check field is valid."""
        if self.field is None:
            return False

        if not isinstance(self.field, np.ndarray):
            raise TypeError("Field must be numpy array")

        if self.field.ndim != 2:
            raise ValueError(f"Field must be 2D, got shape {self.field.shape}")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {'field': np.ndarray}

    def get_output_interface(self) -> Dict[str, Type]:
        return {'terrain_data': Dict[str, np.ndarray]}


class VisionToFieldInterface(DomainInterface):
    """
    Vision → Field: Convert computer vision features to fields.

    Use cases:
    - Edge map → scalar field
    - Optical flow → vector field
    - Feature map → field initialization
    """

    source_domain = "vision"
    target_domain = "field"

    def __init__(self, image: np.ndarray, mode: str = "edges"):
        """
        Args:
            image: Input image (grayscale or RGB)
            mode: Conversion mode ("edges", "gradient", "intensity")
        """
        super().__init__(source_data=image)
        self.image = image
        self.mode = mode

    def transform(self, source_data: Any) -> np.ndarray:
        """Convert vision data to field."""
        image = source_data if source_data is not None else self.image

        # Convert to grayscale if RGB
        if image.ndim == 3:
            image = np.mean(image, axis=2)

        if self.mode == "edges":
            # Edge detection produces scalar field
            from scipy.ndimage import sobel
            sx = sobel(image, axis=1)
            sy = sobel(image, axis=0)
            edge_mag = np.sqrt(sx**2 + sy**2)
            return edge_mag

        elif self.mode == "gradient":
            # Gradient field (vector)
            gy, gx = np.gradient(image)
            # Return as vector field (H, W, 2)
            return np.stack([gy, gx], axis=2)

        elif self.mode == "intensity":
            # Direct intensity mapping
            return image.astype(np.float32)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def validate(self) -> bool:
        """Check image is valid."""
        if self.image is None:
            return False

        if not isinstance(self.image, np.ndarray):
            raise TypeError("Image must be numpy array")

        if self.image.ndim not in [2, 3]:
            raise ValueError(f"Image must be 2D or 3D, got shape {self.image.shape}")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {'image': np.ndarray}

    def get_output_interface(self) -> Dict[str, Type]:
        return {'field': np.ndarray}


class GraphToVisualInterface(DomainInterface):
    """
    Graph → Visual: Network graph visualization.

    Use cases:
    - Network structure → visual layout
    - Graph metrics → node colors/sizes
    - Connectivity → edge rendering
    """

    source_domain = "graph"
    target_domain = "visual"

    def __init__(
        self,
        graph_data: Dict[str, Any],
        width: int = 512,
        height: int = 512,
        layout: str = "spring"
    ):
        """
        Args:
            graph_data: Dict with 'nodes' and 'edges' keys
            width: Output image width
            height: Output image height
            layout: Layout algorithm ("spring", "circular", "random")
        """
        super().__init__(source_data=graph_data)
        self.graph_data = graph_data
        self.width = width
        self.height = height
        self.layout = layout

    def transform(self, source_data: Any) -> Dict[str, Any]:
        """
        Convert graph to visual representation.

        Returns:
            Dict with 'node_positions', 'edge_list', 'image' keys
        """
        graph = source_data if source_data is not None else self.graph_data

        n_nodes = len(graph.get('nodes', []))

        # Simple layout algorithms
        if self.layout == "circular":
            # Circular layout
            angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
            radius = min(self.width, self.height) * 0.4
            cx, cy = self.width / 2, self.height / 2

            positions = np.zeros((n_nodes, 2))
            positions[:, 0] = cx + radius * np.cos(angles)
            positions[:, 1] = cy + radius * np.sin(angles)

        elif self.layout == "random":
            # Random layout
            positions = np.random.rand(n_nodes, 2)
            positions[:, 0] *= self.width
            positions[:, 1] *= self.height

        elif self.layout == "spring":
            # Simple spring layout (simplified force-directed)
            positions = np.random.rand(n_nodes, 2)
            positions[:, 0] *= self.width
            positions[:, 1] *= self.height

            # Simple relaxation (could be improved with proper spring algorithm)
            for _ in range(50):
                # Repulsion between all nodes
                for i in range(n_nodes):
                    for j in range(i + 1, n_nodes):
                        delta = positions[i] - positions[j]
                        dist = np.linalg.norm(delta) + 1e-10
                        force = delta / dist * (100.0 / dist)
                        positions[i] += force * 0.1
                        positions[j] -= force * 0.1

                # Clamp to bounds
                positions[:, 0] = np.clip(positions[:, 0], 0, self.width)
                positions[:, 1] = np.clip(positions[:, 1], 0, self.height)

        return {
            'node_positions': positions,
            'edge_list': graph.get('edges', []),
            'n_nodes': n_nodes,
            'width': self.width,
            'height': self.height,
        }

    def validate(self) -> bool:
        """Check graph data is valid."""
        if self.graph_data is None:
            return False

        if 'nodes' not in self.graph_data:
            raise ValueError("Graph data must have 'nodes' key")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {'graph_data': Dict}

    def get_output_interface(self) -> Dict[str, Type]:
        return {'visual_data': Dict[str, Any]}


class CellularToFieldInterface(DomainInterface):
    """
    Cellular → Field: Convert cellular automata state to field.

    Use cases:
    - CA state → initial conditions for PDEs
    - Game of Life → density field
    - Pattern state → field patterns
    """

    source_domain = "cellular"
    target_domain = "field"

    def __init__(self, ca_state: np.ndarray, normalize: bool = True):
        """
        Args:
            ca_state: Cellular automata state array
            normalize: If True, normalize to [0, 1]
        """
        super().__init__(source_data=ca_state)
        self.ca_state = ca_state
        self.normalize = normalize

    def transform(self, source_data: Any) -> np.ndarray:
        """Convert CA state to field."""
        ca_state = source_data if source_data is not None else self.ca_state

        field = ca_state.astype(np.float32)

        if self.normalize:
            field_min = field.min()
            field_max = field.max()
            if field_max > field_min:
                field = (field - field_min) / (field_max - field_min)

        return field

    def validate(self) -> bool:
        """Check CA state is valid."""
        if self.ca_state is None:
            return False

        if not isinstance(self.ca_state, np.ndarray):
            raise TypeError("CA state must be numpy array")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {'ca_state': np.ndarray}

    def get_output_interface(self) -> Dict[str, Type]:
        return {'field': np.ndarray}


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


# ==============================================================================
# TIME-FREQUENCY DOMAIN TRANSFORMS
# ==============================================================================


class TimeToCepstralInterface(DomainInterface):
    """
    Time → Cepstral domain via Discrete Cosine Transform (DCT).

    DCT is widely used for:
    - Audio compression (e.g., MP3, AAC)
    - MFCC computation (Mel-frequency cepstral coefficients)
    - Cepstral analysis for pitch detection
    - Feature extraction for speech recognition

    Supports DCT types 1-4 with orthogonal normalization.
    """

    source_domain = "time"
    target_domain = "cepstral"

    def __init__(self, signal: np.ndarray, dct_type: int = 2,
                 norm: str = "ortho", metadata: Optional[Dict] = None):
        """
        Initialize DCT transform.

        Args:
            signal: Time-domain signal (1D array)
            dct_type: DCT type (1, 2, 3, or 4). Type-2 is most common.
            norm: Normalization mode ("ortho" or None)
            metadata: Optional metadata dict
        """
        super().__init__(signal, metadata)
        self.dct_type = dct_type
        self.norm = norm

        if dct_type not in [1, 2, 3, 4]:
            raise ValueError(f"DCT type must be 1-4, got {dct_type}")
        if norm not in ["ortho", None]:
            raise ValueError(f"Norm must be 'ortho' or None, got {norm}")

    def transform(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply DCT to transform time-domain signal to cepstral domain.

        Args:
            signal: Time-domain signal (1D array)

        Returns:
            Cepstral coefficients (1D array, same length as input)
        """
        from scipy.fft import dct

        if signal.ndim != 1:
            raise ValueError(f"Signal must be 1D, got shape {signal.shape}")

        # Apply DCT
        cepstral = dct(signal, type=self.dct_type, norm=self.norm)

        return cepstral.astype(np.float32)

    def validate(self) -> bool:
        """Validate signal is 1D array."""
        if self.source_data is None:
            return True

        if not isinstance(self.source_data, np.ndarray):
            raise TypeError("Signal must be numpy array")

        if self.source_data.ndim != 1:
            raise ValueError("Signal must be 1D array")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {'signal': np.ndarray}

    def get_output_interface(self) -> Dict[str, Type]:
        return {'cepstral_coefficients': np.ndarray}


class CepstralToTimeInterface(DomainInterface):
    """
    Cepstral → Time domain via Inverse Discrete Cosine Transform (IDCT).

    Reconstructs time-domain signal from DCT coefficients.
    """

    source_domain = "cepstral"
    target_domain = "time"

    def __init__(self, cepstral: np.ndarray, dct_type: int = 2,
                 norm: str = "ortho", metadata: Optional[Dict] = None):
        """
        Initialize IDCT transform.

        Args:
            cepstral: Cepstral coefficients (1D array)
            dct_type: DCT type used in forward transform (1, 2, 3, or 4)
            norm: Normalization mode ("ortho" or None)
            metadata: Optional metadata dict
        """
        super().__init__(cepstral, metadata)
        self.dct_type = dct_type
        self.norm = norm

        if dct_type not in [1, 2, 3, 4]:
            raise ValueError(f"DCT type must be 1-4, got {dct_type}")
        if norm not in ["ortho", None]:
            raise ValueError(f"Norm must be 'ortho' or None, got {norm}")

    def transform(self, cepstral: np.ndarray) -> np.ndarray:
        """
        Apply IDCT to transform cepstral coefficients back to time domain.

        Args:
            cepstral: Cepstral coefficients (1D array)

        Returns:
            Time-domain signal (1D array, same length as input)
        """
        from scipy.fft import idct

        if cepstral.ndim != 1:
            raise ValueError(f"Cepstral coefficients must be 1D, got shape {cepstral.shape}")

        # Apply IDCT
        signal = idct(cepstral, type=self.dct_type, norm=self.norm)

        return signal.astype(np.float32)

    def validate(self) -> bool:
        """Validate cepstral coefficients are 1D array."""
        if self.source_data is None:
            return True

        if not isinstance(self.source_data, np.ndarray):
            raise TypeError("Cepstral coefficients must be numpy array")

        if self.source_data.ndim != 1:
            raise ValueError("Cepstral coefficients must be 1D array")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {'cepstral_coefficients': np.ndarray}

    def get_output_interface(self) -> Dict[str, Type]:
        return {'signal': np.ndarray}


class TimeToWaveletInterface(DomainInterface):
    """
    Time → Wavelet domain via Continuous Wavelet Transform (CWT).

    CWT provides time-scale representation of signals, useful for:
    - Non-stationary signal analysis
    - Feature detection at multiple scales
    - Edge detection in images
    - Transient analysis in audio

    Uses scipy.signal.cwt with various mother wavelets (Morlet, Ricker, etc.)
    """

    source_domain = "time"
    target_domain = "wavelet"

    def __init__(self, signal: np.ndarray, scales: np.ndarray,
                 wavelet: str = "morlet", metadata: Optional[Dict] = None):
        """
        Initialize CWT transform.

        Args:
            signal: Time-domain signal (1D array)
            scales: Array of scales to use (e.g., np.arange(1, 128))
            wavelet: Wavelet type ("morlet", "ricker", "mexh", "morl")
            metadata: Optional metadata dict
        """
        super().__init__(signal, metadata)
        self.scales = scales
        self.wavelet = wavelet

        valid_wavelets = ["morlet", "ricker", "mexh", "morl"]
        if wavelet not in valid_wavelets:
            raise ValueError(f"Wavelet must be one of {valid_wavelets}, got {wavelet}")

    def transform(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply CWT to transform signal to wavelet domain.

        Args:
            signal: Time-domain signal (1D array)

        Returns:
            Wavelet coefficients (2D array: scales × time)
        """
        if signal.ndim != 1:
            raise ValueError(f"Signal must be 1D, got shape {signal.shape}")

        # Apply CWT using convolution with scaled wavelets
        # Generate Ricker wavelet (Mexican hat) for each scale
        coefficients = []

        for scale in self.scales:
            # Generate Ricker wavelet
            wavelet = self._ricker_wavelet(scale)

            # Convolve signal with wavelet
            coeff = np.convolve(signal, wavelet, mode='same')
            coefficients.append(coeff)

        coefficients = np.array(coefficients, dtype=np.float32)
        return coefficients

    def _ricker_wavelet(self, scale: float) -> np.ndarray:
        """Generate Ricker (Mexican hat) wavelet at given scale."""
        # Ricker wavelet: ψ(t) = (1 - t²) * exp(-t²/2)
        # Scale determines the width
        points = min(int(10 * scale), 100)  # Wavelet support, capped at 100
        if points < 3:
            points = 3  # Minimum wavelet size
        if points % 2 == 0:
            points += 1  # Make odd for symmetry
        t = np.linspace(-5, 5, points)
        wavelet = (1.0 - t**2) * np.exp(-t**2 / 2.0)
        # Normalize
        wavelet = wavelet / (np.sqrt(np.sum(wavelet**2)) + 1e-10)
        return wavelet.astype(np.float32)

    def validate(self) -> bool:
        """Validate signal and scales."""
        if self.source_data is None:
            return True

        if not isinstance(self.source_data, np.ndarray):
            raise TypeError("Signal must be numpy array")

        if self.source_data.ndim != 1:
            raise ValueError("Signal must be 1D array")

        if not isinstance(self.scales, np.ndarray):
            raise TypeError("Scales must be numpy array")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {'signal': np.ndarray, 'scales': np.ndarray}

    def get_output_interface(self) -> Dict[str, Type]:
        return {'wavelet_coefficients': np.ndarray}


# ==============================================================================
# SPATIAL DOMAIN TRANSFORMS
# ==============================================================================


class SpatialAffineInterface(DomainInterface):
    """
    Spatial → Spatial via Affine transformations (translate, rotate, scale, shear).

    Affine transformations preserve:
    - Points, straight lines, and planes
    - Parallel lines remain parallel
    - Ratios of distances along lines

    Useful for:
    - Image/geometry registration
    - Data augmentation
    - Coordinate system alignment
    - Field transformations
    """

    source_domain = "spatial"
    target_domain = "spatial"

    def __init__(self, data: np.ndarray,
                 translate: Optional[Tuple[float, float]] = None,
                 rotate: Optional[float] = None,
                 scale: Optional[Tuple[float, float]] = None,
                 shear: Optional[float] = None,
                 order: int = 1,
                 metadata: Optional[Dict] = None):
        """
        Initialize affine transform.

        Args:
            data: 2D spatial data (image or field)
            translate: Translation (dx, dy) in pixels
            rotate: Rotation angle in degrees (counter-clockwise)
            scale: Scale factors (sx, sy)
            shear: Shear angle in degrees
            order: Interpolation order (0=nearest, 1=linear, 3=cubic)
            metadata: Optional metadata dict
        """
        super().__init__(data, metadata)
        self.translate = translate or (0, 0)
        self.rotate = rotate or 0
        self.scale = scale or (1, 1)
        self.shear = shear or 0
        self.order = order

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Apply affine transformation to spatial data.

        Args:
            data: 2D spatial data

        Returns:
            Transformed spatial data (same shape as input)
        """
        from scipy.ndimage import affine_transform

        if data.ndim not in [2, 3]:
            raise ValueError(f"Data must be 2D or 3D (2D + channels), got shape {data.shape}")

        # Get image center for rotation/scaling
        height, width = data.shape[:2]
        center_y, center_x = height / 2.0, width / 2.0

        # Build transformation matrix (applies transformations in order: scale, rotate, shear, translate)
        # We use homogeneous coordinates for easier composition

        # Start with identity
        matrix = np.eye(3)

        # Translate to origin (for rotation/scaling around center)
        T1 = np.array([
            [1, 0, -center_x],
            [0, 1, -center_y],
            [0, 0, 1]
        ])

        # Apply scale
        sx, sy = self.scale
        S = np.array([
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, 1]
        ])

        # Apply rotation (counter-clockwise)
        angle_rad = np.radians(self.rotate)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        R = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])

        # Apply shear
        shear_rad = np.radians(self.shear)
        SH = np.array([
            [1, np.tan(shear_rad), 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        # Translate back from origin
        T2 = np.array([
            [1, 0, center_x],
            [0, 1, center_y],
            [0, 0, 1]
        ])

        # Apply user translation
        dx, dy = self.translate
        T3 = np.array([
            [1, 0, dx],
            [0, 1, dy],
            [0, 0, 1]
        ])

        # Compose transformations: T3 * T2 * SH * R * S * T1
        matrix = T3 @ T2 @ SH @ R @ S @ T1

        # Extract 2x2 matrix and offset for scipy
        # Note: scipy uses (y, x) indexing, so we need to swap
        M = matrix[:2, :2]
        offset = matrix[:2, 2]

        # scipy.ndimage.affine_transform uses inverse mapping (output -> input)
        # So we need to invert the matrix
        M_inv = np.linalg.inv(M)
        # Compute inverse offset: -M_inv @ offset
        offset_inv = -M_inv @ offset

        # Apply transformation
        if data.ndim == 2:
            transformed = affine_transform(data, M_inv, offset=offset_inv,
                                          order=self.order, mode='constant', cval=0)
        else:
            # Handle multi-channel data
            channels = []
            for i in range(data.shape[2]):
                channel = affine_transform(data[:, :, i], M_inv, offset=offset_inv,
                                          order=self.order, mode='constant', cval=0)
                channels.append(channel)
            transformed = np.stack(channels, axis=2)

        return transformed.astype(data.dtype)

    def validate(self) -> bool:
        """Validate data is 2D or 3D array."""
        if self.source_data is None:
            return True

        if not isinstance(self.source_data, np.ndarray):
            raise TypeError("Data must be numpy array")

        if self.source_data.ndim not in [2, 3]:
            raise ValueError("Data must be 2D or 3D (2D + channels)")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {'spatial_data': np.ndarray}

    def get_output_interface(self) -> Dict[str, Type]:
        return {'transformed_spatial_data': np.ndarray}


class CartesianToPolarInterface(DomainInterface):
    """
    Cartesian → Polar coordinate conversion.

    Converts (x, y) Cartesian coordinates to (r, theta) polar coordinates.

    Useful for:
    - Radial pattern analysis
    - Rotational symmetry detection
    - Circular/angular data visualization
    - Fourier-Bessel transforms
    """

    source_domain = "cartesian"
    target_domain = "polar"

    def __init__(self, data: np.ndarray, center: Optional[Tuple[float, float]] = None,
                 metadata: Optional[Dict] = None):
        """
        Initialize Cartesian to Polar transform.

        Args:
            data: 2D field in Cartesian coordinates
            center: Origin for polar conversion (cx, cy). If None, uses image center.
            metadata: Optional metadata dict
        """
        super().__init__(data, metadata)
        self.center = center

    def transform(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert Cartesian field to polar coordinates.

        Args:
            data: 2D field in Cartesian coordinates

        Returns:
            Tuple of (radius_array, angle_array) in polar coordinates
        """
        if data.ndim != 2:
            raise ValueError(f"Data must be 2D, got shape {data.shape}")

        height, width = data.shape

        # Determine center
        if self.center is None:
            cx, cy = width / 2, height / 2
        else:
            cx, cy = self.center

        # Create coordinate grids
        y, x = np.indices(data.shape, dtype=np.float32)

        # Convert to polar
        dx = x - cx
        dy = y - cy
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)  # Range: [-pi, pi]

        return r, theta

    def validate(self) -> bool:
        """Validate data is 2D array."""
        if self.source_data is None:
            return True

        if not isinstance(self.source_data, np.ndarray):
            raise TypeError("Data must be numpy array")

        if self.source_data.ndim != 2:
            raise ValueError("Data must be 2D array")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {'cartesian_data': np.ndarray}

    def get_output_interface(self) -> Dict[str, Type]:
        return {'radius': np.ndarray, 'angle': np.ndarray}


class PolarToCartesianInterface(DomainInterface):
    """
    Polar → Cartesian coordinate conversion.

    Converts (r, theta) polar coordinates back to (x, y) Cartesian coordinates.

    Useful for:
    - Reconstructing fields after radial processing
    - Visualization of polar data
    - Inverse transforms after polar filtering
    """

    source_domain = "polar"
    target_domain = "cartesian"

    def __init__(self, radius: np.ndarray, angle: np.ndarray,
                 output_shape: Tuple[int, int],
                 center: Optional[Tuple[float, float]] = None,
                 metadata: Optional[Dict] = None):
        """
        Initialize Polar to Cartesian transform.

        Args:
            radius: Radius values (2D array)
            angle: Angle values in radians (2D array)
            output_shape: Desired output shape (height, width)
            center: Origin for conversion (cx, cy). If None, uses image center.
            metadata: Optional metadata dict
        """
        super().__init__(radius, metadata)
        self.angle = angle
        self.output_shape = output_shape
        self.center = center

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Convert polar field back to Cartesian coordinates.

        Args:
            data: Values in polar space (typically same shape as radius/angle)

        Returns:
            2D field in Cartesian coordinates
        """
        from scipy.interpolate import griddata

        # Use source_data (radius) for coordinate conversion
        radius = self.source_data if self.source_data is not None else data

        if data.shape != radius.shape:
            raise ValueError(f"Data shape {data.shape} must match radius shape {radius.shape}")

        height, width = self.output_shape

        # Determine center
        if self.center is None:
            cx, cy = width / 2, height / 2
        else:
            cx, cy = self.center

        # Convert polar to Cartesian coordinates
        x_polar = radius * np.cos(self.angle) + cx
        y_polar = radius * np.sin(self.angle) + cy

        # Flatten for interpolation
        points = np.column_stack([x_polar.ravel(), y_polar.ravel()])
        values = data.ravel()

        # Create output grid
        y_out, x_out = np.indices((height, width), dtype=np.float32)
        grid_points = np.column_stack([x_out.ravel(), y_out.ravel()])

        # Interpolate
        cartesian = griddata(points, values, grid_points, method='linear', fill_value=0)
        cartesian = cartesian.reshape(height, width)

        return cartesian.astype(np.float32)

    def validate(self) -> bool:
        """Validate radius and angle arrays."""
        if self.source_data is None:
            return True

        if not isinstance(self.source_data, np.ndarray):
            raise TypeError("Radius must be numpy array")

        if not isinstance(self.angle, np.ndarray):
            raise TypeError("Angle must be numpy array")

        if self.source_data.shape != self.angle.shape:
            raise ValueError("Radius and angle must have same shape")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {'radius': np.ndarray, 'angle': np.ndarray, 'values': np.ndarray}

    def get_output_interface(self) -> Dict[str, Type]:
        return {'cartesian_data': np.ndarray}
