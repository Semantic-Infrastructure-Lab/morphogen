"""Morphogen Audio Dialect (v0.7.0 Phase 5)

This module defines the Morphogen Audio dialect for MLIR, providing high-level
operations for audio synthesis and processing that lower to SCF loops and memref operations.

Status: Phase 5 Implementation (Months 10-12)

Operations:
- morphogen.audio.buffer.create: Allocate audio buffer with sample rate and duration
- morphogen.audio.oscillator: Generate waveforms (sine, square, saw, triangle)
- morphogen.audio.envelope: Apply ADSR envelope to audio signal
- morphogen.audio.filter: Apply IIR/FIR filters (lowpass, highpass, bandpass)
- morphogen.audio.mix: Mix multiple audio signals with scaling

Type System:
- !morphogen.audio<sample_rate, channels>: Audio buffer type (opaque for Phase 5)

Integration:
- Audio buffers can be converted to/from field data for sonification
- Audio operations integrate with stdlib audio DSP (FFT, spectral analysis)
- Temporal operations can modulate audio parameters over time
"""

from __future__ import annotations
from typing import Optional, List, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..context import MorphogenMLIRContext

# Import MLIR if available
try:
    from mlir import ir
    from mlir.dialects import builtin, arith, memref, scf
    MLIR_AVAILABLE = True
except ImportError:
    MLIR_AVAILABLE = False
    if TYPE_CHECKING:
        from mlir import ir
    else:
        ir = None


class AudioType:
    """Wrapper for !morphogen.audio<sample_rate, channels> type.

    In Phase 5, we use OpaqueType to represent custom audio types.
    Future phases may use proper IRDL dialect definition.

    Example:
        >>> ctx = MorphogenMLIRContext()
        >>> audio_type = AudioType.get(44100, 1, ctx.ctx)
        >>> print(audio_type)  # !morphogen.audio<44100, 1>
    """

    @staticmethod
    def get(sample_rate: int, channels: int, context: Any) -> Any:
        """Get audio type for given sample rate and channel count.

        Args:
            sample_rate: Sample rate in Hz (e.g., 44100, 48000)
            channels: Number of audio channels (1=mono, 2=stereo)
            context: MLIR context

        Returns:
            Opaque audio type !morphogen.audio<sample_rate, channels>
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        # Use OpaqueType for custom types in Phase 5
        # Format: !morphogen.audio<sample_rate, channels>
        return ir.OpaqueType.get("morphogen", f"audio<{sample_rate},{channels}>", context=context)


class AudioBufferCreateOp:
    """Operation: morphogen.audio.buffer.create

    Creates a new audio buffer with specified sample rate, channels, and duration.

    Syntax:
        %buffer = morphogen.audio.buffer.create %sample_rate, %channels, %duration
                  : !morphogen.audio<44100, 1>

    Attributes:
        - sample_rate: Audio sample rate in Hz (index type)
        - channels: Number of channels (index type, 1=mono, 2=stereo)
        - duration: Duration in seconds (f32 type)

    Results:
        - Audio buffer of type !morphogen.audio<sample_rate, channels>

    Lowering:
        Lowers to memref.alloc with size = sample_rate * duration * channels

        Example:
            sample_rate=44100, duration=1.0, channels=1
            → memref.alloc() : memref<44100xf32>
    """

    @staticmethod
    def create(
        sample_rate: Any,  # ir.Value with index type
        channels: Any,  # ir.Value with index type
        duration: Any,  # ir.Value with f32 type
        loc: Any,  # ir.Location
        ip: Any  # ir.InsertionPoint
    ) -> Any:  # ir.OpResult
        """Create an audio buffer creation operation.

        Args:
            sample_rate: Sample rate value (e.g., 44100)
            channels: Number of channels (1 or 2)
            duration: Duration in seconds
            loc: Source location
            ip: Insertion point

        Returns:
            Audio buffer value representing the created buffer
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        with loc, ip:
            # Create the audio type (using placeholder values in opaque type)
            audio_type = AudioType.get(44100, 1, loc.context)

            # Use unrealized_conversion_cast as a placeholder
            result = builtin.UnrealizedConversionCastOp(
                [audio_type],
                [sample_rate, channels, duration]
            )

            # Mark as audio.buffer.create op
            result.operation.attributes["op_name"] = ir.StringAttr.get(
                "morphogen.audio.buffer.create", context=loc.context
            )

            return result.results[0]


class AudioOscillatorOp:
    """Operation: morphogen.audio.oscillator

    Generates audio waveforms (sine, square, sawtooth, triangle).

    Syntax:
        %osc = morphogen.audio.oscillator %buffer, %waveform, %freq, %phase
               : !morphogen.audio<44100, 1>

    Arguments:
        - buffer: Target audio buffer (will be filled)
        - waveform: Waveform type (0=sine, 1=square, 2=saw, 3=triangle)
        - freq: Frequency in Hz (f32 type)
        - phase: Initial phase in radians (f32 type, 0 to 2π)

    Results:
        - Audio buffer filled with oscillator output

    Lowering:
        Lowers to scf.for loop generating samples:

        For sine wave:
            for i in range(num_samples):
                t = i / sample_rate
                buffer[i] = sin(2π * freq * t + phase)
    """

    @staticmethod
    def create(
        buffer: Any,  # ir.Value with audio type
        waveform: Any,  # ir.Value with index type (0=sine, 1=square, 2=saw, 3=triangle)
        frequency: Any,  # ir.Value with f32 type
        phase: Any,  # ir.Value with f32 type
        loc: Any,  # ir.Location
        ip: Any  # ir.InsertionPoint
    ) -> Any:  # ir.OpResult
        """Create an oscillator operation.

        Args:
            buffer: Target audio buffer
            waveform: Waveform type (0=sine, 1=square, 2=saw, 3=triangle)
            frequency: Frequency in Hz
            phase: Initial phase in radians
            loc: Source location
            ip: Insertion point

        Returns:
            Audio buffer filled with oscillator output
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        with loc, ip:
            audio_type = buffer.type

            # Create placeholder op
            result = builtin.UnrealizedConversionCastOp(
                [audio_type],
                [buffer, waveform, frequency, phase]
            )

            # Mark as audio.oscillator op
            result.operation.attributes["op_name"] = ir.StringAttr.get(
                "morphogen.audio.oscillator", context=loc.context
            )

            return result.results[0]


class AudioEnvelopeOp:
    """Operation: morphogen.audio.envelope

    Applies ADSR (Attack, Decay, Sustain, Release) envelope to audio signal.

    Syntax:
        %env = morphogen.audio.envelope %buffer, %attack, %decay, %sustain, %release
               : !morphogen.audio<44100, 1>

    Arguments:
        - buffer: Input audio buffer
        - attack: Attack time in seconds (f32)
        - decay: Decay time in seconds (f32)
        - sustain: Sustain level (0.0 to 1.0, f32)
        - release: Release time in seconds (f32)

    Results:
        - Audio buffer with envelope applied

    Lowering:
        Lowers to scf.for loop with ADSR state machine:

        for i in range(num_samples):
            t = i / sample_rate
            if t < attack:
                env = t / attack
            elif t < attack + decay:
                env = 1.0 - (1.0 - sustain) * (t - attack) / decay
            elif t < duration - release:
                env = sustain
            else:
                env = sustain * (1.0 - (t - (duration - release)) / release)

            buffer[i] *= env
    """

    @staticmethod
    def create(
        buffer: Any,  # ir.Value with audio type
        attack: Any,  # ir.Value with f32 type
        decay: Any,  # ir.Value with f32 type
        sustain: Any,  # ir.Value with f32 type (0.0 to 1.0)
        release: Any,  # ir.Value with f32 type
        loc: Any,  # ir.Location
        ip: Any  # ir.InsertionPoint
    ) -> Any:  # ir.OpResult
        """Create an envelope operation.

        Args:
            buffer: Input audio buffer
            attack: Attack time in seconds
            decay: Decay time in seconds
            sustain: Sustain level (0.0 to 1.0)
            release: Release time in seconds
            loc: Source location
            ip: Insertion point

        Returns:
            Audio buffer with envelope applied
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        with loc, ip:
            audio_type = buffer.type

            # Create placeholder op
            result = builtin.UnrealizedConversionCastOp(
                [audio_type],
                [buffer, attack, decay, sustain, release]
            )

            # Mark as audio.envelope op
            result.operation.attributes["op_name"] = ir.StringAttr.get(
                "morphogen.audio.envelope", context=loc.context
            )

            return result.results[0]


class AudioFilterOp:
    """Operation: morphogen.audio.filter

    Applies IIR/FIR filters (lowpass, highpass, bandpass) to audio signal.

    Syntax:
        %filtered = morphogen.audio.filter %buffer, %filter_type, %cutoff, %resonance
                    : !morphogen.audio<44100, 1>

    Arguments:
        - buffer: Input audio buffer
        - filter_type: Filter type (0=lowpass, 1=highpass, 2=bandpass)
        - cutoff: Cutoff frequency in Hz (f32)
        - resonance: Resonance/Q factor (f32, typically 0.5 to 10.0)

    Results:
        - Filtered audio buffer

    Lowering:
        Lowers to scf.for loop with biquad IIR filter:

        Biquad coefficients computed from cutoff and resonance:
            a0, a1, a2, b1, b2 = compute_biquad_coeffs(filter_type, cutoff, Q)

        for i in range(num_samples):
            y[i] = a0*x[i] + a1*x[i-1] + a2*x[i-2] - b1*y[i-1] - b2*y[i-2]
    """

    @staticmethod
    def create(
        buffer: Any,  # ir.Value with audio type
        filter_type: Any,  # ir.Value with index type (0=lowpass, 1=highpass, 2=bandpass)
        cutoff: Any,  # ir.Value with f32 type
        resonance: Any,  # ir.Value with f32 type
        loc: Any,  # ir.Location
        ip: Any  # ir.InsertionPoint
    ) -> Any:  # ir.OpResult
        """Create a filter operation.

        Args:
            buffer: Input audio buffer
            filter_type: Filter type (0=lowpass, 1=highpass, 2=bandpass)
            cutoff: Cutoff frequency in Hz
            resonance: Resonance/Q factor
            loc: Source location
            ip: Insertion point

        Returns:
            Filtered audio buffer
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        with loc, ip:
            audio_type = buffer.type

            # Create placeholder op
            result = builtin.UnrealizedConversionCastOp(
                [audio_type],
                [buffer, filter_type, cutoff, resonance]
            )

            # Mark as audio.filter op
            result.operation.attributes["op_name"] = ir.StringAttr.get(
                "morphogen.audio.filter", context=loc.context
            )

            return result.results[0]


class AudioMixOp:
    """Operation: morphogen.audio.mix

    Mixes multiple audio signals with optional scaling/gain.

    Syntax:
        %mixed = morphogen.audio.mix %buffer1, %buffer2, %gain1, %gain2
                 : !morphogen.audio<44100, 1>

    Arguments:
        - buffers: List of input audio buffers (variable length)
        - gains: List of gain values for each buffer (f32, variable length)

    Results:
        - Mixed audio buffer

    Lowering:
        Lowers to scf.for loop summing scaled samples:

        for i in range(num_samples):
            output[i] = gain1 * buffer1[i] + gain2 * buffer2[i] + ...
    """

    @staticmethod
    def create(
        buffers: List[Any],  # List of ir.Value with audio type
        gains: List[Any],  # List of ir.Value with f32 type
        loc: Any,  # ir.Location
        ip: Any  # ir.InsertionPoint
    ) -> Any:  # ir.OpResult
        """Create a mix operation.

        Args:
            buffers: List of input audio buffers to mix
            gains: List of gain values (one per buffer)
            loc: Source location
            ip: Insertion point

        Returns:
            Mixed audio buffer

        Note:
            All input buffers must have the same sample rate and duration.
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        if len(buffers) == 0:
            raise ValueError("Must provide at least one buffer to mix")

        if len(buffers) != len(gains):
            raise ValueError("Number of buffers and gains must match")

        with loc, ip:
            audio_type = buffers[0].type

            # Flatten buffers and gains into a single operand list
            operands = []
            for buf, gain in zip(buffers, gains):
                operands.append(buf)
                operands.append(gain)

            # Create placeholder op
            result = builtin.UnrealizedConversionCastOp(
                [audio_type],
                operands
            )

            # Mark as audio.mix op and store buffer count
            result.operation.attributes["op_name"] = ir.StringAttr.get(
                "morphogen.audio.mix", context=loc.context
            )
            result.operation.attributes["num_buffers"] = ir.IntegerAttr.get(
                ir.IntegerType.get_signless(32), len(buffers)
            )

            return result.results[0]


class AudioDialect:
    """Morphogen Audio Dialect helper class.

    This class provides utility methods for working with audio operations
    and checking if operations belong to the audio dialect.
    """

    @staticmethod
    def buffer_create(
        sample_rate: Any,
        channels: Any,
        duration: Any,
        loc: Any,
        ip: Any
    ) -> Any:
        """Create audio buffer creation operation."""
        return AudioBufferCreateOp.create(sample_rate, channels, duration, loc, ip)

    @staticmethod
    def oscillator(
        buffer: Any,
        waveform: Any,
        frequency: Any,
        phase: Any,
        loc: Any,
        ip: Any
    ) -> Any:
        """Create oscillator operation."""
        return AudioOscillatorOp.create(buffer, waveform, frequency, phase, loc, ip)

    @staticmethod
    def envelope(
        buffer: Any,
        attack: Any,
        decay: Any,
        sustain: Any,
        release: Any,
        loc: Any,
        ip: Any
    ) -> Any:
        """Create envelope operation."""
        return AudioEnvelopeOp.create(buffer, attack, decay, sustain, release, loc, ip)

    @staticmethod
    def filter(
        buffer: Any,
        filter_type: Any,
        cutoff: Any,
        resonance: Any,
        loc: Any,
        ip: Any
    ) -> Any:
        """Create filter operation."""
        return AudioFilterOp.create(buffer, filter_type, cutoff, resonance, loc, ip)

    @staticmethod
    def mix(
        buffers: List[Any],
        gains: List[Any],
        loc: Any,
        ip: Any
    ) -> Any:
        """Create mix operation."""
        return AudioMixOp.create(buffers, gains, loc, ip)

    @staticmethod
    def is_audio_op(op: Any) -> bool:
        """Check if operation is an audio dialect operation.

        Args:
            op: MLIR operation

        Returns:
            True if op is an audio operation
        """
        if not hasattr(op, "operation"):
            return False

        if not hasattr(op.operation, "attributes"):
            return False

        if "op_name" not in op.operation.attributes:
            return False

        op_name = str(op.operation.attributes["op_name"])
        return "morphogen.audio." in op_name

    @staticmethod
    def get_audio_op_name(op: Any) -> Optional[str]:
        """Get the audio operation name.

        Args:
            op: MLIR operation

        Returns:
            Operation name string (e.g., "morphogen.audio.buffer.create") or None
        """
        if not AudioDialect.is_audio_op(op):
            return None

        op_name_attr = op.operation.attributes["op_name"]
        # Remove quotes from StringAttr representation
        return str(op_name_attr).strip('"')
