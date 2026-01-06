"""Audio operations for deterministic audio synthesis.

This package provides NumPy-based implementations of all core audio operations
for deterministic audio synthesis, including oscillators, filters, envelopes,
effects, and physical modeling primitives.

All operations follow the audio-rate model (44.1kHz default) with deterministic
semantics ensuring same seed = same output.
"""

# Core types
from .buffer import AudioBuffer, DEFAULT_SAMPLE_RATE, DEFAULT_CONTROL_RATE

# Oscillators
from .oscillators import sine, saw, square, triangle, noise, impulse, constant

# Filters
from .filters import (
    lowpass, highpass, bandpass, notch, eq3,
    vcf_lowpass, vcf_highpass, vcf_bandpass,
    # Biquad coefficient calculators (for advanced use)
    biquad_lowpass, biquad_highpass, biquad_bandpass, biquad_notch,
    biquad_low_shelf, biquad_high_shelf, biquad_peaking,
    # Time-varying filter functions
    apply_time_varying_lowpass, apply_time_varying_highpass, apply_time_varying_bandpass,
)

# Envelopes
from .envelopes import adsr, ar, envexp

# Effects
from .effects import delay, reverb, chorus, flanger, drive, limiter

# Mixing and utilities
from .mixing import mix, gain, multiply, vca, pan, clip, normalize, db2lin, lin2db

# Physical modeling synthesis
from .synthesis import string, modal

# I/O
from .io import play, save, load, record

# Buffer operations
from .transform import slice, concat, resample, reverse, fade_in, fade_out

# Analysis
from .analysis import (
    fft, ifft, stft, istft,
    spectrum, phase_spectrum,
    spectral_centroid, spectral_rolloff, spectral_flux, spectral_peaks,
    rms, zero_crossings,
    spectral_gate, spectral_filter, convolution,
)

# Helper functions
from .helpers import apply_iir_filter


# Backward compatibility: AudioOperations class that wraps all functions
class AudioOperations:
    """Namespace for audio operations (accessed as 'audio' in DSL).

    This class provides backward compatibility with code that used
    AudioOperations.method() syntax. All methods are now also available
    as module-level functions.
    """

    # Core types
    AudioBuffer = AudioBuffer
    DEFAULT_SAMPLE_RATE = DEFAULT_SAMPLE_RATE
    DEFAULT_CONTROL_RATE = DEFAULT_CONTROL_RATE

    # Oscillators
    sine = staticmethod(sine)
    saw = staticmethod(saw)
    square = staticmethod(square)
    triangle = staticmethod(triangle)
    noise = staticmethod(noise)
    impulse = staticmethod(impulse)
    constant = staticmethod(constant)

    # Filters
    lowpass = staticmethod(lowpass)
    highpass = staticmethod(highpass)
    bandpass = staticmethod(bandpass)
    notch = staticmethod(notch)
    eq3 = staticmethod(eq3)
    vcf_lowpass = staticmethod(vcf_lowpass)
    vcf_highpass = staticmethod(vcf_highpass)
    vcf_bandpass = staticmethod(vcf_bandpass)

    # Envelopes
    adsr = staticmethod(adsr)
    ar = staticmethod(ar)
    envexp = staticmethod(envexp)

    # Effects
    delay = staticmethod(delay)
    reverb = staticmethod(reverb)
    chorus = staticmethod(chorus)
    flanger = staticmethod(flanger)
    drive = staticmethod(drive)
    limiter = staticmethod(limiter)

    # Mixing
    mix = staticmethod(mix)
    gain = staticmethod(gain)
    multiply = staticmethod(multiply)
    vca = staticmethod(vca)
    pan = staticmethod(pan)
    clip = staticmethod(clip)
    normalize = staticmethod(normalize)
    db2lin = staticmethod(db2lin)
    lin2db = staticmethod(lin2db)

    # Synthesis
    string = staticmethod(string)
    modal = staticmethod(modal)

    # I/O
    play = staticmethod(play)
    save = staticmethod(save)
    load = staticmethod(load)
    record = staticmethod(record)

    # Transform
    slice = staticmethod(slice)
    concat = staticmethod(concat)
    resample = staticmethod(resample)
    reverse = staticmethod(reverse)
    fade_in = staticmethod(fade_in)
    fade_out = staticmethod(fade_out)

    # Analysis
    fft = staticmethod(fft)
    ifft = staticmethod(ifft)
    stft = staticmethod(stft)
    istft = staticmethod(istft)
    spectrum = staticmethod(spectrum)
    phase_spectrum = staticmethod(phase_spectrum)
    spectral_centroid = staticmethod(spectral_centroid)
    spectral_rolloff = staticmethod(spectral_rolloff)
    spectral_flux = staticmethod(spectral_flux)
    spectral_peaks = staticmethod(spectral_peaks)
    rms = staticmethod(rms)
    zero_crossings = staticmethod(zero_crossings)
    spectral_gate = staticmethod(spectral_gate)
    spectral_filter = staticmethod(spectral_filter)
    convolution = staticmethod(convolution)

    # Internal helpers (for backward compatibility)
    _apply_iir_filter = staticmethod(apply_iir_filter)
    _biquad_lowpass = staticmethod(biquad_lowpass)
    _biquad_highpass = staticmethod(biquad_highpass)
    _biquad_bandpass = staticmethod(biquad_bandpass)
    _biquad_notch = staticmethod(biquad_notch)
    _biquad_low_shelf = staticmethod(biquad_low_shelf)
    _biquad_high_shelf = staticmethod(biquad_high_shelf)
    _biquad_peaking = staticmethod(biquad_peaking)
    _apply_time_varying_lowpass = staticmethod(apply_time_varying_lowpass)
    _apply_time_varying_highpass = staticmethod(apply_time_varying_highpass)
    _apply_time_varying_bandpass = staticmethod(apply_time_varying_bandpass)


# Namespace instance for runtime (audio.method() syntax)
audio = AudioOperations()


__all__ = [
    # Core types
    'AudioBuffer', 'DEFAULT_SAMPLE_RATE', 'DEFAULT_CONTROL_RATE',

    # Namespace instance
    'audio',

    # Backward compatibility class
    'AudioOperations',

    # Oscillators
    'sine', 'saw', 'square', 'triangle', 'noise', 'impulse', 'constant',

    # Filters
    'lowpass', 'highpass', 'bandpass', 'notch', 'eq3',
    'vcf_lowpass', 'vcf_highpass', 'vcf_bandpass',

    # Envelopes
    'adsr', 'ar', 'envexp',

    # Effects
    'delay', 'reverb', 'chorus', 'flanger', 'drive', 'limiter',

    # Mixing
    'mix', 'gain', 'multiply', 'vca', 'pan', 'clip', 'normalize', 'db2lin', 'lin2db',

    # Synthesis
    'string', 'modal',

    # I/O
    'play', 'save', 'load', 'record',

    # Transform
    'slice', 'concat', 'resample', 'reverse', 'fade_in', 'fade_out',

    # Analysis
    'fft', 'ifft', 'stft', 'istft',
    'spectrum', 'phase_spectrum',
    'spectral_centroid', 'spectral_rolloff', 'spectral_flux', 'spectral_peaks',
    'rms', 'zero_crossings',
    'spectral_gate', 'spectral_filter', 'convolution',
]
