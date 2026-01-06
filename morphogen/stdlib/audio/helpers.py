"""Shared helper functions for audio processing.

This module contains internal utilities used across audio submodules.
"""

import numpy as np


def apply_iir_filter(signal: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Apply IIR filter using Direct Form II.

    Args:
        signal: Input signal array
        b: Feedforward (numerator) coefficients
        a: Feedback (denominator) coefficients

    Returns:
        Filtered signal array
    """
    # Normalize coefficients
    a0 = a[0]
    if abs(a0) < 1e-10:
        return signal.copy()

    b = b / a0
    a = a / a0

    # Initialize state
    n_b = len(b)
    n_a = len(a)
    max_order = max(n_b, n_a) - 1

    # Pad coefficients
    if n_b < max_order + 1:
        b = np.pad(b, (0, max_order + 1 - n_b))
    if n_a < max_order + 1:
        a = np.pad(a, (0, max_order + 1 - n_a))

    # Apply filter
    output = np.zeros_like(signal)
    state = np.zeros(max_order)

    for i in range(len(signal)):
        # Direct Form II
        w = signal[i]
        for j in range(1, len(a)):
            w -= a[j] * state[j - 1] if j - 1 < len(state) else 0

        output[i] = b[0] * w
        for j in range(1, len(b)):
            output[i] += b[j] * state[j - 1] if j - 1 < len(state) else 0

        # Update state
        state = np.roll(state, 1)
        state[0] = w

    return output
