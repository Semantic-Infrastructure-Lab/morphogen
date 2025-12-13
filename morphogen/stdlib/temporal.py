"""
Temporal Logic and Scheduling Domain

Provides temporal reasoning, timing, and scheduling operations:
- Temporal logic operators (delay, history, temporal queries)
- Event detection (edge detection, threshold crossings)
- Scheduling primitives (timers, rate conversion, clocks)
- Sequence analysis (pattern matching, temporal correlation)
- Synchronization (barriers, triggers, gates)

Follows Morphogen's immutability pattern: all operations return new instances.

Version: v0.10.0
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Callable, List, Any
from enum import Enum
from collections import deque

from morphogen.core.operator import operator, OpCategory


class EdgeType(Enum):
    """Edge detection types"""
    RISING = "rising"
    FALLING = "falling"
    BOTH = "both"


class ClockPhase(Enum):
    """Clock phase states"""
    HIGH = "high"
    LOW = "low"


@dataclass
class DelayLine:
    """Delay line for storing past values

    Attributes:
        buffer: Circular buffer of past values
        max_delay: Maximum delay in samples
        current_index: Current write position
    """
    buffer: np.ndarray
    max_delay: int
    current_index: int = 0

    def copy(self) -> 'DelayLine':
        """Create a deep copy"""
        return DelayLine(
            buffer=self.buffer.copy(),
            max_delay=self.max_delay,
            current_index=self.current_index
        )


@dataclass
class Timer:
    """Timer for measuring elapsed time

    Attributes:
        start_time: Start time in samples
        duration: Timer duration in samples
        elapsed: Samples elapsed since start
        active: Whether timer is running
    """
    start_time: float
    duration: float
    elapsed: float = 0.0
    active: bool = True

    def copy(self) -> 'Timer':
        """Create a deep copy"""
        return Timer(
            start_time=self.start_time,
            duration=self.duration,
            elapsed=self.elapsed,
            active=self.active
        )


@dataclass
class Clock:
    """Clock generator with adjustable rate

    Attributes:
        rate: Clock rate in Hz
        phase: Current phase (HIGH or LOW)
        sample_count: Total samples elapsed
        duty_cycle: Duty cycle [0, 1]
    """
    rate: float
    phase: ClockPhase
    sample_count: int = 0
    duty_cycle: float = 0.5

    def copy(self) -> 'Clock':
        """Create a deep copy"""
        return Clock(
            rate=self.rate,
            phase=self.phase,
            sample_count=self.sample_count,
            duty_cycle=self.duty_cycle
        )


@dataclass
class EventSequence:
    """Sequence of timestamped events

    Attributes:
        timestamps: Event timestamps (in samples)
        values: Event values (optional payload)
    """
    timestamps: np.ndarray
    values: Optional[np.ndarray] = None

    def copy(self) -> 'EventSequence':
        """Create a deep copy"""
        return EventSequence(
            timestamps=self.timestamps.copy(),
            values=self.values.copy() if self.values is not None else None
        )


class TemporalOperations:
    """Temporal logic and scheduling operations"""

    @staticmethod
    @operator(
        domain="temporal",
        category=OpCategory.CONSTRUCT,
        signature="(max_delay: int) -> DelayLine",
        deterministic=True,
        doc="Create a delay line for storing past values"
    )
    def create_delay_line(max_delay: int) -> DelayLine:
        """Create a delay line for storing past values

        Args:
            max_delay: Maximum delay in samples

        Returns:
            DelayLine instance
        """
        return DelayLine(
            buffer=np.zeros(max_delay),
            max_delay=max_delay,
            current_index=0
        )

    @staticmethod
    @operator(
        domain="temporal",
        category=OpCategory.TRANSFORM,
        signature="(delay_line: DelayLine, value: float) -> DelayLine",
        deterministic=True,
        doc="Write value to delay line and advance"
    )
    def delay_write(delay_line: DelayLine, value: float) -> DelayLine:
        """Write value to delay line and advance

        Args:
            delay_line: Delay line to write to
            value: Value to write

        Returns:
            Updated delay line
        """
        result = delay_line.copy()
        result.buffer[result.current_index] = value
        result.current_index = (result.current_index + 1) % result.max_delay
        return result

    @staticmethod
    @operator(
        domain="temporal",
        category=OpCategory.QUERY,
        signature="(delay_line: DelayLine, delay: int) -> float",
        deterministic=True,
        doc="Read value from delay line at specified delay"
    )
    def delay_read(delay_line: DelayLine, delay: int) -> float:
        """Read value from delay line at specified delay

        Args:
            delay_line: Delay line to read from
            delay: Delay in samples (0 = most recent)

        Returns:
            Delayed value
        """
        if delay >= delay_line.max_delay:
            raise ValueError(f"Delay {delay} exceeds max delay {delay_line.max_delay}")

        read_index = (delay_line.current_index - delay - 1) % delay_line.max_delay
        return float(delay_line.buffer[read_index])

    @staticmethod
    @operator(
        domain="temporal",
        category=OpCategory.CONSTRUCT,
        signature="(duration: float, start_time: float) -> Timer",
        deterministic=True,
        doc="Create a timer"
    )
    def create_timer(duration: float, start_time: float = 0.0) -> Timer:
        """Create a timer

        Args:
            duration: Timer duration in seconds
            start_time: Start time in seconds

        Returns:
            Timer instance
        """
        return Timer(
            start_time=start_time,
            duration=duration,
            elapsed=0.0,
            active=True
        )

    @staticmethod
    @operator(
        domain="temporal",
        category=OpCategory.TRANSFORM,
        signature="(timer: Timer, dt: float) -> Timer",
        deterministic=True,
        doc="Advance timer by time step"
    )
    def timer_tick(timer: Timer, dt: float) -> Timer:
        """Advance timer by time step

        Args:
            timer: Timer to advance
            dt: Time step in seconds

        Returns:
            Updated timer
        """
        result = timer.copy()
        if result.active:
            result.elapsed += dt
            if result.elapsed >= result.duration:
                result.active = False
        return result

    @staticmethod
    @operator(
        domain="temporal",
        category=OpCategory.QUERY,
        signature="(timer: Timer) -> bool",
        deterministic=True,
        doc="Check if timer has expired"
    )
    def timer_expired(timer: Timer) -> bool:
        """Check if timer has expired

        Args:
            timer: Timer to check

        Returns:
            True if timer has elapsed
        """
        return timer.elapsed >= timer.duration

    @staticmethod
    @operator(
        domain="temporal",
        category=OpCategory.CONSTRUCT,
        signature="(rate: float, duty_cycle: float) -> Clock",
        deterministic=True,
        doc="Create a clock generator"
    )
    def create_clock(rate: float, duty_cycle: float = 0.5) -> Clock:
        """Create a clock generator

        Args:
            rate: Clock rate in Hz
            duty_cycle: Duty cycle [0, 1]

        Returns:
            Clock instance
        """
        return Clock(
            rate=rate,
            phase=ClockPhase.LOW,
            sample_count=0,
            duty_cycle=duty_cycle
        )

    @staticmethod
    @operator(
        domain="temporal",
        category=OpCategory.TRANSFORM,
        signature="(clock: Clock, sample_rate: float) -> Clock",
        deterministic=True,
        doc="Advance clock by one sample"
    )
    def clock_tick(clock: Clock, sample_rate: float) -> Clock:
        """Advance clock by one sample

        Args:
            clock: Clock to advance
            sample_rate: Sample rate in Hz

        Returns:
            Updated clock
        """
        result = clock.copy()
        result.sample_count += 1

        # Calculate phase based on clock rate and duty cycle
        period_samples = sample_rate / result.rate
        phase_in_period = (result.sample_count % period_samples) / period_samples

        if phase_in_period < result.duty_cycle:
            result.phase = ClockPhase.HIGH
        else:
            result.phase = ClockPhase.LOW

        return result

    @staticmethod
    @operator(
        domain="temporal",
        category=OpCategory.QUERY,
        signature="(clock: Clock) -> bool",
        deterministic=True,
        doc="Check if clock is high"
    )
    def clock_is_high(clock: Clock) -> bool:
        """Check if clock is high

        Args:
            clock: Clock to check

        Returns:
            True if clock phase is HIGH
        """
        return clock.phase == ClockPhase.HIGH

    @staticmethod
    @operator(
        domain="temporal",
        category=OpCategory.TRANSFORM,
        signature="(signal: ndarray, edge_type: str) -> ndarray",
        deterministic=True,
        doc="Detect edges in signal"
    )
    def detect_edges(signal: np.ndarray, edge_type: str = "rising") -> np.ndarray:
        """Detect edges in signal

        Args:
            signal: Input signal (boolean or numeric)
            edge_type: Edge type ('rising', 'falling', 'both')

        Returns:
            Boolean array marking edge locations
        """
        signal_bool = np.asarray(signal, dtype=bool)
        diff = np.diff(signal_bool.astype(int))

        edges = np.zeros(len(signal), dtype=bool)

        if edge_type == "rising" or edge_type == "both":
            rising = np.where(diff > 0)[0] + 1
            edges[rising] = True

        if edge_type == "falling" or edge_type == "both":
            falling = np.where(diff < 0)[0] + 1
            edges[falling] = True

        return edges

    @staticmethod
    @operator(
        domain="temporal",
        category=OpCategory.TRANSFORM,
        signature="(signal: ndarray, threshold: float, edge_type: str) -> ndarray",
        deterministic=True,
        doc="Detect threshold crossings in signal"
    )
    def threshold_crossings(signal: np.ndarray, threshold: float,
                          edge_type: str = "rising") -> np.ndarray:
        """Detect threshold crossings in signal

        Args:
            signal: Input signal
            threshold: Threshold value
            edge_type: Edge type ('rising', 'falling', 'both')

        Returns:
            Boolean array marking crossing locations
        """
        above_threshold = signal >= threshold
        return TemporalOperations.detect_edges(above_threshold, edge_type)

    @staticmethod
    @operator(
        domain="temporal",
        category=OpCategory.CONSTRUCT,
        signature="(timestamps: ndarray, values: ndarray) -> EventSequence",
        deterministic=True,
        doc="Create event sequence from timestamps and values"
    )
    def create_event_sequence(timestamps: np.ndarray,
                            values: Optional[np.ndarray] = None) -> EventSequence:
        """Create event sequence from timestamps and values

        Args:
            timestamps: Event timestamps
            values: Event values (optional)

        Returns:
            EventSequence instance
        """
        return EventSequence(
            timestamps=np.asarray(timestamps),
            values=np.asarray(values) if values is not None else None
        )

    @staticmethod
    @operator(
        domain="temporal",
        category=OpCategory.QUERY,
        signature="(sequence: EventSequence, start_time: float, end_time: float) -> EventSequence",
        deterministic=True,
        doc="Filter events within time range"
    )
    def filter_events(sequence: EventSequence, start_time: float,
                     end_time: float) -> EventSequence:
        """Filter events within time range

        Args:
            sequence: Event sequence
            start_time: Start time (inclusive)
            end_time: End time (exclusive)

        Returns:
            Filtered event sequence
        """
        mask = (sequence.timestamps >= start_time) & (sequence.timestamps < end_time)

        return EventSequence(
            timestamps=sequence.timestamps[mask],
            values=sequence.values[mask] if sequence.values is not None else None
        )

    @staticmethod
    @operator(
        domain="temporal",
        category=OpCategory.QUERY,
        signature="(sequence: EventSequence) -> int",
        deterministic=True,
        doc="Count events in sequence"
    )
    def count_events(sequence: EventSequence) -> int:
        """Count events in sequence

        Args:
            sequence: Event sequence

        Returns:
            Number of events
        """
        return len(sequence.timestamps)

    @staticmethod
    @operator(
        domain="temporal",
        category=OpCategory.TRANSFORM,
        signature="(signal: ndarray, window_size: int) -> ndarray",
        deterministic=True,
        doc="Apply sliding window operation"
    )
    def sliding_window(signal: np.ndarray, window_size: int) -> np.ndarray:
        """Apply sliding window operation

        Args:
            signal: Input signal
            window_size: Window size in samples

        Returns:
            2D array of windowed views (time x window_size)
        """
        n = len(signal)
        if window_size > n:
            raise ValueError(f"Window size {window_size} exceeds signal length {n}")

        shape = (n - window_size + 1, window_size)
        strides = (signal.strides[0], signal.strides[0])
        return np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)

    @staticmethod
    @operator(
        domain="temporal",
        category=OpCategory.TRANSFORM,
        signature="(signal: ndarray, rate_in: float, rate_out: float) -> ndarray",
        deterministic=False,
        doc="Resample signal to different rate"
    )
    def resample(signal: np.ndarray, rate_in: float, rate_out: float) -> np.ndarray:
        """Resample signal to different rate

        Args:
            signal: Input signal
            rate_in: Input sample rate
            rate_out: Output sample rate

        Returns:
            Resampled signal
        """
        if rate_in == rate_out:
            return signal.copy()

        from scipy import signal as sp_signal

        num_samples = int(len(signal) * rate_out / rate_in)
        return sp_signal.resample(signal, num_samples)

    @staticmethod
    @operator(
        domain="temporal",
        category=OpCategory.QUERY,
        signature="(sequence: EventSequence) -> float",
        deterministic=True,
        doc="Calculate mean inter-event interval"
    )
    def mean_interval(sequence: EventSequence) -> float:
        """Calculate mean inter-event interval

        Args:
            sequence: Event sequence

        Returns:
            Mean interval between events
        """
        if len(sequence.timestamps) < 2:
            return 0.0

        intervals = np.diff(sequence.timestamps)
        return float(np.mean(intervals))

    @staticmethod
    @operator(
        domain="temporal",
        category=OpCategory.QUERY,
        signature="(signal: ndarray, condition: ndarray) -> bool",
        deterministic=True,
        doc="Check if condition holds for entire signal (always)"
    )
    def temporal_always(signal: np.ndarray, condition: np.ndarray) -> bool:
        """Check if condition holds for entire signal (temporal logic 'always')

        Args:
            signal: Input signal
            condition: Boolean condition array

        Returns:
            True if condition holds for all samples
        """
        return bool(np.all(condition))

    @staticmethod
    @operator(
        domain="temporal",
        category=OpCategory.QUERY,
        signature="(signal: ndarray, condition: ndarray) -> bool",
        deterministic=True,
        doc="Check if condition holds at any point (eventually)"
    )
    def temporal_eventually(signal: np.ndarray, condition: np.ndarray) -> bool:
        """Check if condition holds at any point (temporal logic 'eventually')

        Args:
            signal: Input signal
            condition: Boolean condition array

        Returns:
            True if condition holds for at least one sample
        """
        return bool(np.any(condition))

    @staticmethod
    @operator(
        domain="temporal",
        category=OpCategory.QUERY,
        signature="(signal: ndarray, condition_a: ndarray, condition_b: ndarray) -> bool",
        deterministic=True,
        doc="Check if condition_a holds until condition_b becomes true"
    )
    def temporal_until(signal: np.ndarray, condition_a: np.ndarray,
                      condition_b: np.ndarray) -> bool:
        """Check if condition_a holds until condition_b becomes true (temporal logic 'until')

        Args:
            signal: Input signal
            condition_a: Condition that must hold
            condition_b: Condition that releases the constraint

        Returns:
            True if a holds until b becomes true
        """
        # Find first occurrence of condition_b
        b_indices = np.where(condition_b)[0]

        if len(b_indices) == 0:
            # If b never occurs, a must hold everywhere
            return bool(np.all(condition_a))

        # Check if a holds until first b
        first_b = b_indices[0]
        return bool(np.all(condition_a[:first_b]))

    @staticmethod
    @operator(
        domain="temporal",
        category=OpCategory.TRANSFORM,
        signature="(signal: ndarray, lag: int) -> ndarray",
        deterministic=True,
        doc="Create lagged version of signal"
    )
    def lag(signal: np.ndarray, lag: int) -> np.ndarray:
        """Create lagged version of signal

        Args:
            signal: Input signal
            lag: Number of samples to lag (positive = past, negative = future)

        Returns:
            Lagged signal (padded with zeros)
        """
        if lag == 0:
            return signal.copy()

        result = np.zeros_like(signal)

        if lag > 0:
            # Shift forward (delay)
            result[lag:] = signal[:-lag]
        else:
            # Shift backward (advance)
            result[:lag] = signal[-lag:]

        return result

    @staticmethod
    @operator(
        domain="temporal",
        category=OpCategory.TRANSFORM,
        signature="(signal: ndarray, lags: list) -> ndarray",
        deterministic=True,
        doc="Create multiple lagged versions"
    )
    def create_lags(signal: np.ndarray, lags: List[int]) -> np.ndarray:
        """Create multiple lagged versions

        Args:
            signal: Input signal
            lags: List of lag values

        Returns:
            2D array (time x lags)
        """
        result = np.zeros((len(signal), len(lags)))

        for i, lag in enumerate(lags):
            result[:, i] = TemporalOperations.lag(signal, lag)

        return result

    @staticmethod
    @operator(
        domain="temporal",
        category=OpCategory.TRANSFORM,
        signature="(signal: ndarray) -> ndarray",
        deterministic=True,
        doc="Compute first difference (discrete derivative)"
    )
    def difference(signal: np.ndarray) -> np.ndarray:
        """Compute first difference (discrete derivative)

        Args:
            signal: Input signal

        Returns:
            First difference (length n-1)
        """
        return np.diff(signal)

    @staticmethod
    @operator(
        domain="temporal",
        category=OpCategory.TRANSFORM,
        signature="(signal: ndarray, initial: float) -> ndarray",
        deterministic=True,
        doc="Compute cumulative sum (discrete integral)"
    )
    def cumsum(signal: np.ndarray, initial: float = 0.0) -> np.ndarray:
        """Compute cumulative sum (discrete integral)

        Args:
            signal: Input signal
            initial: Initial value

        Returns:
            Cumulative sum
        """
        result = np.cumsum(signal)
        if initial != 0.0:
            result += initial
        return result


# Export operators for domain registry discovery
create_delay_line = TemporalOperations.create_delay_line
delay_write = TemporalOperations.delay_write
delay_read = TemporalOperations.delay_read
create_timer = TemporalOperations.create_timer
timer_tick = TemporalOperations.timer_tick
timer_expired = TemporalOperations.timer_expired
create_clock = TemporalOperations.create_clock
clock_tick = TemporalOperations.clock_tick
clock_is_high = TemporalOperations.clock_is_high
detect_edges = TemporalOperations.detect_edges
threshold_crossings = TemporalOperations.threshold_crossings
create_event_sequence = TemporalOperations.create_event_sequence
filter_events = TemporalOperations.filter_events
count_events = TemporalOperations.count_events
sliding_window = TemporalOperations.sliding_window
resample = TemporalOperations.resample
mean_interval = TemporalOperations.mean_interval
temporal_always = TemporalOperations.temporal_always
temporal_eventually = TemporalOperations.temporal_eventually
temporal_until = TemporalOperations.temporal_until
lag = TemporalOperations.lag
create_lags = TemporalOperations.create_lags
difference = TemporalOperations.difference
cumsum = TemporalOperations.cumsum

# Export main classes and operations
__all__ = [
    'DelayLine',
    'Timer',
    'Clock',
    'EventSequence',
    'EdgeType',
    'ClockPhase',
    'TemporalOperations',
]
