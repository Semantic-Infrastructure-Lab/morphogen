"""
Operator Execution System for Morphogen Scheduler

This module provides operator discovery, registration, and execution for the
SimplifiedScheduler. It replaces the mock operator execution with real operator calls.

Key features:
- Automatic operator discovery from stdlib
- Parameter conversion (GraphIR params → operator args)
- AudioBuffer ↔ np.ndarray conversion
- Convention-based stateful operator management (phase continuity, filter state, etc.)
"""

from typing import Dict, Any, Callable, Optional
import numpy as np
import inspect
from morphogen.core.operator import get_operator_metadata, is_operator


# Convention-based stateful parameter detection
# Different state management patterns for different parameter types:
#
# EXECUTOR_MANAGED_STATE: Executor calculates and updates state after execution
#   - 'phase': Oscillator phase (float) - executor computes final phase from frequency
#
# OPERATOR_MANAGED_STATE: Operators update state via in-place AudioBuffer modification
#   - 'filter_state': Biquad filter state (AudioBuffer with 2 samples: [z1, z2])
EXECUTOR_MANAGED_STATE = {
    "phase": float,  # Oscillator phase continuity
}

OPERATOR_MANAGED_STATE = {
    "filter_state": 2,  # Biquad filter state (2 samples: [z1, z2])
}


class OperatorRegistry:
    """
    Registry of available operators discovered from stdlib.

    Maps operator names (e.g., "sine", "lowpass") to their implementation functions.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._operators: Dict[str, Callable] = {}
        self._metadata: Dict[str, Any] = {}

    def register(self, name: str, func: Callable) -> None:
        """
        Register an operator function.

        Args:
            name: Operator name (e.g., "sine")
            func: Operator function
        """
        self._operators[name] = func
        metadata = get_operator_metadata(func)
        if metadata:
            self._metadata[name] = metadata

    def get(self, name: str) -> Optional[Callable]:
        """
        Get operator function by name.

        Args:
            name: Operator name

        Returns:
            Operator function if found, None otherwise
        """
        return self._operators.get(name)

    def has(self, name: str) -> bool:
        """Check if operator is registered."""
        return name in self._operators

    def discover_from_module(self, module) -> int:
        """
        Discover and register operators from a module.

        Args:
            module: Python module to scan for operators

        Returns:
            Number of operators discovered
        """
        count = 0

        # Scan module for operator-decorated functions
        for attr_name in dir(module):
            attr = getattr(module, attr_name)

            # Skip private attributes
            if attr_name.startswith('_'):
                continue

            # Check if it's an operator
            if callable(attr) and is_operator(attr):
                # Use the attribute name as the operator name
                self.register(attr_name, attr)
                count += 1

        return count

    def list_operators(self) -> list:
        """List all registered operator names."""
        return sorted(self._operators.keys())


class OperatorExecutor:
    """
    Executes operators with proper parameter conversion and convention-based state management.

    Handles:
    - np.ndarray ↔ AudioBuffer conversion
    - Parameter parsing (e.g., "440Hz" → 440.0)
    - Convention-based stateful operators (phase continuity, filter state, etc.)
    - Multi-output operators

    State Management:
    Automatically detects and manages stateful parameters via convention-based naming.
    Two state management patterns:

    1. EXECUTOR-MANAGED STATE:
       - 'phase': Oscillator phase continuity (float)
       - Executor calculates final state after execution
       - Used when operators don't update state themselves

    2. OPERATOR-MANAGED STATE:
       - 'filter_state': Biquad filter state (AudioBuffer with 2 samples: [z1, z2])
       - Operators update state via in-place AudioBuffer modification
       - Used when operators handle their own state updates

    State is automatically:
    1. Created on first operator execution
    2. Injected into operator calls
    3. Updated (by executor or operator, depending on pattern)
    4. Persisted across executions (per node_id)

    No manual state management required!
    """

    def __init__(self, registry: OperatorRegistry, sample_rate: int = 48000):
        """
        Initialize executor.

        Args:
            registry: OperatorRegistry instance
            sample_rate: Audio sample rate in Hz
        """
        self.registry = registry
        self.sample_rate = sample_rate

        # Operator state management
        self._operator_state: Dict[str, Dict[str, Any]] = {}

    def _build_op_args(self, sig, audio_inputs, parsed_params, node_state, rate_hz, num_samples):
        """Match operator function parameters to available inputs, params, and state."""
        from morphogen.stdlib.audio import AudioBuffer
        op_args = {}
        for param_name in sig.parameters:
            if param_name in audio_inputs:
                op_args[param_name] = audio_inputs[param_name]
            elif param_name in parsed_params:
                op_args[param_name] = parsed_params[param_name]
            elif param_name == "sample_rate":
                op_args[param_name] = int(rate_hz)
            elif param_name == "duration":
                op_args[param_name] = num_samples / rate_hz
            elif param_name in EXECUTOR_MANAGED_STATE:
                if param_name not in node_state:
                    node_state[param_name] = 0.0
                op_args[param_name] = node_state[param_name]
            elif param_name in OPERATOR_MANAGED_STATE:
                if param_name not in node_state:
                    state_size = OPERATOR_MANAGED_STATE[param_name]
                    node_state[param_name] = AudioBuffer(
                        data=np.zeros(state_size), sample_rate=int(rate_hz)
                    )
                op_args[param_name] = node_state[param_name]
        return op_args

    def _update_phase_state(self, node_state, parsed_params, op_args, sig, rate_hz, num_samples):
        """Advance and persist oscillator phase for phase continuity."""
        if "freq" in parsed_params and "phase" in EXECUTOR_MANAGED_STATE and "phase" in sig.parameters:
            freq = parsed_params["freq"]
            phase_advance = (2.0 * np.pi * freq * num_samples / rate_hz) % (2.0 * np.pi)
            node_state["phase"] = (op_args.get("phase", 0.0) + phase_advance) % (2.0 * np.pi)

    @staticmethod
    def _convert_result(result, num_samples) -> Dict[str, np.ndarray]:
        """Convert an operator return value to a port→ndarray output dict."""
        from morphogen.stdlib.audio import AudioBuffer
        if isinstance(result, AudioBuffer):
            return {"out": result.data[:num_samples]}
        if isinstance(result, np.ndarray):
            return {"out": result[:num_samples]}
        if isinstance(result, dict):
            outputs = {}
            for port_name, value in result.items():
                if isinstance(value, AudioBuffer):
                    outputs[port_name] = value.data[:num_samples]
                elif isinstance(value, np.ndarray):
                    outputs[port_name] = value[:num_samples]
                else:
                    outputs[port_name] = np.array([value] * num_samples)
            return outputs
        return {"out": np.array([result] * num_samples)}

    def execute(
        self,
        operator_name: str,
        node_id: str,
        params: Dict[str, Any],
        inputs: Dict[str, np.ndarray],
        num_samples: int,
        rate_hz: float,
    ) -> Dict[str, np.ndarray]:
        """Execute an operator and return output buffers (port_name → np.ndarray)."""
        op_func = self.registry.get(operator_name)
        if op_func is None:
            return {"out": np.zeros(num_samples)}

        from morphogen.stdlib.audio import AudioBuffer
        audio_inputs = {
            port: AudioBuffer(data=buf, sample_rate=int(rate_hz))
            for port, buf in inputs.items()
        }
        parsed_params = self._parse_params(params, rate_hz, num_samples)
        sig = inspect.signature(op_func)

        if node_id not in self._operator_state:
            self._operator_state[node_id] = {}
        node_state = self._operator_state[node_id]

        op_args = self._build_op_args(sig, audio_inputs, parsed_params, node_state, rate_hz, num_samples)

        try:
            result = op_func(**op_args)
        except Exception as e:
            print(f"Error executing operator '{operator_name}': {e}")
            return {"out": np.zeros(num_samples)}

        self._update_phase_state(node_state, parsed_params, op_args, sig, rate_hz, num_samples)
        return self._convert_result(result, num_samples)

    def _parse_params(
        self,
        params: Dict[str, Any],
        rate_hz: float,
        num_samples: int,
    ) -> Dict[str, Any]:
        """
        Parse GraphIR parameters for operator execution.

        Handles:
        - Unit annotations (e.g., "440Hz" → 440.0)
        - Rate-dependent parameters
        - Type conversions

        Args:
            params: Raw parameters from GraphIR
            rate_hz: Execution rate in Hz
            num_samples: Number of samples

        Returns:
            Parsed parameters ready for operator
        """
        parsed = {}

        for key, value in params.items():
            if isinstance(value, str):
                # Parse unit annotations
                if value.endswith("Hz"):
                    # Frequency
                    parsed[key] = float(value[:-2])
                elif value.endswith("s"):
                    # Time in seconds
                    parsed[key] = float(value[:-1])
                elif value.endswith("dB"):
                    # Decibels
                    parsed[key] = float(value[:-2])
                elif value.endswith("ms"):
                    # Milliseconds
                    parsed[key] = float(value[:-2]) / 1000.0
                else:
                    # Try to parse as float, otherwise keep as string
                    try:
                        parsed[key] = float(value)
                    except ValueError:
                        parsed[key] = value
            else:
                # Pass through as-is
                parsed[key] = value

        return parsed


def create_audio_registry() -> OperatorRegistry:
    """
    Create and populate operator registry with audio stdlib operators.

    Returns:
        OperatorRegistry with all audio operators registered
    """
    registry = OperatorRegistry()

    # Discover operators from audio module
    from morphogen.stdlib import audio
    count = registry.discover_from_module(audio)

    print(f"Discovered {count} audio operators")

    return registry
