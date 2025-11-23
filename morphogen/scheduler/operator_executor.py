"""
Operator Execution System for Morphogen Scheduler

This module provides operator discovery, registration, and execution for the
SimplifiedScheduler. It replaces the mock operator execution with real operator calls.

Key features:
- Automatic operator discovery from stdlib
- Parameter conversion (GraphIR params → operator args)
- AudioBuffer ↔ np.ndarray conversion
- Stateful operator management (phase continuity, filter state, etc.)
"""

from typing import Dict, Any, Callable, Optional
import numpy as np
import inspect
from morphogen.core.operator import get_operator_metadata, is_operator


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
    Executes operators with proper parameter conversion and state management.

    Handles:
    - np.ndarray ↔ AudioBuffer conversion
    - Parameter parsing (e.g., "440Hz" → 440.0)
    - Stateful operators (phase continuity, filter state)
    - Multi-output operators
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

    def execute(
        self,
        operator_name: str,
        node_id: str,
        params: Dict[str, Any],
        inputs: Dict[str, np.ndarray],
        num_samples: int,
        rate_hz: float,
    ) -> Dict[str, np.ndarray]:
        """
        Execute an operator.

        Args:
            operator_name: Operator name (e.g., "sine")
            node_id: Unique node ID (for state management)
            params: Operator parameters from GraphIR
            inputs: Input buffers (port_name → np.ndarray)
            num_samples: Number of samples to generate
            rate_hz: Execution rate in Hz

        Returns:
            Output buffers (port_name → np.ndarray)
        """
        # Get operator function
        op_func = self.registry.get(operator_name)
        if op_func is None:
            # Operator not found - return zeros
            return {"out": np.zeros(num_samples)}

        # Convert inputs to AudioBuffer if needed
        from morphogen.stdlib.audio import AudioBuffer
        audio_inputs = {}
        for port_name, buffer in inputs.items():
            audio_inputs[port_name] = AudioBuffer(data=buffer, sample_rate=int(rate_hz))

        # Parse parameters
        parsed_params = self._parse_params(params, rate_hz, num_samples)

        # Build operator arguments
        op_args = {}
        sig = inspect.signature(op_func)

        # Load operator state (for stateful operators like oscillators, filters)
        node_state = self._operator_state.get(node_id, {})

        # Match parameters to function signature
        for param_name in sig.parameters:
            if param_name in audio_inputs:
                # Use input buffer
                op_args[param_name] = audio_inputs[param_name]
            elif param_name in parsed_params:
                # Use parsed parameter
                op_args[param_name] = parsed_params[param_name]
            elif param_name == "sample_rate":
                # Provide sample rate
                op_args[param_name] = int(rate_hz)
            elif param_name == "duration":
                # Calculate duration from num_samples
                op_args[param_name] = num_samples / rate_hz
            elif param_name == "phase" and "phase" in node_state:
                # Inject phase state for oscillators
                op_args[param_name] = node_state["phase"]

        # Execute operator
        try:
            result = op_func(**op_args)
        except Exception as e:
            print(f"Error executing operator '{operator_name}': {e}")
            return {"out": np.zeros(num_samples)}

        # Update operator state after execution (phase continuity for oscillators)
        if "freq" in parsed_params and "phase" in sig.parameters:
            # This is an oscillator - calculate final phase
            freq = parsed_params["freq"]
            duration = num_samples / rate_hz
            phase_advance = (2.0 * np.pi * freq * duration) % (2.0 * np.pi)
            current_phase = op_args.get("phase", 0.0)
            final_phase = (current_phase + phase_advance) % (2.0 * np.pi)

            # Save state for next execution
            if node_id not in self._operator_state:
                self._operator_state[node_id] = {}
            self._operator_state[node_id]["phase"] = final_phase

        # Convert result to output dict
        outputs = {}

        if isinstance(result, AudioBuffer):
            # Single AudioBuffer output
            outputs["out"] = result.data[:num_samples]
        elif isinstance(result, np.ndarray):
            # Raw numpy array
            outputs["out"] = result[:num_samples]
        elif isinstance(result, dict):
            # Dictionary of outputs
            for port_name, value in result.items():
                if isinstance(value, AudioBuffer):
                    outputs[port_name] = value.data[:num_samples]
                elif isinstance(value, np.ndarray):
                    outputs[port_name] = value[:num_samples]
                else:
                    outputs[port_name] = np.array([value] * num_samples)
        else:
            # Scalar or other - convert to constant buffer
            outputs["out"] = np.array([result] * num_samples)

        return outputs

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
