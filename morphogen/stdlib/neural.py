"""
Neural Network Domain
=====================

Simple feedforward neural network (MLP) implementation for Morphogen simulations.
Designed for fast batched inference in evolutionary/genetic algorithm contexts.

This domain provides:
- Dense layer operations (linear + activation)
- Common activation functions (tanh, relu, sigmoid, softmax)
- Batch inference for parallel agent evaluation
- Parameter initialization and manipulation for GA
- Deterministic operations for reproducibility

Layer 1: Atomic operators (linear, activation functions)
Layer 2: Composite operators (dense layer, forward pass)
Layer 3: Network constructs (MLP, parameter management)
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
from morphogen.core.operator import operator, OpCategory


@dataclass
class DenseLayer:
    """
    Represents a dense (fully-connected) neural network layer.

    Attributes:
        weights: Weight matrix [input_size, output_size]
        biases: Bias vector [output_size,]
        activation: Activation function name ('linear', 'tanh', 'relu', 'sigmoid')
    """
    weights: np.ndarray
    biases: np.ndarray
    activation: str = 'linear'

    def copy(self) -> 'DenseLayer':
        """Return a copy of this layer (immutable semantics)"""
        return DenseLayer(
            weights=self.weights.copy(),
            biases=self.biases.copy(),
            activation=self.activation
        )


@dataclass
class MLP:
    """
    Multi-layer perceptron (feedforward neural network).

    Attributes:
        layers: List of dense layers
        input_size: Size of input vector
        output_size: Size of output vector
    """
    layers: List[DenseLayer]
    input_size: int
    output_size: int

    def copy(self) -> 'MLP':
        """Return a deep copy of this network"""
        return MLP(
            layers=[layer.copy() for layer in self.layers],
            input_size=self.input_size,
            output_size=self.output_size
        )

    def count_parameters(self) -> int:
        """Count total number of trainable parameters"""
        return sum(layer.weights.size + layer.biases.size for layer in self.layers)


class NeuralOperations:
    """
    Namespace for neural network operations.

    Follows Morphogen's 4-layer operator hierarchy:
    - Layer 1: Atomic (linear, activations)
    - Layer 2: Composite (dense layer, forward pass)
    - Layer 3: Constructs (MLP, parameter ops)
    - Layer 4: Presets (common architectures)
    """

    # === LAYER 1: ATOMIC OPERATORS ===

    @staticmethod
    @operator(
        domain="neural",
        category=OpCategory.TRANSFORM,
        signature="(x: ndarray, weights: ndarray, biases: ndarray) -> ndarray",
        deterministic=True,
        doc="Linear transformation (matrix multiplication + bias)"
    )
    def linear(x: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> np.ndarray:
        """
        Layer 1: Linear transformation (matrix multiplication + bias).

        Supports both single and batch inputs:
        - Single: x [input_size,] -> output [output_size,]
        - Batch: x [batch_size, input_size] -> output [batch_size, output_size]

        Args:
            x: Input vector or batch
            weights: Weight matrix [input_size, output_size]
            biases: Bias vector [output_size,]

        Returns:
            Transformed output
        """
        return np.dot(x, weights) + biases

    @staticmethod
    @operator(
        domain="neural",
        category=OpCategory.TRANSFORM,
        signature="(x: ndarray) -> ndarray",
        deterministic=True,
        doc="Hyperbolic tangent activation"
    )
    def tanh(x: np.ndarray) -> np.ndarray:
        """Layer 1: Hyperbolic tangent activation [-1, 1]"""
        return np.tanh(x)

    @staticmethod
    @operator(
        domain="neural",
        category=OpCategory.TRANSFORM,
        signature="(x: ndarray) -> ndarray",
        deterministic=True,
        doc="Rectified linear unit activation"
    )
    def relu(x: np.ndarray) -> np.ndarray:
        """Layer 1: Rectified linear unit activation [0, inf)"""
        return np.maximum(0, x)

    @staticmethod
    @operator(
        domain="neural",
        category=OpCategory.TRANSFORM,
        signature="(x: ndarray) -> ndarray",
        deterministic=True,
        doc="Sigmoid activation"
    )
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Layer 1: Sigmoid activation [0, 1]"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))  # Clip for numerical stability

    @staticmethod
    @operator(
        domain="neural",
        category=OpCategory.TRANSFORM,
        signature="(x: ndarray, axis: int) -> ndarray",
        deterministic=True,
        doc="Softmax activation (normalized exponentials)"
    )
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Layer 1: Softmax activation (normalized exponentials).

        Args:
            x: Input logits
            axis: Axis along which to compute softmax

        Returns:
            Probability distribution (sums to 1 along axis)
        """
        # Subtract max for numerical stability
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    @staticmethod
    @operator(
        domain="neural",
        category=OpCategory.TRANSFORM,
        signature="(x: ndarray, alpha: float) -> ndarray",
        deterministic=True,
        doc="Leaky ReLU activation"
    )
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Layer 1: Leaky ReLU activation"""
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    @operator(
        domain="neural",
        category=OpCategory.TRANSFORM,
        signature="(x: ndarray, activation: str) -> ndarray",
        deterministic=True,
        doc="Apply named activation function"
    )
    def apply_activation(x: np.ndarray, activation: str) -> np.ndarray:
        """
        Layer 1: Apply named activation function.

        Args:
            x: Input array
            activation: One of ['linear', 'tanh', 'relu', 'sigmoid', 'softmax', 'leaky_relu']

        Returns:
            Activated output
        """
        if activation == 'linear' or activation is None:
            return x
        elif activation == 'tanh':
            return NeuralOperations.tanh(x)
        elif activation == 'relu':
            return NeuralOperations.relu(x)
        elif activation == 'sigmoid':
            return NeuralOperations.sigmoid(x)
        elif activation == 'softmax':
            return NeuralOperations.softmax(x)
        elif activation == 'leaky_relu':
            return NeuralOperations.leaky_relu(x)
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    # === LAYER 2: COMPOSITE OPERATORS ===

    @staticmethod
    @operator(
        domain="neural",
        category=OpCategory.TRANSFORM,
        signature="(x: ndarray, layer: DenseLayer) -> ndarray",
        deterministic=True,
        doc="Dense layer (linear + activation)"
    )
    def dense(x: np.ndarray, layer: DenseLayer) -> np.ndarray:
        """
        Layer 2: Dense layer (linear + activation).

        Args:
            x: Input array (single or batch)
            layer: Dense layer parameters

        Returns:
            Activated output
        """
        z = NeuralOperations.linear(x, layer.weights, layer.biases)
        return NeuralOperations.apply_activation(z, layer.activation)

    @staticmethod
    @operator(
        domain="neural",
        category=OpCategory.TRANSFORM,
        signature="(x: ndarray, network: MLP) -> ndarray",
        deterministic=True,
        doc="Forward pass through entire network"
    )
    def forward(x: np.ndarray, network: MLP) -> np.ndarray:
        """
        Layer 2: Forward pass through entire network.

        Args:
            x: Input vector or batch [batch_size, input_size]
            network: MLP network

        Returns:
            Network output [batch_size, output_size]
        """
        activations = x
        for layer in network.layers:
            activations = NeuralOperations.dense(activations, layer)
        return activations

    # === LAYER 3: NETWORK CONSTRUCTS ===

    @staticmethod
    @operator(
        domain="neural",
        category=OpCategory.CONSTRUCT,
        signature="(input_size: int, output_size: int, activation: str, init_method: str, seed: Optional[int]) -> DenseLayer",
        deterministic=True,
        doc="Allocate and initialize a dense layer"
    )
    def alloc_layer(input_size: int, output_size: int,
                    activation: str = 'linear',
                    init_method: str = 'xavier',
                    seed: Optional[int] = None) -> DenseLayer:
        """
        Layer 3: Allocate and initialize a dense layer.

        Initialization methods:
        - 'xavier': Xavier/Glorot uniform [-limit, limit] where limit = sqrt(6 / (fan_in + fan_out))
        - 'he': He uniform for ReLU networks
        - 'normal': Standard normal distribution
        - 'zeros': All zeros (not recommended except for debugging)

        Args:
            input_size: Number of input neurons
            output_size: Number of output neurons
            activation: Activation function name
            init_method: Weight initialization method
            seed: Random seed for deterministic initialization

        Returns:
            Initialized DenseLayer
        """
        if seed is not None:
            np.random.seed(seed)

        if init_method == 'xavier':
            limit = np.sqrt(6.0 / (input_size + output_size))
            weights = np.random.uniform(-limit, limit, (input_size, output_size)).astype(np.float32)
        elif init_method == 'he':
            std = np.sqrt(2.0 / input_size)
            weights = np.random.normal(0, std, (input_size, output_size)).astype(np.float32)
        elif init_method == 'normal':
            weights = np.random.randn(input_size, output_size).astype(np.float32) * 0.1
        elif init_method == 'zeros':
            weights = np.zeros((input_size, output_size), dtype=np.float32)
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")

        biases = np.zeros(output_size, dtype=np.float32)

        return DenseLayer(weights=weights, biases=biases, activation=activation)

    @staticmethod
    @operator(
        domain="neural",
        category=OpCategory.CONSTRUCT,
        signature="(layer_sizes: List[int], activations: Optional[List[str]], init_method: str, seed: Optional[int]) -> MLP",
        deterministic=True,
        doc="Allocate a multi-layer perceptron"
    )
    def alloc_mlp(layer_sizes: List[int],
                  activations: Optional[List[str]] = None,
                  init_method: str = 'xavier',
                  seed: Optional[int] = None) -> MLP:
        """
        Layer 3: Allocate a multi-layer perceptron.

        Example:
            # Create 4 -> 8 -> 8 -> 1 network with tanh hidden, sigmoid output
            network = NeuralOperations.alloc_mlp(
                layer_sizes=[4, 8, 8, 1],
                activations=['tanh', 'tanh', 'sigmoid']
            )

        Args:
            layer_sizes: List of layer sizes [input, hidden1, ..., output]
            activations: List of activation functions (length = len(layer_sizes) - 1)
                        If None, uses 'tanh' for hidden layers and 'linear' for output
            init_method: Weight initialization method
            seed: Random seed for deterministic initialization

        Returns:
            Initialized MLP
        """
        if len(layer_sizes) < 2:
            raise ValueError("Need at least input and output layer sizes")

        # Default activations: tanh for hidden, linear for output
        if activations is None:
            activations = ['tanh'] * (len(layer_sizes) - 2) + ['linear']

        if len(activations) != len(layer_sizes) - 1:
            raise ValueError(f"Expected {len(layer_sizes) - 1} activations, got {len(activations)}")

        layers = []
        for i in range(len(layer_sizes) - 1):
            layer_seed = None if seed is None else seed + i
            layer = NeuralOperations.alloc_layer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                activation=activations[i],
                init_method=init_method,
                seed=layer_seed
            )
            layers.append(layer)

        return MLP(
            layers=layers,
            input_size=layer_sizes[0],
            output_size=layer_sizes[-1]
        )

    @staticmethod
    @operator(
        domain="neural",
        category=OpCategory.QUERY,
        signature="(network: MLP) -> ndarray",
        deterministic=True,
        doc="Extract all network parameters into a flat vector"
    )
    def get_parameters(network: MLP) -> np.ndarray:
        """
        Layer 3: Extract all network parameters into a flat vector.

        Useful for genetic algorithms that operate on parameter vectors.

        Args:
            network: MLP network

        Returns:
            Flat parameter vector [total_params,]
        """
        params = []
        for layer in network.layers:
            params.append(layer.weights.ravel())
            params.append(layer.biases.ravel())
        return np.concatenate(params)

    @staticmethod
    @operator(
        domain="neural",
        category=OpCategory.TRANSFORM,
        signature="(network: MLP, params: ndarray) -> MLP",
        deterministic=True,
        doc="Set network parameters from a flat vector"
    )
    def set_parameters(network: MLP, params: np.ndarray) -> MLP:
        """
        Layer 3: Set network parameters from a flat vector.

        Creates a new network with updated parameters (immutable semantics).

        Args:
            network: Template MLP network (defines architecture)
            params: Flat parameter vector

        Returns:
            New MLP with updated parameters
        """
        new_network = network.copy()
        idx = 0

        for layer in new_network.layers:
            # Extract weights
            weight_size = layer.weights.size
            layer.weights = params[idx:idx + weight_size].reshape(layer.weights.shape)
            idx += weight_size

            # Extract biases
            bias_size = layer.biases.size
            layer.biases = params[idx:idx + bias_size]
            idx += bias_size

        return new_network

    @staticmethod
    @operator(
        domain="neural",
        category=OpCategory.TRANSFORM,
        signature="(params: ndarray, mutation_rate: float, mutation_scale: float, seed: Optional[int]) -> ndarray",
        deterministic=False,
        doc="Mutate parameters with Gaussian noise"
    )
    def mutate_parameters(params: np.ndarray, mutation_rate: float = 0.1,
                         mutation_scale: float = 0.3,
                         seed: Optional[int] = None) -> np.ndarray:
        """
        Layer 3: Mutate parameters with Gaussian noise.

        Args:
            params: Parameter vector
            mutation_rate: Probability of mutating each parameter
            mutation_scale: Standard deviation of Gaussian noise
            seed: Random seed for deterministic mutation

        Returns:
            Mutated parameter vector
        """
        if seed is not None:
            np.random.seed(seed)

        new_params = params.copy()
        mutation_mask = np.random.rand(len(params)) < mutation_rate
        new_params[mutation_mask] += np.random.randn(np.sum(mutation_mask)) * mutation_scale
        return new_params

    @staticmethod
    @operator(
        domain="neural",
        category=OpCategory.TRANSFORM,
        signature="(params1: ndarray, params2: ndarray, method: str, seed: Optional[int]) -> Tuple[ndarray, ndarray]",
        deterministic=False,
        doc="Crossover two parameter vectors"
    )
    def crossover_parameters(params1: np.ndarray, params2: np.ndarray,
                           method: str = 'uniform',
                           seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Layer 3: Crossover two parameter vectors.

        Methods:
        - 'uniform': Each parameter randomly chosen from parent1 or parent2
        - 'single_point': Split at random point and swap
        - 'blend': Weighted average (alpha=0.5)

        Args:
            params1: First parent parameter vector
            params2: Second parent parameter vector
            method: Crossover method
            seed: Random seed for deterministic crossover

        Returns:
            (offspring1, offspring2) parameter vectors
        """
        if seed is not None:
            np.random.seed(seed)

        if method == 'uniform':
            mask = np.random.rand(len(params1)) < 0.5
            offspring1 = np.where(mask, params1, params2)
            offspring2 = np.where(mask, params2, params1)
        elif method == 'single_point':
            point = np.random.randint(0, len(params1))
            offspring1 = np.concatenate([params1[:point], params2[point:]])
            offspring2 = np.concatenate([params2[:point], params1[point:]])
        elif method == 'blend':
            alpha = 0.5
            offspring1 = alpha * params1 + (1 - alpha) * params2
            offspring2 = (1 - alpha) * params1 + alpha * params2
        else:
            raise ValueError(f"Unknown crossover method: {method}")

        return offspring1, offspring2

    # === LAYER 4: PRESETS ===

    @staticmethod
    @operator(
        domain="neural",
        category=OpCategory.CONSTRUCT,
        signature="(seed: Optional[int]) -> MLP",
        deterministic=True,
        doc="Preset MLP for Flappy Bird control"
    )
    def flappy_bird_controller(seed: Optional[int] = None) -> MLP:
        """
        Layer 4: Preset MLP for Flappy Bird control.

        Architecture: [4, 8, 1]
        - Input: [bird_y, bird_velocity, next_pipe_x, next_pipe_gap_y]
        - Hidden: 8 neurons with tanh activation
        - Output: flap probability (sigmoid)

        Args:
            seed: Random seed for initialization

        Returns:
            Initialized MLP controller
        """
        return NeuralOperations.alloc_mlp(
            layer_sizes=[4, 8, 1],
            activations=['tanh', 'sigmoid'],
            init_method='xavier',
            seed=seed
        )


# Module-level singleton for convenience
neural = NeuralOperations()


# === HELPER FUNCTIONS ===

def create_controller_from_params(params: np.ndarray, architecture: List[int] = [4, 8, 1]) -> MLP:
    """
    Helper: Create a controller network from parameter vector.

    Useful for genetic algorithm populations where each individual is a param vector.

    Args:
        params: Flat parameter vector
        architecture: Network layer sizes

    Returns:
        MLP with parameters set
    """
    template = neural.alloc_mlp(architecture, activations=['tanh', 'sigmoid'])
    return neural.set_parameters(template, params)


def flappy_bird_decision(network: MLP, observation: np.ndarray,
                        threshold: float = 0.5) -> bool:
    """
    Helper: Make binary flap decision from network output.

    Args:
        network: MLP controller
        observation: Sensor observation [4,]
        threshold: Flap if output > threshold

    Returns:
        True to flap, False otherwise
    """
    output = neural.forward(observation, network)
    return output[0] > threshold


def batch_flappy_bird_decisions(networks: List[MLP], observations: np.ndarray,
                                threshold: float = 0.5) -> np.ndarray:
    """
    Helper: Make batch decisions for multiple networks.

    Args:
        networks: List of MLP controllers [n_networks]
        observations: Batch of observations [n_networks, 4]
        threshold: Flap threshold

    Returns:
        Boolean array [n_networks,] indicating flap decisions
    """
    decisions = np.zeros(len(networks), dtype=bool)
    for i, network in enumerate(networks):
        output = neural.forward(observations[i:i+1], network)
        decisions[i] = output[0, 0] > threshold
    return decisions


# Export operators for domain registry discovery
alloc_layer = NeuralOperations.alloc_layer
alloc_mlp = NeuralOperations.alloc_mlp
forward = NeuralOperations.forward
linear = NeuralOperations.linear
relu = NeuralOperations.relu
leaky_relu = NeuralOperations.leaky_relu
sigmoid = NeuralOperations.sigmoid
tanh = NeuralOperations.tanh
softmax = NeuralOperations.softmax
apply_activation = NeuralOperations.apply_activation
dense = NeuralOperations.dense
get_parameters = NeuralOperations.get_parameters
set_parameters = NeuralOperations.set_parameters
mutate_parameters = NeuralOperations.mutate_parameters
crossover_parameters = NeuralOperations.crossover_parameters
flappy_bird_controller = NeuralOperations.flappy_bird_controller
