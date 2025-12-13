"""
Flappy Bird Game Domain
=======================

A simple 2D platformer physics domain for demonstrating Morphogen's multi-domain
capabilities through a classic Flappy Bird game with neural network controllers
and genetic algorithm training.

This domain provides:
- Bird physics (gravity, flapping, velocity)
- Pipe obstacles (movement, collision detection)
- Batch/vectorized operations for parallel agent training
- Deterministic simulation for reproducibility

Layer 1: Atomic operators (gravity, collision, spawn)
Layer 2: Composite operators (step, reset, batch_step)
Layer 3: Game constructs (full game loop, training episode)
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class Bird:
    """
    Represents a single bird character.

    Attributes:
        y: Vertical position [0..1] (normalized screen space)
        velocity: Vertical velocity (positive = upward)
        alive: Whether the bird is still alive
        score: Current score (frames survived + pipes passed)
        pipes_passed: Number of pipes successfully passed
    """
    y: float = 0.5
    velocity: float = 0.0
    alive: bool = True
    score: float = 0.0
    pipes_passed: int = 0

    def copy(self) -> 'Bird':
        """Return a copy of this bird (immutable semantics)"""
        return Bird(
            y=self.y,
            velocity=self.velocity,
            alive=self.alive,
            score=self.score,
            pipes_passed=self.pipes_passed
        )


@dataclass
class Pipe:
    """
    Represents a pipe obstacle.

    Attributes:
        x: Horizontal position [0..1+] (normalized screen space)
        gap_center_y: Vertical center of the gap [0..1]
        gap_size: Size of the gap to fly through
        width: Width of the pipe
    """
    x: float = 1.0
    gap_center_y: float = 0.5
    gap_size: float = 0.25
    width: float = 0.1


@dataclass
class GameState:
    """
    Complete game state for vectorized simulation.

    Supports batch processing of multiple birds at once for parallel training.
    """
    # Bird states (shape: [n_birds,])
    bird_y: np.ndarray  # Vertical positions
    bird_velocity: np.ndarray  # Vertical velocities
    bird_alive: np.ndarray  # Alive flags (bool)
    bird_score: np.ndarray  # Scores
    bird_pipes_passed: np.ndarray  # Pipes passed count

    # Pipe states (shape: [n_pipes,])
    pipe_x: np.ndarray  # Horizontal positions
    pipe_gap_y: np.ndarray  # Gap centers
    pipe_gap_size: np.ndarray  # Gap sizes

    # Game config
    n_birds: int
    n_pipes: int

    def copy(self) -> 'GameState':
        """Return a copy of this game state (immutable semantics)"""
        return GameState(
            bird_y=self.bird_y.copy(),
            bird_velocity=self.bird_velocity.copy(),
            bird_alive=self.bird_alive.copy(),
            bird_score=self.bird_score.copy(),
            bird_pipes_passed=self.bird_pipes_passed.copy(),
            pipe_x=self.pipe_x.copy(),
            pipe_gap_y=self.pipe_gap_y.copy(),
            pipe_gap_size=self.pipe_gap_size.copy(),
            n_birds=self.n_birds,
            n_pipes=self.n_pipes
        )


class FlappyOperations:
    """
    Namespace for Flappy Bird game operations.

    Follows Morphogen's 4-layer operator hierarchy:
    - Layer 1: Atomic (gravity, flap, collision)
    - Layer 2: Composite (step, reset)
    - Layer 3: Constructs (game loop, episode)
    - Layer 4: Presets (full training environment)
    """

    # === LAYER 1: ATOMIC OPERATORS ===

    @staticmethod
    def alloc_bird(y: float = 0.5, velocity: float = 0.0) -> Bird:
        """
        Layer 1: Allocate a single bird.

        Args:
            y: Initial vertical position [0..1]
            velocity: Initial vertical velocity

        Returns:
            New Bird instance
        """
        return Bird(y=y, velocity=velocity, alive=True, score=0.0, pipes_passed=0)

    @staticmethod
    def alloc_pipe(x: float = 1.0, gap_center_y: float = 0.5,
                   gap_size: float = 0.25, width: float = 0.1) -> Pipe:
        """
        Layer 1: Allocate a single pipe obstacle.

        Args:
            x: Horizontal position [0..1+]
            gap_center_y: Vertical center of gap [0..1]
            gap_size: Size of the gap
            width: Width of the pipe

        Returns:
            New Pipe instance
        """
        return Pipe(x=x, gap_center_y=gap_center_y, gap_size=gap_size, width=width)

    @staticmethod
    def alloc_game(n_birds: int = 1, n_pipes: int = 3,
                   seed: Optional[int] = None) -> GameState:
        """
        Layer 1: Allocate a batch game state for parallel simulation.

        Args:
            n_birds: Number of birds to simulate in parallel
            n_pipes: Number of pipes in the game
            seed: Random seed for deterministic initialization

        Returns:
            New GameState with randomized pipes
        """
        if seed is not None:
            np.random.seed(seed)

        # Initialize birds (all start at same position)
        bird_y = np.full(n_birds, 0.5, dtype=np.float32)
        bird_velocity = np.zeros(n_birds, dtype=np.float32)
        bird_alive = np.ones(n_birds, dtype=bool)
        bird_score = np.zeros(n_birds, dtype=np.float32)
        bird_pipes_passed = np.zeros(n_birds, dtype=np.int32)

        # Initialize pipes (randomized gaps, evenly spaced)
        pipe_spacing = 0.4
        pipe_x = np.array([1.0 + i * pipe_spacing for i in range(n_pipes)], dtype=np.float32)
        pipe_gap_y = np.random.uniform(0.3, 0.7, n_pipes).astype(np.float32)
        pipe_gap_size = np.full(n_pipes, 0.25, dtype=np.float32)

        return GameState(
            bird_y=bird_y,
            bird_velocity=bird_velocity,
            bird_alive=bird_alive,
            bird_score=bird_score,
            bird_pipes_passed=bird_pipes_passed,
            pipe_x=pipe_x,
            pipe_gap_y=pipe_gap_y,
            pipe_gap_size=pipe_gap_size,
            n_birds=n_birds,
            n_pipes=n_pipes
        )

    @staticmethod
    def apply_gravity(state: GameState, gravity: float = 1.5,
                      dt: float = 0.016) -> GameState:
        """
        Layer 1: Apply gravity to all birds.

        Args:
            state: Current game state
            gravity: Gravity strength (positive = downward)
            dt: Time step

        Returns:
            Updated game state with gravity applied
        """
        new_state = state.copy()
        # Only apply to alive birds
        mask = new_state.bird_alive
        new_state.bird_velocity[mask] -= gravity * dt
        new_state.bird_y[mask] += new_state.bird_velocity[mask] * dt
        return new_state

    @staticmethod
    def flap(state: GameState, flap_mask: np.ndarray,
             flap_strength: float = 0.35) -> GameState:
        """
        Layer 1: Apply flap impulse to birds based on mask.

        Args:
            state: Current game state
            flap_mask: Boolean array indicating which birds should flap
            flap_strength: Upward velocity impulse

        Returns:
            Updated game state with flaps applied
        """
        new_state = state.copy()
        # Only flap alive birds
        mask = new_state.bird_alive & flap_mask
        new_state.bird_velocity[mask] = flap_strength
        return new_state

    @staticmethod
    def move_pipes(state: GameState, speed: float = 0.5,
                   dt: float = 0.016) -> GameState:
        """
        Layer 1: Move all pipes horizontally.

        When a pipe goes off-screen (x < -0.1), it wraps around to the right
        with a new randomized gap position.

        Args:
            state: Current game state
            speed: Horizontal speed (normalized units per second)
            dt: Time step

        Returns:
            Updated game state with pipes moved
        """
        new_state = state.copy()
        new_state.pipe_x -= speed * dt

        # Wrap pipes that go off-screen
        wrap_mask = new_state.pipe_x < -0.1
        if np.any(wrap_mask):
            # Find rightmost pipe position
            max_x = np.max(new_state.pipe_x)
            # If max_x is negative (all pipes off-screen), wrap to right edge
            # Otherwise wrap to max_x + spacing
            wrap_position = max(1.2, max_x + 0.4)
            new_state.pipe_x[wrap_mask] = wrap_position
            # Randomize gap positions for wrapped pipes
            new_state.pipe_gap_y[wrap_mask] = np.random.uniform(0.3, 0.7, np.sum(wrap_mask)).astype(np.float32)

        return new_state

    @staticmethod
    def check_collisions(state: GameState, bird_size: float = 0.03) -> GameState:
        """
        Layer 1: Check collisions between birds and pipes/boundaries.

        Collision detection:
        - Pipes: AABB collision with gap regions
        - Boundaries: Top (y > 1.0) and bottom (y < 0.0) of screen

        Args:
            state: Current game state
            bird_size: Radius of bird collision box

        Returns:
            Updated game state with alive flags updated
        """
        new_state = state.copy()

        # Only check alive birds
        alive_mask = new_state.bird_alive
        if not np.any(alive_mask):
            return new_state

        # Boundary collisions
        boundary_collision = (new_state.bird_y > 1.0 - bird_size) | (new_state.bird_y < bird_size)
        new_state.bird_alive[alive_mask] &= ~boundary_collision[alive_mask]

        # Pipe collisions (check each pipe against all birds)
        bird_x = 0.2  # Birds are at fixed x position
        for i in range(state.n_pipes):
            pipe_left = state.pipe_x[i]
            pipe_right = pipe_left + 0.1  # pipe width

            # Check if bird is horizontally within pipe
            in_pipe_x = (bird_x + bird_size > pipe_left) & (bird_x - bird_size < pipe_right)

            if not np.any(in_pipe_x & alive_mask):
                continue

            # Check if bird is vertically outside gap
            gap_top = state.pipe_gap_y[i] + state.pipe_gap_size[i] / 2
            gap_bottom = state.pipe_gap_y[i] - state.pipe_gap_size[i] / 2
            outside_gap = (new_state.bird_y < gap_bottom - bird_size) | \
                         (new_state.bird_y > gap_top + bird_size)

            # Collision = in pipe horizontally AND outside gap vertically
            collision = in_pipe_x & outside_gap
            new_state.bird_alive[alive_mask] &= ~collision[alive_mask]

        return new_state

    @staticmethod
    def update_scores(state: GameState, frame_reward: float = 1.0,
                      pipe_reward: float = 10.0) -> GameState:
        """
        Layer 1: Update scores and track pipe passage.

        Rewards:
        - frame_reward: Points per frame survived
        - pipe_reward: Bonus points for passing a pipe

        Args:
            state: Current game state
            frame_reward: Reward per frame alive
            pipe_reward: Reward per pipe passed

        Returns:
            Updated game state with scores updated
        """
        new_state = state.copy()

        # Alive birds get frame reward
        new_state.bird_score[new_state.bird_alive] += frame_reward

        # Check pipe passage (bird passes pipe center)
        bird_x = 0.2
        for i in range(state.n_pipes):
            # Check if pipe center just passed bird
            pipe_center = state.pipe_x[i] + 0.05  # pipe width / 2
            just_passed = (pipe_center < bird_x) and (pipe_center + 0.01 >= bird_x)

            if just_passed:
                # All alive birds get pipe bonus
                new_state.bird_score[new_state.bird_alive] += pipe_reward
                new_state.bird_pipes_passed[new_state.bird_alive] += 1

        return new_state

    @staticmethod
    def extract_sensors(state: GameState, bird_idx: int = 0) -> np.ndarray:
        """
        Layer 1: Extract sensor observations for a single bird.

        Returns 4D observation vector:
        [bird_y, bird_velocity, next_pipe_x, next_pipe_gap_y]

        Args:
            state: Current game state
            bird_idx: Index of bird to extract sensors for

        Returns:
            Observation vector [4,] (normalized to ~[-1, 1])
        """
        # Find closest pipe ahead of bird
        bird_x = 0.2
        pipes_ahead = state.pipe_x > bird_x - 0.05

        if np.any(pipes_ahead):
            closest_idx = np.argmin(np.where(pipes_ahead, state.pipe_x, 999.0))
            next_pipe_x = state.pipe_x[closest_idx]
            next_pipe_gap_y = state.pipe_gap_y[closest_idx]
        else:
            # No pipes ahead, use default
            next_pipe_x = 1.0
            next_pipe_gap_y = 0.5

        # Normalize observations
        obs = np.array([
            state.bird_y[bird_idx] * 2 - 1,  # [-1, 1]
            state.bird_velocity[bird_idx] * 2,  # ~[-2, 2]
            (next_pipe_x - bird_x) * 2,  # ~[0, 2]
            next_pipe_gap_y * 2 - 1  # [-1, 1]
        ], dtype=np.float32)

        return obs

    @staticmethod
    def extract_sensors_batch(state: GameState) -> np.ndarray:
        """
        Layer 1: Extract sensor observations for all birds.

        Args:
            state: Current game state

        Returns:
            Batch of observations [n_birds, 4]
        """
        observations = np.zeros((state.n_birds, 4), dtype=np.float32)
        for i in range(state.n_birds):
            observations[i] = FlappyOperations.extract_sensors(state, bird_idx=i)
        return observations

    # === LAYER 2: COMPOSITE OPERATORS ===

    @staticmethod
    def step(state: GameState, actions: np.ndarray,
             gravity: float = 1.5, flap_strength: float = 0.35,
             pipe_speed: float = 0.5, dt: float = 0.016,
             frame_reward: float = 1.0, pipe_reward: float = 10.0) -> GameState:
        """
        Layer 2: Execute one complete game step for all birds.

        Combines: gravity → flap → move_pipes → collisions → scores

        Args:
            state: Current game state
            actions: Boolean array [n_birds,] indicating flap actions
            gravity: Gravity strength
            flap_strength: Flap impulse strength
            pipe_speed: Pipe movement speed
            dt: Time step
            frame_reward: Reward per frame alive
            pipe_reward: Reward per pipe passed

        Returns:
            Updated game state after one step
        """
        # Physics
        state = FlappyOperations.apply_gravity(state, gravity=gravity, dt=dt)
        state = FlappyOperations.flap(state, flap_mask=actions, flap_strength=flap_strength)
        state = FlappyOperations.move_pipes(state, speed=pipe_speed, dt=dt)

        # Collision detection
        state = FlappyOperations.check_collisions(state)

        # Scoring
        state = FlappyOperations.update_scores(state, frame_reward=frame_reward,
                                               pipe_reward=pipe_reward)

        return state

    @staticmethod
    def reset(state: GameState, seed: Optional[int] = None) -> GameState:
        """
        Layer 2: Reset game to initial state.

        Preserves n_birds and n_pipes but resets all game state.

        Args:
            state: Current game state
            seed: Optional random seed for deterministic reset

        Returns:
            Fresh game state
        """
        return FlappyOperations.alloc_game(
            n_birds=state.n_birds,
            n_pipes=state.n_pipes,
            seed=seed
        )

    # === LAYER 3: GAME CONSTRUCTS ===

    @staticmethod
    def run_episode(state: GameState, controller, max_steps: int = 1000,
                    **step_kwargs) -> Tuple[GameState, np.ndarray]:
        """
        Layer 3: Run a complete game episode until all birds die or max steps.

        Args:
            state: Initial game state
            controller: Callable that takes (state) -> actions [n_birds,]
            max_steps: Maximum number of steps before termination
            **step_kwargs: Additional arguments passed to step()

        Returns:
            (final_state, trajectory) where trajectory is [steps, n_birds, 4] observations
        """
        trajectory = []

        for step_idx in range(max_steps):
            # Extract observations
            obs = FlappyOperations.extract_sensors_batch(state)
            trajectory.append(obs)

            # Get actions from controller
            actions = controller(state)

            # Step simulation
            state = FlappyOperations.step(state, actions, **step_kwargs)

            # Check termination
            if not np.any(state.bird_alive):
                break

        trajectory = np.array(trajectory)
        return state, trajectory


# Module-level singleton for convenience
flappy = FlappyOperations()


# === HELPER FUNCTIONS ===

def random_controller(state: GameState, flap_prob: float = 0.05) -> np.ndarray:
    """
    Simple random baseline controller for testing.

    Args:
        state: Current game state
        flap_prob: Probability of flapping each frame

    Returns:
        Random actions [n_birds,]
    """
    return np.random.rand(state.n_birds) < flap_prob
