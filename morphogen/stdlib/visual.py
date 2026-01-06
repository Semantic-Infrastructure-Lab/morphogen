"""Visual operations implementation.

This module provides visualization capabilities including field colorization,
PNG output, and interactive real-time display for the MVP.
"""

from typing import Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
import numpy as np

from morphogen.core.operator import operator, OpCategory
import time


class Visual:
    """Opaque visual representation (linear RGB).

    Stores rendered image data ready for output.
    """

    def __init__(self, data: np.ndarray):
        """Initialize visual.

        Args:
            data: RGB image data (shape: (height, width, 3), dtype: float32, range: [0, 1])
        """
        if len(data.shape) != 3 or data.shape[2] != 3:
            raise ValueError(f"Visual data must be (height, width, 3), got {data.shape}")

        self.data = np.clip(data, 0.0, 1.0).astype(np.float32)
        self.shape = data.shape[:2]

    @property
    def height(self) -> int:
        """Get image height."""
        return self.shape[0]

    @property
    def width(self) -> int:
        """Get image width."""
        return self.shape[1]

    def copy(self) -> 'Visual':
        """Create a copy of this visual."""
        return Visual(self.data.copy())

    def __repr__(self) -> str:
        return f"Visual(shape={self.shape})"


# =============================================================================
# CONFIG DATACLASSES
# =============================================================================

@dataclass
class AgentRenderConfig:
    """Configuration for agent visualization."""
    width: int = 512
    height: int = 512
    position_property: str = 'pos'
    color_property: Optional[str] = None
    size_property: Optional[str] = None
    alpha_property: Optional[str] = None
    rotation_property: Optional[str] = None
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    size: float = 2.0
    alpha: float = 1.0
    palette: str = "viridis"
    background: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    blend_mode: str = "alpha"
    trail: bool = False
    trail_length: int = 10
    trail_alpha: float = 0.5


@dataclass
class PhaseSpaceConfig:
    """Configuration for phase space visualization."""
    position_property: str = 'pos'
    velocity_property: str = 'vel'
    width: int = 512
    height: int = 512
    color_property: Optional[str] = None
    palette: str = "viridis"
    point_size: float = 2.0
    alpha: float = 0.6
    show_trajectories: bool = False
    background: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class GraphConfig:
    """Configuration for graph visualization."""
    width: int = 800
    height: int = 800
    node_size: float = 8.0
    node_color: Tuple[float, float, float] = (0.3, 0.6, 1.0)
    edge_color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    edge_width: float = 1.0
    layout: str = "force"
    iterations: int = 50
    color_by_centrality: bool = False
    palette: str = "viridis"
    show_labels: bool = False
    background: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class MetricsConfig:
    """Configuration for metrics overlay."""
    position: str = "top-left"
    font_size: int = 14
    text_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bg_alpha: float = 0.7


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _compute_bounds(positions: np.ndarray, padding: float = 0.1) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute bounds for positions with padding."""
    xmin, xmax = np.min(positions[:, 0]), np.max(positions[:, 0])
    ymin, ymax = np.min(positions[:, 1]), np.max(positions[:, 1])
    x_padding = (xmax - xmin) * padding
    y_padding = (ymax - ymin) * padding
    return ((xmin - x_padding, xmax + x_padding), (ymin - y_padding, ymax + y_padding))


def _positions_to_pixels(positions: np.ndarray, bounds: Tuple, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """Map positions to pixel coordinates."""
    (xmin, xmax), (ymin, ymax) = bounds
    x_range = xmax - xmin
    y_range = ymax - ymin

    if x_range == 0.0:
        x_norm = np.full(len(positions), 0.5)
    else:
        x_norm = (positions[:, 0] - xmin) / x_range

    if y_range == 0.0:
        y_norm = np.full(len(positions), 0.5)
    else:
        y_norm = (positions[:, 1] - ymin) / y_range

    px = np.clip((x_norm * (width - 1)).astype(int), 0, width - 1)
    py = np.clip(((1.0 - y_norm) * (height - 1)).astype(int), 0, height - 1)
    return px, py


def _map_property_to_colors(values: np.ndarray, palette_colors: np.ndarray) -> np.ndarray:
    """Map property values to colors using palette interpolation."""
    # Handle vector properties (use magnitude)
    if len(values.shape) > 1:
        values = np.linalg.norm(values, axis=1)

    # Normalize to [0, 1]
    vmin, vmax = np.min(values), np.max(values)
    if vmax - vmin < 1e-10:
        color_norm = np.zeros_like(values)
    else:
        color_norm = (values - vmin) / (vmax - vmin)

    # Interpolate colors
    n_colors = len(palette_colors)
    indices = color_norm * (n_colors - 1)
    idx_low = np.floor(indices).astype(int)
    idx_high = np.minimum(idx_low + 1, n_colors - 1)
    frac = (indices - idx_low)[:, np.newaxis]

    return palette_colors[idx_low] * (1 - frac) + palette_colors[idx_high] * frac


def _map_property_to_sizes(values: np.ndarray, base_size: float) -> np.ndarray:
    """Map property values to sizes with normalization."""
    if len(values.shape) > 1:
        values = np.linalg.norm(values, axis=1)

    vmin, vmax = np.min(values), np.max(values)
    if vmax - vmin < 1e-10:
        return np.ones(len(values)) * base_size
    else:
        size_norm = (values - vmin) / (vmax - vmin)
        return size_norm * base_size * 2


def _blend_pixel(img: np.ndarray, y: int, x: int, color: np.ndarray, alpha: float, mode: str) -> None:
    """Blend a single pixel with the image."""
    if mode == "alpha":
        img[y, x] = img[y, x] * (1 - alpha) + color * alpha
    else:  # additive
        img[y, x] = np.clip(img[y, x] + color * alpha, 0.0, 1.0)


def _render_filled_circle(img: np.ndarray, cx: int, cy: int, radius: int,
                          color: np.ndarray, alpha: float, blend_mode: str) -> None:
    """Render a filled circle with alpha blending."""
    height, width = img.shape[:2]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < height and 0 <= nx < width:
                    _blend_pixel(img, ny, nx, color, alpha, blend_mode)


def _render_rotation_indicator(img: np.ndarray, cx: int, cy: int, angle: float,
                               length: int, color: np.ndarray, alpha: float, blend_mode: str) -> None:
    """Render a rotation indicator line from center."""
    height, width = img.shape[:2]
    dx_line = int(length * np.cos(angle))
    dy_line = int(length * np.sin(angle))
    indicator_color = np.minimum(color * 1.5, 1.0)

    steps = max(abs(dx_line), abs(dy_line)) + 1
    for s in range(steps):
        t = s / max(steps - 1, 1)
        lx = cx + int(t * dx_line)
        ly = cy - int(t * dy_line)
        if 0 <= ly < height and 0 <= lx < width:
            _blend_pixel(img, ly, lx, indicator_color, alpha, blend_mode)


def _render_agent_trails(img: np.ndarray, trail_history: np.ndarray, colors: np.ndarray,
                         alphas: np.ndarray, bounds: Tuple, width: int, height: int,
                         trail_alpha: float, blend_mode: str) -> None:
    """Render trails for all agents."""
    (xmin, xmax), (ymin, ymax) = bounds
    x_range = xmax - xmin if xmax - xmin > 0 else 1.0
    y_range = ymax - ymin if ymax - ymin > 0 else 1.0

    for i in range(len(trail_history)):
        history = trail_history[i]
        agent_color = colors[i]

        for t in range(len(history)):
            if np.any(np.isnan(history[t])):
                continue

            x_norm = (history[t, 0] - xmin) / x_range
            y_norm = (history[t, 1] - ymin) / y_range
            tx = int(np.clip(x_norm * (width - 1), 0, width - 1))
            ty = int(np.clip((1.0 - y_norm) * (height - 1), 0, height - 1))

            trail_t_alpha = (t / len(history)) * trail_alpha * alphas[i]
            _blend_pixel(img, ty, tx, agent_color, trail_t_alpha, blend_mode)


def _draw_background_box(img: np.ndarray, x_start: int, y_start: int,
                         box_width: int, box_height: int,
                         bg_color: Tuple, bg_alpha: float) -> None:
    """Draw a semi-transparent background box on image."""
    height, width = img.shape[:2]
    x_end = min(x_start + box_width, width)
    y_end = min(y_start + box_height, height)
    bg_arr = np.array(bg_color)
    for y in range(max(0, y_start), y_end):
        for x in range(max(0, x_start), x_end):
            img[y, x] = img[y, x] * (1 - bg_alpha) + bg_arr * bg_alpha


def _compute_text_position(position: str, width: int, height: int,
                           text_width: int, text_height: int) -> Tuple[int, int]:
    """Compute starting position for text based on position string."""
    positions = {
        "top-left": (10, 10),
        "top-right": (width - text_width - 10, 10),
        "bottom-left": (10, height - text_height - 10),
        "bottom-right": (width - text_width - 10, height - text_height - 10),
    }
    if position not in positions:
        raise ValueError(f"Unknown position: {position}. Supported: {list(positions.keys())}")
    return positions[position]


def _get_edge_weight(adjacency_list: dict, i: int, j: int) -> float:
    """Get edge weight between two nodes from adjacency list."""
    for neighbor, weight in adjacency_list.get(i, []):
        if neighbor == j:
            return weight
    return 0.0


class VisualOperations:
    """Namespace for visual operations (accessed as 'visual' in DSL)."""

    # Color palettes (linear RGB values)
    PALETTES = {
        "grayscale": [
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
        ],
        "fire": [
            (0.0, 0.0, 0.0),      # Black
            (0.5, 0.0, 0.0),      # Dark red
            (1.0, 0.0, 0.0),      # Red
            (1.0, 0.5, 0.0),      # Orange
            (1.0, 1.0, 0.0),      # Yellow
            (1.0, 1.0, 1.0),      # White
        ],
        "viridis": [
            (0.267, 0.005, 0.329),  # Dark purple
            (0.283, 0.141, 0.458),  # Purple
            (0.254, 0.266, 0.530),  # Blue-purple
            (0.207, 0.372, 0.554),  # Blue
            (0.164, 0.471, 0.558),  # Cyan-blue
            (0.135, 0.568, 0.551),  # Cyan
            (0.196, 0.664, 0.523),  # Green-cyan
            (0.395, 0.762, 0.420),  # Green
            (0.671, 0.867, 0.253),  # Yellow-green
            (0.993, 0.906, 0.144),  # Yellow
        ],
        "coolwarm": [
            (0.23, 0.30, 0.75),     # Cool blue
            (0.57, 0.77, 0.87),     # Light blue
            (0.87, 0.87, 0.87),     # White
            (0.96, 0.68, 0.52),     # Light orange
            (0.71, 0.02, 0.15),     # Warm red
        ],
    }

    @staticmethod
    @operator(
        domain="visual",
        category=OpCategory.TRANSFORM,
        signature="(field: Field2D, palette: str, vmin: Optional[float], vmax: Optional[float]) -> Visual",
        deterministic=True,
        doc="Map scalar field to colors using a palette"
    )
    def colorize(field, palette: str = "grayscale",
                 vmin: Optional[float] = None,
                 vmax: Optional[float] = None) -> Visual:
        """Map scalar field to colors using a palette.

        Args:
            field: Field2D to colorize
            palette: Palette name ("grayscale", "fire", "viridis", "coolwarm")
            vmin: Minimum value for mapping (default: field min)
            vmax: Maximum value for mapping (default: field max)

        Returns:
            Visual representation of the field
        """
        from .field import Field2D

        if not isinstance(field, Field2D):
            raise TypeError(f"Expected Field2D, got {type(field)}")

        # Get field data
        data = field.data

        # Handle multi-channel fields (use magnitude)
        if len(data.shape) == 3:
            data = np.linalg.norm(data, axis=2)

        # Normalize to [0, 1]
        if vmin is None:
            vmin = np.min(data)
        if vmax is None:
            vmax = np.max(data)

        # Avoid division by zero
        if vmax - vmin < 1e-10:
            normalized = np.zeros_like(data)
        else:
            normalized = (data - vmin) / (vmax - vmin)

        normalized = np.clip(normalized, 0.0, 1.0)

        # Get palette colors
        if palette not in VisualOperations.PALETTES:
            raise ValueError(f"Unknown palette: {palette}. Available: {list(VisualOperations.PALETTES.keys())}")

        palette_colors = np.array(VisualOperations.PALETTES[palette])
        n_colors = len(palette_colors)

        # Map normalized values to palette indices
        indices = normalized * (n_colors - 1)
        idx_low = np.floor(indices).astype(int)
        idx_high = np.minimum(idx_low + 1, n_colors - 1)
        frac = indices - idx_low

        # Interpolate between palette colors
        h, w = normalized.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)

        for c in range(3):
            rgb[:, :, c] = (
                palette_colors[idx_low, c] * (1 - frac) +
                palette_colors[idx_high, c] * frac
            )

        return Visual(rgb)

    @staticmethod
    @operator(
        domain="visual",
        category=OpCategory.TRANSFORM,
        signature="(visual: Visual, path: str, format: str) -> None",
        deterministic=True,
        doc="Save visual to file"
    )
    def output(visual: Visual, path: str, format: str = "auto") -> None:
        """Save visual to file.

        Args:
            visual: Visual to save
            path: Output file path
            format: Output format ("auto", "png", "jpg") - auto infers from extension

        Raises:
            ImportError: If PIL/Pillow is not installed
        """
        if not isinstance(visual, Visual):
            raise TypeError(f"Expected Visual, got {type(visual)}")

        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "PIL/Pillow is required for visual output. "
                "Install with: pip install Pillow"
            )

        # Infer format from path if auto
        if format == "auto":
            if path.endswith(".png"):
                format = "png"
            elif path.endswith(".jpg") or path.endswith(".jpeg"):
                format = "jpeg"
            else:
                format = "png"  # Default

        # Normalize format for PIL
        format_map = {
            "jpg": "JPEG",
            "jpeg": "JPEG",
            "png": "PNG"
        }
        pil_format = format_map.get(format.lower(), "PNG")

        # Convert linear RGB to sRGB (gamma correction)
        srgb = VisualOperations._linear_to_srgb(visual.data)

        # Convert to 8-bit
        rgb_8bit = (srgb * 255).astype(np.uint8)

        # Save image (Pillow auto-detects RGB from uint8 array shape)
        img = Image.fromarray(rgb_8bit)
        img.save(path, pil_format)

        print(f"Saved visual to: {path}")

    @staticmethod
    def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
        """Convert linear RGB to sRGB with gamma correction.

        Args:
            linear: Linear RGB values in [0, 1]

        Returns:
            sRGB values in [0, 1]
        """
        # sRGB gamma correction
        srgb = np.where(
            linear <= 0.0031308,
            linear * 12.92,
            1.055 * np.power(linear, 1.0 / 2.4) - 0.055
        )
        return np.clip(srgb, 0.0, 1.0)

    @staticmethod
    def _render_status_overlay(screen, font, actual_fps, current_fps, total_frames, paused):
        """Render status text overlay on pygame screen."""
        status_lines = [
            f"FPS: {actual_fps:.1f} / {current_fps}",
            f"Frame: {total_frames}" if paused else "",
            "PAUSED" if paused else "RUNNING",
            "", "Controls:", "SPACE: Pause/Resume",
            "→: Step (paused)", "↑↓: Speed", "Q: Quit"
        ]
        y_offset = 10
        for line in status_lines:
            if not line:
                y_offset += 22
                continue
            text = font.render(line, True, (255, 255, 255))
            import pygame
            text_bg = pygame.Surface((text.get_width() + 10, text.get_height() + 4))
            text_bg.set_alpha(180)
            text_bg.fill((0, 0, 0))
            screen.blit(text_bg, (5, y_offset))
            screen.blit(text, (10, y_offset + 2))
            y_offset += 22

    @staticmethod
    @operator(
        domain="visual",
        category=OpCategory.TRANSFORM,
        signature="(frame_generator: Callable, title: str, target_fps: int, scale: int) -> None",
        deterministic=True,
        doc="Display simulation in real-time interactive window"
    )
    def display(frame_generator: Callable[[], Optional[Visual]],
                title: str = "Creative Computation DSL",
                target_fps: int = 30,
                scale: int = 2) -> None:
        """Display simulation in real-time interactive window.

        Args:
            frame_generator: Callable that generates frames. Should return Visual or None to quit.
            title: Window title
            target_fps: Target frames per second
            scale: Scale factor for display (1 = native resolution)

        Controls: SPACE (pause), RIGHT (step), UP/DOWN (speed), Q/ESC (quit)
        """
        if not callable(frame_generator):
            raise TypeError(f"frame_generator must be callable, got {type(frame_generator)}")
        if not isinstance(target_fps, int) or target_fps <= 0:
            raise ValueError(f"target_fps must be positive integer, got {target_fps}")
        if not isinstance(scale, int) or scale <= 0:
            raise ValueError(f"scale must be positive integer, got {scale}")

        try:
            import pygame
        except ImportError:
            raise ImportError("pygame is required for interactive display. Install with: pip install pygame")

        pygame.init()
        first_frame = frame_generator()
        if first_frame is None:
            return
        if not isinstance(first_frame, Visual):
            raise TypeError(f"frame_generator must return Visual, got {type(first_frame)}")

        width, height = first_frame.width * scale, first_frame.height * scale
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        clock, font = pygame.time.Clock(), pygame.font.Font(None, 24)

        # State variables
        paused, current_fps = False, target_fps
        fps_frame_count, total_frames = 0, 0
        fps_timer, actual_fps = time.time(), 0.0
        current_visual = first_frame

        try:
            running = True
            while running:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            paused = not paused
                        elif event.key == pygame.K_RIGHT and paused:
                            new_frame = frame_generator()
                            if new_frame is not None:
                                current_visual, total_frames, fps_frame_count = new_frame, total_frames + 1, fps_frame_count + 1
                        elif event.key == pygame.K_UP:
                            current_fps = min(current_fps + 5, 120)
                        elif event.key == pygame.K_DOWN:
                            current_fps = max(current_fps - 5, 1)
                        elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                            running = False

                # Generate next frame (if not paused)
                if not paused:
                    new_frame = frame_generator()
                    if new_frame is None:
                        running = False
                        continue
                    current_visual, total_frames, fps_frame_count = new_frame, total_frames + 1, fps_frame_count + 1

                # Render frame
                srgb = VisualOperations._linear_to_srgb(current_visual.data)
                rgb_8bit = (srgb * 255).astype(np.uint8)
                surf = pygame.surfarray.make_surface(np.transpose(rgb_8bit, (1, 0, 2)))
                if scale != 1:
                    surf = pygame.transform.scale(surf, (width, height))
                screen.blit(surf, (0, 0))

                # Update FPS counter
                now = time.time()
                if now - fps_timer >= 1.0:
                    actual_fps, fps_frame_count, fps_timer = fps_frame_count / (now - fps_timer), 0, now

                # Draw status overlay
                VisualOperations._render_status_overlay(screen, font, actual_fps, current_fps, total_frames, paused)
                pygame.display.flip()
                clock.tick(current_fps)
        finally:
            pygame.quit()

    # ========================================================================
    # VISUAL EXTENSIONS (v0.6.0)
    # ========================================================================

    @staticmethod
    @operator(
        domain="visual",
        category=OpCategory.TRANSFORM,
        signature="(agents: Agents, width: int, height: int, position_property: str, color_property: Optional[str], size_property: Optional[str], alpha_property: Optional[str], rotation_property: Optional[str], color: tuple, size: float, alpha: float, palette: str, background: tuple, bounds: Optional[Tuple], blend_mode: str, trail: bool, trail_length: int, trail_alpha: float) -> Visual",
        deterministic=True,
        doc="Render agents as points or circles with particle effects support"
    )
    def agents(agents, width: int = 512, height: int = 512,
               position_property: str = 'pos',
               color_property: Optional[str] = None,
               size_property: Optional[str] = None,
               alpha_property: Optional[str] = None,
               rotation_property: Optional[str] = None,
               color: tuple = (1.0, 1.0, 1.0),
               size: float = 2.0,
               alpha: float = 1.0,
               palette: str = "viridis",
               background: tuple = (0.0, 0.0, 0.0),
               bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
               blend_mode: str = "alpha",
               trail: bool = False,
               trail_length: int = 10,
               trail_alpha: float = 0.5) -> Visual:
        """Render agents as points or circles with particle effects support.

        Args:
            agents: Agents instance to visualize
            width: Output image width in pixels
            height: Output image height in pixels
            position_property: Name of position property (default: 'pos')
            color_property: Name of property to colorize by (optional)
            size_property: Name of property to size by (optional)
            alpha_property: Name of property for transparency (optional, 0-1)
            rotation_property: Name of property for rotation visualization (optional)
            color: Default color as (R, G, B) in [0, 1] (used if color_property=None)
            size: Default point size in pixels (used if size_property=None)
            alpha: Default alpha transparency in [0, 1] (used if alpha_property=None)
            palette: Color palette name for color_property mapping
            background: Background color as (R, G, B) in [0, 1]
            bounds: ((xmin, xmax), (ymin, ymax)) for position mapping, auto if None
            blend_mode: Blending mode ("alpha" or "additive")
            trail: If True, render agent trails (requires 'trail_history' property)
            trail_length: Number of trail points to render
            trail_alpha: Alpha transparency multiplier for trail

        Returns:
            Visual representation of agents
        """
        from .agents import Agents

        if not isinstance(agents, Agents):
            raise TypeError(f"Expected Agents, got {type(agents)}")

        # Get and validate positions
        positions = agents.get(position_property)
        if len(positions.shape) != 2 or positions.shape[1] != 2:
            raise ValueError(f"Position property must be (N, 2) array, got shape {positions.shape}")

        # Compute bounds and pixel coordinates
        if bounds is None:
            bounds = _compute_bounds(positions)
        px, py = _positions_to_pixels(positions, bounds, width, height)

        # Create output image
        img = np.zeros((height, width, 3), dtype=np.float32)
        img[:, :, :] = background

        # Compute colors, sizes, and alphas
        palette_colors = np.array(VisualOperations.PALETTES[palette])
        if color_property is not None:
            colors = _map_property_to_colors(agents.get(color_property), palette_colors)
        else:
            colors = np.tile(color, (len(positions), 1))

        if size_property is not None:
            sizes = _map_property_to_sizes(agents.get(size_property), size)
        else:
            sizes = np.ones(len(positions)) * size

        if alpha_property is not None:
            alpha_values = agents.get(alpha_property)
            if len(alpha_values.shape) > 1:
                alpha_values = np.linalg.norm(alpha_values, axis=1)
            alphas = np.clip(alpha_values, 0.0, 1.0)
        else:
            alphas = np.ones(len(positions)) * alpha

        # Get rotations if specified
        rotations = None
        if rotation_property is not None and rotation_property in agents.properties:
            rotation_values = agents.get(rotation_property)
            if len(rotation_values.shape) > 1 and rotation_values.shape[1] == 2:
                rotations = np.arctan2(rotation_values[:, 1], rotation_values[:, 0])
            else:
                rotations = rotation_values

        # Render trails
        if trail and 'trail_history' in agents.properties:
            trail_history = agents.get('trail_history')
            if len(trail_history.shape) == 3:
                _render_agent_trails(img, trail_history, colors, alphas, bounds,
                                     width, height, trail_alpha, blend_mode)

        # Render agents
        for i in range(len(positions)):
            x, y = px[i], py[i]
            agent_color = colors[i]
            agent_alpha = alphas[i]
            agent_size = int(sizes[i])

            if rotations is not None:
                _render_rotation_indicator(img, x, y, rotations[i], agent_size,
                                           agent_color, agent_alpha, blend_mode)

            _render_filled_circle(img, x, y, agent_size, agent_color, agent_alpha, blend_mode)

        return Visual(img)

    @staticmethod
    @operator(
        domain="visual",
        category=OpCategory.CONSTRUCT,
        signature="(visual: Optional[Visual], width: int, height: int, background: tuple) -> Visual",
        deterministic=True,
        doc="Create a visual layer for composition"
    )
    def layer(visual: Optional[Visual] = None, width: int = 512, height: int = 512,
              background: tuple = (0.0, 0.0, 0.0)) -> Visual:
        """Create a visual layer for composition.

        Args:
            visual: Existing visual to convert to layer (optional)
            width: Layer width if creating new layer
            height: Layer height if creating new layer
            background: Background color as (R, G, B) in [0, 1]

        Returns:
            Visual layer

        Example:
            # Create empty layer
            layer1 = visual.layer(width=512, height=512)

            # Convert existing visual to layer
            layer2 = visual.layer(existing_visual)
        """
        if visual is not None:
            if not isinstance(visual, Visual):
                raise TypeError(f"Expected Visual, got {type(visual)}")
            return visual.copy()
        else:
            # Create new empty layer
            img = np.zeros((height, width, 3), dtype=np.float32)
            img[:, :, :] = background
            return Visual(img)

    @staticmethod
    @operator(
        domain="visual",
        category=OpCategory.TRANSFORM,
        signature="(*layers: Visual, mode: str, opacity: Optional[Union[float, list]]) -> Visual",
        deterministic=True,
        doc="Composite multiple visual layers"
    )
    def composite(*layers: Visual, mode: str = "over",
                  opacity: Optional[Union[float, list]] = None) -> Visual:
        """Composite multiple visual layers.

        Args:
            *layers: Visual layers to composite (bottom to top)
            mode: Blending mode ("over", "add", "multiply", "screen", "overlay")
            opacity: Opacity for each layer (0.0 to 1.0), or single value for all

        Returns:
            Composited visual

        Example:
            # Composite field and agents
            field_vis = visual.colorize(temperature, palette="fire")
            agent_vis = visual.agents(particles, color=(1, 1, 1))
            result = visual.composite(field_vis, agent_vis, mode="add")
        """
        if len(layers) == 0:
            raise ValueError("At least one layer required")

        # Validate all layers are Visual instances
        for i, layer in enumerate(layers):
            if not isinstance(layer, Visual):
                raise TypeError(f"Layer {i} is not a Visual instance")

        # Check all layers have same dimensions
        base_shape = layers[0].shape
        for i, layer in enumerate(layers[1:], 1):
            if layer.shape != base_shape:
                raise ValueError(
                    f"Layer {i} has shape {layer.shape}, expected {base_shape}"
                )

        # Handle opacity
        if opacity is None:
            opacities = [1.0] * len(layers)
        elif isinstance(opacity, (int, float)):
            opacities = [float(opacity)] * len(layers)
        else:
            if len(opacity) != len(layers):
                raise ValueError(
                    f"opacity list length {len(opacity)} doesn't match layers {len(layers)}"
                )
            opacities = list(opacity)

        # Start with first layer
        result = layers[0].data.copy() * opacities[0]

        # Composite remaining layers
        for i, layer in enumerate(layers[1:], 1):
            alpha = opacities[i]
            top = layer.data
            bottom = result

            if mode == "over":
                # Standard alpha compositing (over operator)
                result = bottom * (1 - alpha) + top * alpha
            elif mode == "add":
                # Additive blending
                result = bottom + top * alpha
            elif mode == "multiply":
                # Multiply blending
                result = bottom * (1 - alpha + top * alpha)
            elif mode == "screen":
                # Screen blending
                result = 1 - (1 - bottom) * (1 - top * alpha)
            elif mode == "overlay":
                # Overlay blending
                mask = bottom < 0.5
                result = np.where(
                    mask,
                    2 * bottom * top * alpha + bottom * (1 - alpha),
                    1 - 2 * (1 - bottom) * (1 - top) * alpha + bottom * (1 - alpha)
                )
            else:
                raise ValueError(
                    f"Unknown blending mode: {mode}. "
                    f"Supported: 'over', 'add', 'multiply', 'screen', 'overlay'"
                )

        return Visual(result)

    @staticmethod
    @operator(
        domain="visual",
        category=OpCategory.TRANSFORM,
        signature="(frames: Union[list, Callable], path: str, fps: int, format: str, max_frames: Optional[int]) -> None",
        deterministic=True,
        doc="Export animation sequence to video file"
    )
    def video(frames: Union[list, Callable[[], Optional[Visual]]],
              path: str,
              fps: int = 30,
              format: str = "auto",
              max_frames: Optional[int] = None) -> None:
        """Export animation sequence to video file.

        Supports MP4 and GIF output formats.

        Args:
            frames: List of Visual frames or generator function
            path: Output file path
            fps: Frames per second
            format: Output format ("auto", "mp4", "gif") - auto infers from extension
            max_frames: Maximum number of frames to export (for generators)

        Raises:
            ImportError: If imageio is not installed

        Example:
            # From list of frames
            frames = [generate_frame(i) for i in range(100)]
            visual.video(frames, "output.mp4", fps=30)

            # From generator
            def gen_frames():
                temp = field.random((128, 128))
                for i in range(100):
                    temp = field.diffuse(temp, rate=0.1)
                    yield visual.colorize(temp, palette="fire")

            visual.video(gen_frames, "output.gif", fps=10)
        """
        try:
            import imageio
        except ImportError:
            raise ImportError(
                "imageio is required for video export. "
                "Install with: pip install imageio imageio-ffmpeg"
            )

        # Infer format from path if auto
        if format == "auto":
            if path.endswith(".mp4"):
                format = "mp4"
            elif path.endswith(".gif"):
                format = "gif"
            else:
                format = "mp4"  # Default

        format = format.lower()

        # Collect frames
        if callable(frames):
            # Generator function
            frame_list = []
            count = 0
            while True:
                if max_frames is not None and count >= max_frames:
                    break

                frame = frames()
                if frame is None:
                    break

                if not isinstance(frame, Visual):
                    raise TypeError(f"Frame {count} is not a Visual instance")

                frame_list.append(frame)
                count += 1
        else:
            # List of frames
            frame_list = list(frames)

            # Validate all frames
            for i, frame in enumerate(frame_list):
                if not isinstance(frame, Visual):
                    raise TypeError(f"Frame {i} is not a Visual instance")

        if len(frame_list) == 0:
            raise ValueError("No frames to export")

        print(f"Exporting {len(frame_list)} frames to {path}...")

        # Convert frames to 8-bit RGB
        rgb_frames = []
        for frame in frame_list:
            # Apply gamma correction
            srgb = VisualOperations._linear_to_srgb(frame.data)

            # Convert to 8-bit
            rgb_8bit = (srgb * 255).astype(np.uint8)
            rgb_frames.append(rgb_8bit)

        # Write video
        if format == "mp4":
            # MP4 requires ffmpeg
            try:
                imageio.mimwrite(
                    path,
                    rgb_frames,
                    fps=fps,
                    codec='libx264',
                    quality=8,
                    pixelformat='yuv420p'
                )
            except Exception as e:
                # Fall back to basic MP4 if codec not available
                imageio.mimwrite(path, rgb_frames, fps=fps)
        elif format == "gif":
            # GIF export
            imageio.mimwrite(
                path,
                rgb_frames,
                fps=fps,
                loop=0  # Infinite loop
            )
        else:
            raise ValueError(f"Unsupported format: {format}. Supported: 'mp4', 'gif'")

        print(f"Video export complete: {path}")

    # ========================================================================
    # ADVANCED VISUALIZATIONS (v0.11.0)
    # ========================================================================

    @staticmethod
    @operator(
        domain="visual",
        category=OpCategory.TRANSFORM,
        signature="(signal: Union[AudioBuffer, np.ndarray], sample_rate: int, window_size: int, hop_size: int, palette: str, log_scale: bool, freq_range: Optional[Tuple[float, float]]) -> Visual",
        deterministic=True,
        doc="Generate spectrogram visualization from audio signal"
    )
    def spectrogram(signal, sample_rate: int = 44100, window_size: int = 2048,
                   hop_size: int = 512, palette: str = "viridis",
                   log_scale: bool = True,
                   freq_range: Optional[Tuple[float, float]] = None) -> Visual:
        """Generate spectrogram visualization from audio signal.

        Args:
            signal: Audio signal (AudioBuffer or numpy array)
            sample_rate: Sample rate in Hz
            window_size: FFT window size
            hop_size: Hop size between windows
            palette: Color palette for magnitude mapping
            log_scale: Use logarithmic scale for magnitude (dB)
            freq_range: (min_freq, max_freq) in Hz to display (None = full range)

        Returns:
            Visual representation of spectrogram (time on x-axis, frequency on y-axis)

        Example:
            # Visualize audio signal
            audio = audio_io.read("sound.wav")
            spec_vis = visual.spectrogram(audio, palette="fire", log_scale=True)
            visual.output(spec_vis, "spectrogram.png")
        """
        from .audio import AudioBuffer

        # Extract audio data
        if isinstance(signal, AudioBuffer):
            audio_data = signal.data
            if signal.sample_rate is not None:
                sample_rate = signal.sample_rate
        elif isinstance(signal, np.ndarray):
            audio_data = signal
        else:
            raise TypeError(f"Expected AudioBuffer or ndarray, got {type(signal)}")

        # Flatten if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=0)

        # Compute STFT
        n_frames = (len(audio_data) - window_size) // hop_size + 1
        n_freqs = window_size // 2 + 1

        # Create spectrogram array
        spectrogram_data = np.zeros((n_freqs, n_frames))

        # Hann window
        window = np.hanning(window_size)

        for i in range(n_frames):
            start = i * hop_size
            end = start + window_size
            if end > len(audio_data):
                break

            # Apply window and FFT
            windowed = audio_data[start:end] * window
            spectrum = np.fft.rfft(windowed)
            magnitude = np.abs(spectrum)

            spectrogram_data[:, i] = magnitude

        # Apply frequency range filter if specified
        if freq_range is not None:
            min_freq, max_freq = freq_range
            freq_bins = np.fft.rfftfreq(window_size, 1.0 / sample_rate)
            min_bin = np.searchsorted(freq_bins, min_freq)
            max_bin = np.searchsorted(freq_bins, max_freq)
            spectrogram_data = spectrogram_data[min_bin:max_bin, :]

        # Convert to dB scale if requested
        if log_scale:
            spectrogram_data = 20 * np.log10(spectrogram_data + 1e-10)  # Avoid log(0)
            vmin, vmax = -80, 0  # dB range
        else:
            vmin, vmax = None, None

        # Flip vertically (high frequencies at top)
        spectrogram_data = np.flip(spectrogram_data, axis=0)

        # Create Field2D for colorization
        from .field import Field2D
        spec_field = Field2D(spectrogram_data, dx=1.0, dy=1.0)

        # Colorize using existing palette system
        return VisualOperations.colorize(spec_field, palette=palette, vmin=vmin, vmax=vmax)

    @staticmethod
    @operator(
        domain="visual",
        category=OpCategory.TRANSFORM,
        signature="(graph: Graph, width: int, height: int, node_size: float, node_color: tuple, edge_color: tuple, edge_width: float, layout: str, iterations: int, color_by_centrality: bool, palette: str, show_labels: bool, background: tuple) -> Visual",
        deterministic=True,
        doc="Visualize graph/network with force-directed layout"
    )
    def graph(graph, width: int = 800, height: int = 800,
             node_size: float = 8.0, node_color: tuple = (0.3, 0.6, 1.0),
             edge_color: tuple = (0.5, 0.5, 0.5), edge_width: float = 1.0,
             layout: str = "force", iterations: int = 50,
             color_by_centrality: bool = False, palette: str = "viridis",
             show_labels: bool = False,
             background: tuple = (0.0, 0.0, 0.0)) -> Visual:
        """Visualize graph/network with force-directed layout.

        Args:
            graph: Graph instance to visualize
            width: Output image width
            height: Output image height
            node_size: Node radius in pixels
            node_color: Default node color (R, G, B) in [0, 1]
            edge_color: Edge color (R, G, B) in [0, 1]
            edge_width: Edge line width in pixels
            layout: Layout algorithm ("force", "circular", "grid")
            iterations: Number of force-directed iterations
            color_by_centrality: Color nodes by degree centrality
            palette: Palette for centrality coloring
            show_labels: Show node labels (node IDs)
            background: Background color (R, G, B) in [0, 1]

        Returns:
            Visual representation of the graph
        """
        from .graph import Graph

        if not isinstance(graph, Graph):
            raise TypeError(f"Expected Graph, got {type(graph)}")

        n_nodes = graph.num_nodes
        if n_nodes == 0:
            img = np.zeros((height, width, 3), dtype=np.float32)
            img[:, :, :] = background
            return Visual(img)

        # Compute layout
        layout_funcs = {
            "force": lambda: VisualOperations._force_directed_layout(graph, iterations, width, height),
            "circular": lambda: VisualOperations._circular_layout(n_nodes, width, height),
            "grid": lambda: VisualOperations._grid_layout(n_nodes, width, height),
        }
        if layout not in layout_funcs:
            raise ValueError(f"Unknown layout: {layout}. Supported: {list(layout_funcs.keys())}")
        positions = layout_funcs[layout]()

        # Create output image
        img = np.zeros((height, width, 3), dtype=np.float32)
        img[:, :, :] = background

        # Draw edges first (so nodes appear on top)
        adj = graph.adjacency_list
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if _get_edge_weight(adj, i, j) > 0:
                    x1, y1 = positions[i]
                    x2, y2 = positions[j]
                    VisualOperations._draw_line(img, int(x1), int(y1), int(x2), int(y2),
                                                edge_color, edge_width)

        # Compute node colors
        if color_by_centrality:
            degrees = np.array([len(graph.adjacency_list.get(i, [])) for i in range(n_nodes)], dtype=float)
            palette_colors = np.array(VisualOperations.PALETTES[palette])
            node_colors = _map_property_to_colors(degrees, palette_colors)
        else:
            node_colors = np.tile(node_color, (n_nodes, 1))

        # Draw nodes
        for i in range(n_nodes):
            x, y = positions[i]
            VisualOperations._draw_circle(img, int(x), int(y), int(node_size), node_colors[i])

        return Visual(img)

    @staticmethod
    def _force_directed_layout(graph, iterations: int = 50,
                              width: int = 800, height: int = 800) -> np.ndarray:
        """Compute force-directed graph layout using Fruchterman-Reingold algorithm."""
        n_nodes = graph.num_nodes

        # Initialize positions randomly
        positions = np.random.rand(n_nodes, 2)
        positions[:, 0] *= width
        positions[:, 1] *= height

        # Parameters
        area = width * height
        k = np.sqrt(area / n_nodes)  # Optimal distance
        t = width / 10.0  # Temperature (decreases over time)
        adj = graph.adjacency_list

        for iteration in range(iterations):
            # Repulsive forces between all nodes
            forces = np.zeros((n_nodes, 2))

            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j:
                        delta = positions[i] - positions[j]
                        distance = np.linalg.norm(delta) + 0.01  # Avoid division by zero
                        repulsion = (k * k) / distance
                        forces[i] += (delta / distance) * repulsion

            # Attractive forces for connected nodes
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if _get_edge_weight(adj, i, j) > 0:
                        delta = positions[i] - positions[j]
                        distance = np.linalg.norm(delta) + 0.01
                        attraction = (distance * distance) / k
                        force_vec = (delta / distance) * attraction
                        forces[i] -= force_vec
                        forces[j] += force_vec

            # Update positions
            for i in range(n_nodes):
                force_mag = np.linalg.norm(forces[i])
                if force_mag > 0:
                    displacement = (forces[i] / force_mag) * min(force_mag, t)
                    positions[i] += displacement

                    # Keep within bounds
                    positions[i, 0] = np.clip(positions[i, 0], 10, width - 10)
                    positions[i, 1] = np.clip(positions[i, 1], 10, height - 10)

            # Cool temperature
            t *= 0.95

        return positions

    @staticmethod
    def _circular_layout(n_nodes: int, width: int, height: int) -> np.ndarray:
        """Arrange nodes in a circle."""
        positions = np.zeros((n_nodes, 2))
        radius = min(width, height) * 0.4
        center_x, center_y = width / 2, height / 2

        for i in range(n_nodes):
            angle = 2 * np.pi * i / n_nodes
            positions[i, 0] = center_x + radius * np.cos(angle)
            positions[i, 1] = center_y + radius * np.sin(angle)

        return positions

    @staticmethod
    def _grid_layout(n_nodes: int, width: int, height: int) -> np.ndarray:
        """Arrange nodes in a grid."""
        cols = int(np.ceil(np.sqrt(n_nodes)))
        rows = int(np.ceil(n_nodes / cols))

        positions = np.zeros((n_nodes, 2))
        spacing_x = width / (cols + 1)
        spacing_y = height / (rows + 1)

        for i in range(n_nodes):
            row = i // cols
            col = i % cols
            positions[i, 0] = spacing_x * (col + 1)
            positions[i, 1] = spacing_y * (row + 1)

        return positions

    @staticmethod
    def _draw_line(img: np.ndarray, x0: int, y0: int, x1: int, y1: int,
                   color: tuple, width: float = 1.0):
        """Draw line using Bresenham's algorithm."""
        height, img_width, _ = img.shape

        # Bresenham's line algorithm
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0
        while True:
            # Draw point with thickness
            for dy_offset in range(int(-width), int(width) + 1):
                for dx_offset in range(int(-width), int(width) + 1):
                    if dx_offset * dx_offset + dy_offset * dy_offset <= width * width:
                        px, py = x + dx_offset, y + dy_offset
                        if 0 <= px < img_width and 0 <= py < height:
                            img[py, px] = color

            if x == x1 and y == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    @staticmethod
    def _draw_circle(img: np.ndarray, cx: int, cy: int, radius: int, color: tuple):
        """Draw filled circle."""
        height, width, _ = img.shape

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    x, y = cx + dx, cy + dy
                    if 0 <= x < width and 0 <= y < height:
                        img[y, x] = color

    @staticmethod
    @operator(
        domain="visual",
        category=OpCategory.TRANSFORM,
        signature="(agents: Agents, position_property: str, velocity_property: str, width: int, height: int, color_property: Optional[str], palette: str, point_size: float, alpha: float, show_trajectories: bool, background: tuple) -> Visual",
        deterministic=True,
        doc="Visualize phase space diagram (position vs velocity)"
    )
    def phase_space(agents, position_property: str = 'pos',
                   velocity_property: str = 'vel',
                   width: int = 512, height: int = 512,
                   color_property: Optional[str] = None,
                   palette: str = "viridis",
                   point_size: float = 2.0,
                   alpha: float = 0.6,
                   show_trajectories: bool = False,
                   background: tuple = (0.0, 0.0, 0.0)) -> Visual:
        """Visualize phase space diagram (position vs velocity).

        Creates a scatter plot showing the relationship between position and velocity,
        useful for analyzing dynamical systems and particle behaviors.

        Args:
            agents: Agents instance to visualize
            position_property: Name of position property (must be 1D or 2D)
            velocity_property: Name of velocity property (must match position dims)
            width: Output image width
            height: Output image height
            color_property: Property to color points by (optional)
            palette: Color palette for color_property mapping
            point_size: Point radius in pixels
            alpha: Point transparency [0, 1]
            show_trajectories: Connect points in agent order
            background: Background color (R, G, B) in [0, 1]

        Returns:
            Visual representation of phase space
        """
        from .agents import Agents

        if not isinstance(agents, Agents):
            raise TypeError(f"Expected Agents, got {type(agents)}")

        # Get position and velocity values (use magnitude for multi-dimensional)
        positions = agents.get(position_property)
        velocities = agents.get(velocity_property)
        pos_values = np.linalg.norm(positions, axis=1) if len(positions.shape) > 1 else positions
        vel_values = np.linalg.norm(velocities, axis=1) if len(velocities.shape) > 1 else velocities

        # Create output image
        img = np.zeros((height, width, 3), dtype=np.float32)
        img[:, :, :] = background

        # Normalize to pixel coordinates with 10px padding
        pos_range = max(np.max(pos_values) - np.min(pos_values), 1e-10)
        vel_range = max(np.max(vel_values) - np.min(vel_values), 1e-10)
        pos_norm = (pos_values - np.min(pos_values)) / pos_range
        vel_norm = (vel_values - np.min(vel_values)) / vel_range

        px = np.clip((pos_norm * (width - 20) + 10).astype(int), 0, width - 1)
        py = np.clip(((1.0 - vel_norm) * (height - 20) + 10).astype(int), 0, height - 1)

        # Compute colors
        palette_colors = np.array(VisualOperations.PALETTES[palette])
        if color_property is not None:
            colors = _map_property_to_colors(agents.get(color_property), palette_colors)
        else:
            colors = np.tile((1.0, 1.0, 1.0), (len(pos_values), 1))

        # Draw trajectories if requested
        if show_trajectories and len(px) > 1:
            for i in range(len(px) - 1):
                VisualOperations._draw_line(img, px[i], py[i], px[i + 1], py[i + 1],
                                            colors[i] * 0.3, 1.0)

        # Draw points
        for i in range(len(px)):
            _render_filled_circle(img, px[i], py[i], int(point_size), colors[i], alpha, "alpha")

        return Visual(img)

    @staticmethod
    @operator(
        domain="visual",
        category=OpCategory.TRANSFORM,
        signature="(visual: Visual, metrics: dict, position: str, font_size: int, text_color: tuple, bg_color: tuple, bg_alpha: float) -> Visual",
        deterministic=True,
        doc="Overlay metrics dashboard on visualization"
    )
    def add_metrics(visual: Visual, metrics: dict,
                   position: str = "top-left",
                   font_size: int = 14,
                   text_color: tuple = (1.0, 1.0, 1.0),
                   bg_color: tuple = (0.0, 0.0, 0.0),
                   bg_alpha: float = 0.7) -> Visual:
        """Overlay metrics dashboard on visualization.

        Args:
            visual: Visual to add metrics to
            metrics: Dictionary of metric name -> value pairs
            position: Position ("top-left", "top-right", "bottom-left", "bottom-right")
            font_size: Font size in pixels
            text_color: Text color (R, G, B) in [0, 1]
            bg_color: Background color (R, G, B) in [0, 1]
            bg_alpha: Background transparency [0, 1]

        Returns:
            Visual with metrics overlay
        """
        if not isinstance(visual, Visual):
            raise TypeError(f"Expected Visual, got {type(visual)}")

        result = visual.copy()
        img = result.data
        height, width = img.shape[:2]

        # Format metrics text
        lines = [f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}"
                 for k, v in metrics.items()]
        if not lines:
            return result

        # Calculate text dimensions
        char_width = font_size * 0.6
        char_height = font_size * 1.2
        line_height = int(char_height * 1.3)
        text_width = int(max(len(line) for line in lines) * char_width) + 20
        text_height = len(lines) * line_height + 10

        # Get position and draw background
        x_start, y_start = _compute_text_position(position, width, height, text_width, text_height)
        _draw_background_box(img, x_start, y_start, text_width, text_height, bg_color, bg_alpha)

        # Draw text (simple raster rendering)
        for i, line in enumerate(lines):
            y_pos = y_start + 5 + i * line_height
            x_pos = x_start + 10
            for char_idx, char in enumerate(line):
                if char == ' ':
                    continue
                char_x = x_pos + int(char_idx * char_width)
                for dy in range(int(char_height)):
                    for dx in range(int(char_width * 0.8)):
                        px, py = char_x + dx, y_pos + dy
                        if 0 <= py < height and 0 <= px < width and (dy % 2 == 0 or dx % 2 == 0):
                            img[py, px] = text_color

        return result


# Create singleton instance for use as 'visual' namespace
visual = VisualOperations()

# Define __all__ for explicit exports (used by `use visual`)
__all__ = [
    'visual',
    'Visual',
    'colorize',
    'layer',
    'composite',
    'agents',
    'display',
    'output',
    'video',
    'spectrogram',
    'graph',
    'phase_space',
    'add_metrics',
]

# Export operators for domain registry discovery
layer = VisualOperations.layer
colorize = VisualOperations.colorize
composite = VisualOperations.composite
agents = VisualOperations.agents
display = VisualOperations.display
output = VisualOperations.output
video = VisualOperations.video
spectrogram = VisualOperations.spectrogram
graph = VisualOperations.graph
phase_space = VisualOperations.phase_space
add_metrics = VisualOperations.add_metrics
