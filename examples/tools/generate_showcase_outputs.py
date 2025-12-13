#!/usr/bin/env python3
"""
Morphogen Showcase Output Generator

A comprehensive utility for generating high-quality outputs from Morphogen examples
in multiple formats with various quality presets.

Features:
- Multi-format export (PNG, MP4, GIF, WAV)
- Quality presets (draft, web, production, print)
- Cross-domain outputs (audio + visual synchronization)
- Organized output directories with metadata
- Progress reporting and timing

Usage:
    python generate_showcase_outputs.py --example fractal --preset production
    python generate_showcase_outputs.py --all --preset web
    python generate_showcase_outputs.py --example physics --formats mp4,png
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable, Any
import time

# Add parent directory to path to import morphogen
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import numpy as np
    from morphogen.stdlib import field, visual, audio
    MORPHOGEN_AVAILABLE = True
except ImportError:
    MORPHOGEN_AVAILABLE = False
    print("Warning: Morphogen not available. Running in documentation mode only.")


# Quality Presets
QUALITY_PRESETS = {
    'draft': {
        'resolution': (512, 512),
        'fps': 15,
        'max_duration': 5,  # seconds
        'audio_sr': 22050,
        'description': 'Quick preview, low quality'
    },
    'web': {
        'resolution': (1280, 720),  # 720p
        'fps': 30,
        'max_duration': 30,
        'audio_sr': 44100,
        'description': 'Social media optimized'
    },
    'production': {
        'resolution': (1920, 1080),  # 1080p
        'fps': 30,
        'max_duration': 60,
        'audio_sr': 44100,
        'description': 'High quality, portfolio ready'
    },
    'print': {
        'resolution': (3840, 2160),  # 4K
        'fps': 60,
        'max_duration': 30,
        'audio_sr': 48000,
        'description': 'Ultra high quality, publication ready'
    }
}


class OutputGenerator:
    """Manages output generation for Morphogen examples."""

    def __init__(self, output_dir: Path = None, preset: str = 'web'):
        """
        Initialize output generator.

        Args:
            output_dir: Base output directory (default: examples/outputs/)
            preset: Quality preset name (draft, web, production, print)
        """
        self.output_dir = output_dir or Path(__file__).parent.parent / "outputs"
        self.preset_name = preset
        self.preset = QUALITY_PRESETS[preset]
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_output_subdir(self, name: str) -> Path:
        """Create timestamped output subdirectory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        subdir = self.output_dir / f"{name}_{self.preset_name}_{timestamp}"
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir

    def save_metadata(self, output_dir: Path, metadata: Dict[str, Any]):
        """Save generation metadata as JSON."""
        metadata['generated_at'] = datetime.now().isoformat()
        metadata['preset'] = self.preset_name
        metadata['preset_config'] = self.preset

        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  ✓ Saved metadata to {output_dir / 'metadata.json'}")

    def export_frames(
        self,
        frames: List,
        output_dir: Path,
        name: str,
        formats: List[str] = None
    ):
        """
        Export frames to multiple formats.

        Args:
            frames: List of Visual objects or numpy arrays
            output_dir: Output directory
            name: Base filename
            formats: List of formats to export (png, mp4, gif)
        """
        if formats is None:
            formats = ['png', 'mp4', 'gif']

        fps = self.preset['fps']

        # Always save thumbnail (first frame or middle frame)
        if 'png' in formats:
            thumb_frame = frames[len(frames) // 2] if len(frames) > 1 else frames[0]
            thumb_path = output_dir / f"{name}_thumbnail.png"
            visual.output(thumb_frame, str(thumb_path))
            print(f"  ✓ Saved thumbnail: {thumb_path.name}")

            # Save key frames
            if len(frames) > 1:
                key_indices = [0, len(frames) // 4, len(frames) // 2, 3 * len(frames) // 4, -1]
                for i, idx in enumerate(key_indices):
                    if 0 <= idx < len(frames) or idx == -1:
                        key_path = output_dir / f"{name}_keyframe_{i}.png"
                        visual.output(frames[idx], str(key_path))
                print(f"  ✓ Saved {len(key_indices)} key frames")

        # Video export (MP4)
        if 'mp4' in formats and len(frames) > 1:
            mp4_path = output_dir / f"{name}.mp4"
            visual.video(frames, str(mp4_path), fps=fps)
            print(f"  ✓ Saved video: {mp4_path.name} ({len(frames)} frames @ {fps} fps)")

        # GIF export (looping, optimized)
        if 'gif' in formats and len(frames) > 1:
            # For GIF, use lower fps and shorter duration to reduce file size
            gif_fps = min(15, fps)
            max_gif_frames = gif_fps * 10  # 10 second max

            gif_frames = frames[:max_gif_frames]

            # Optionally skip frames for smaller file
            if len(gif_frames) > 150:
                gif_frames = gif_frames[::2]  # Every other frame

            gif_path = output_dir / f"{name}_loop.gif"
            visual.video(gif_frames, str(gif_path), fps=gif_fps)
            print(f"  ✓ Saved GIF: {gif_path.name} ({len(gif_frames)} frames @ {gif_fps} fps)")

    def export_audio(
        self,
        audio_data: np.ndarray,
        output_dir: Path,
        name: str,
        formats: List[str] = None
    ):
        """
        Export audio to multiple formats.

        Args:
            audio_data: Audio samples (1D for mono, 2D for stereo)
            output_dir: Output directory
            name: Base filename
            formats: List of formats to export (wav, flac)
        """
        if formats is None:
            formats = ['wav']

        sample_rate = self.preset['audio_sr']

        # Check for clipping
        peak = np.max(np.abs(audio_data))
        if peak > 1.0:
            print(f"  ⚠ Warning: Audio peak = {peak:.2f}, normalizing to prevent clipping")
            audio_data = audio_data / peak * 0.95

        # Create AudioBuffer
        audio_buffer = audio.AudioBuffer(data=audio_data, sample_rate=sample_rate)

        # WAV export
        if 'wav' in formats:
            wav_path = output_dir / f"{name}.wav"
            audio.audio.save(audio_buffer, str(wav_path))
            duration = len(audio_data) / sample_rate
            print(f"  ✓ Saved audio (WAV): {wav_path.name} ({duration:.1f}s @ {sample_rate}Hz)")

        # FLAC export (lossless compression)
        if 'flac' in formats:
            flac_path = output_dir / f"{name}.flac"
            audio.audio.save(audio_buffer, str(flac_path))
            print(f"  ✓ Saved audio (FLAC): {flac_path.name}")


# Example Generators
# These are placeholder functions that would call actual Morphogen examples

def generate_reaction_diffusion(
    generator: OutputGenerator,
    seed: int = 42,
    duration_seconds: float = None
) -> Tuple[List, Optional[np.ndarray], Dict]:
    """
    Generate reaction-diffusion visualization with optional audio.

    Returns:
        (frames, audio, metadata)
    """
    print("Generating reaction-diffusion (Gray-Scott)...")

    if duration_seconds is None:
        duration_seconds = generator.preset['max_duration']

    np.random.seed(seed)

    # Parameters
    Du, Dv = 0.16, 0.08
    F, K = 0.060, 0.062
    width, height = generator.preset['resolution']

    # Scale grid to resolution
    grid_size = min(width, height)

    # Initialize
    u = field.alloc((grid_size, grid_size), fill_value=1.0)
    v = field.alloc((grid_size, grid_size), fill_value=0.0)

    # Perturbation
    cx, cy = grid_size // 2, grid_size // 2
    radius = grid_size // 10

    for y in range(grid_size):
        for x in range(grid_size):
            dx, dy = x - cx, y - cy
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < radius:
                u.data[y, x] = 0.5
                # Add randomness
                u.data[y, x] += (np.random.rand() - 0.5) * 0.1
                v.data[y, x] = 0.25 + np.random.rand() * 0.1

    # Simulation
    fps = generator.preset['fps']
    n_frames = int(duration_seconds * fps)
    frames = []

    print(f"  Simulating {n_frames} frames @ {fps} fps...")

    for step in range(n_frames):
        if step % (fps * 2) == 0:  # Progress every 2 seconds
            print(f"    Frame {step}/{n_frames} ({step/fps:.1f}s)")

        # Gray-Scott step
        uvv = u.data * v.data * v.data
        du_dt = Du * field.laplacian(u).data - uvv + F * (1.0 - u.data)
        dv_dt = Dv * field.laplacian(v).data + uvv - (F + K) * v.data

        u.data += du_dt
        v.data += dv_dt

        # Render
        vis = visual.colorize(v, palette="viridis", vmin=0.0, vmax=1.0)

        # Resize if needed
        if width != grid_size or height != grid_size:
            # Would resize here if needed
            pass

        frames.append(vis)

    metadata = {
        'example': 'reaction_diffusion',
        'algorithm': 'Gray-Scott',
        'parameters': {
            'Du': Du, 'Dv': Dv, 'F': F, 'K': K,
        },
        'frames': len(frames),
        'fps': fps,
        'duration_seconds': len(frames) / fps,
        'resolution': list(generator.preset['resolution']),
        'seed': seed,
    }

    return frames, None, metadata


def generate_physics_particles(
    generator: OutputGenerator,
    seed: int = 42,
    duration_seconds: float = None
) -> Tuple[List, Optional[np.ndarray], Dict]:
    """
    Generate physics-based particle simulation.

    Returns:
        (frames, audio, metadata)
    """
    print("Generating physics particle simulation...")

    if duration_seconds is None:
        duration_seconds = generator.preset['max_duration']

    np.random.seed(seed)

    width, height = generator.preset['resolution']
    fps = generator.preset['fps']
    n_frames = int(duration_seconds * fps)

    # This would use actual agents/physics domain
    # For now, create a simple placeholder visualization

    frames = []

    print(f"  Simulating {n_frames} frames @ {fps} fps...")

    # Placeholder: Create a simple animated field
    for step in range(n_frames):
        if step % (fps * 2) == 0:
            print(f"    Frame {step}/{n_frames} ({step/fps:.1f}s)")

        # Create a time-evolving field
        t = step / fps
        x = np.linspace(0, 4*np.pi, width)
        y = np.linspace(0, 4*np.pi, height)
        X, Y = np.meshgrid(x, y)

        # Animated pattern
        Z = np.sin(X + t) * np.cos(Y + t * 0.5)

        # Colorize
        from morphogen.stdlib.field import Field2D
        field_obj = Field2D(Z.astype(np.float32))
        vis = visual.colorize(field_obj, palette="fire", vmin=-1.0, vmax=1.0)

        frames.append(vis)

    metadata = {
        'example': 'physics_particles',
        'frames': len(frames),
        'fps': fps,
        'duration_seconds': len(frames) / fps,
        'resolution': list(generator.preset['resolution']),
        'seed': seed,
    }

    return frames, None, metadata


def generate_fractal_zoom(
    generator: OutputGenerator,
    seed: int = 42,
    duration_seconds: float = None
) -> Tuple[List, Optional[np.ndarray], Dict]:
    """
    Generate fractal zoom animation.

    Returns:
        (frames, audio, metadata)
    """
    print("Generating fractal zoom (Mandelbrot)...")

    if duration_seconds is None:
        duration_seconds = min(20, generator.preset['max_duration'])

    np.random.seed(seed)

    width, height = generator.preset['resolution']
    fps = generator.preset['fps']
    n_frames = int(duration_seconds * fps)

    frames = []

    # Zoom parameters
    target_x, target_y = -0.7, 0.0  # Interesting region
    zoom_factor = 1.1  # Per frame

    print(f"  Rendering {n_frames} frames @ {fps} fps...")

    for step in range(n_frames):
        if step % (fps * 2) == 0:
            print(f"    Frame {step}/{n_frames} ({step/fps:.1f}s)")

        # Calculate zoom level
        zoom = zoom_factor ** step
        size = 3.0 / zoom

        x_min = target_x - size
        x_max = target_x + size
        y_min = target_y - size
        y_max = target_y + size

        # Render Mandelbrot set
        max_iter = 256

        x = np.linspace(x_min, x_max, width)
        y = np.linspace(y_min, y_max, height)
        X, Y = np.meshgrid(x, y)

        C = X + 1j * Y
        Z = np.zeros_like(C)
        M = np.zeros(C.shape)

        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask]**2 + C[mask]
            M[mask] = i

        # Smooth coloring
        M_smooth = M + 1 - np.log(np.log(np.abs(Z) + 1)) / np.log(2)
        M_smooth = M_smooth / max_iter

        # Colorize
        from morphogen.stdlib.field import Field2D
        field_obj = Field2D(M_smooth.astype(np.float32))
        vis = visual.colorize(field_obj, palette="viridis", vmin=0.0, vmax=1.0)

        frames.append(vis)

    metadata = {
        'example': 'fractal_zoom',
        'fractal_type': 'Mandelbrot',
        'target': [target_x, target_y],
        'zoom_factor': zoom_factor,
        'max_iterations': max_iter,
        'frames': len(frames),
        'fps': fps,
        'duration_seconds': len(frames) / fps,
        'resolution': list(generator.preset['resolution']),
        'seed': seed,
    }

    return frames, None, metadata


# Import example generators
def import_example_generators():
    """Import example generators from their respective modules."""
    generators = {
        'reaction_diffusion': generate_reaction_diffusion,
        'physics': generate_physics_particles,
        'fractal': generate_fractal_zoom,
    }

    # Try to import cross-domain field-agent coupling
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from cross_domain_field_agent_coupling import generate_field_agent_coupling
        generators['field_agent_coupling'] = generate_field_agent_coupling
    except ImportError as e:
        print(f"Warning: Could not import field_agent_coupling generator: {e}")

    # Try to import fireworks with audio
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / 'agents'))
        from fireworks_particles import generate_fireworks_with_audio
        generators['fireworks_audio'] = generate_fireworks_with_audio
    except ImportError as e:
        print(f"Warning: Could not import fireworks_audio generator: {e}")

    # Try to import audio visualizer
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / 'showcase'))
        from showcase.audio_visualizer_05 import generate_audio_visualizer
        generators['audio_visualizer'] = generate_audio_visualizer
    except ImportError:
        # Try alternative import path
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "audio_visualizer",
                Path(__file__).parent.parent / 'showcase' / '05_audio_visualizer.py'
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                generators['audio_visualizer'] = module.generate_audio_visualizer
        except Exception as e:
            print(f"Warning: Could not import audio_visualizer generator: {e}")

    return generators


# Registry of available examples
EXAMPLE_GENERATORS = import_example_generators()


def generate_example(
    example_name: str,
    generator: OutputGenerator,
    formats: List[str] = None,
    seed: int = 42
):
    """Generate outputs for a specific example."""

    if example_name not in EXAMPLE_GENERATORS:
        print(f"Error: Unknown example '{example_name}'")
        print(f"Available examples: {', '.join(EXAMPLE_GENERATORS.keys())}")
        return

    print("=" * 70)
    print(f"Generating: {example_name}")
    print(f"Preset: {generator.preset_name} - {generator.preset['description']}")
    print(f"Resolution: {generator.preset['resolution']}")
    print(f"FPS: {generator.preset['fps']}")
    print("=" * 70)
    print()

    start_time = time.time()

    # Create output directory
    output_dir = generator.create_output_subdir(example_name)
    print(f"Output directory: {output_dir}")
    print()

    # Generate frames and audio
    generator_func = EXAMPLE_GENERATORS[example_name]
    frames, audio_data, metadata = generator_func(generator, seed=seed)

    print()

    # Export visual outputs
    if formats is None:
        formats = ['png', 'mp4', 'gif']

    visual_formats = [f for f in formats if f in ['png', 'mp4', 'gif']]
    audio_formats = [f for f in formats if f in ['wav', 'flac']]

    if visual_formats:
        print("Exporting visual outputs...")
        generator.export_frames(frames, output_dir, example_name, visual_formats)
        print()

    # Export audio outputs
    if audio_data is not None and audio_formats:
        print("Exporting audio outputs...")
        generator.export_audio(audio_data, output_dir, example_name, audio_formats)
        print()

    # Save metadata
    elapsed = time.time() - start_time
    metadata['generation_time_seconds'] = elapsed

    generator.save_metadata(output_dir, metadata)

    print()
    print("=" * 70)
    print(f"✓ Completed in {elapsed:.1f} seconds")
    print(f"✓ Outputs saved to: {output_dir}")
    print("=" * 70)
    print()


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description='Generate high-quality outputs from Morphogen examples',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --example reaction_diffusion --preset production
  %(prog)s --example fractal --preset web --formats mp4,gif
  %(prog)s --all --preset draft
  %(prog)s --list

Available presets:
  draft      - Quick preview (512x512, 15fps, 5s)
  web        - Social media (1280x720, 30fps, 30s)
  production - High quality (1920x1080, 30fps, 60s)
  print      - Ultra quality (3840x2160, 60fps, 30s)
        """
    )

    parser.add_argument(
        '--example', '-e',
        type=str,
        help='Example to generate (reaction_diffusion, physics, fractal, field_agent_coupling, fireworks_audio)'
    )

    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Generate all examples'
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available examples and exit'
    )

    parser.add_argument(
        '--preset', '-p',
        type=str,
        default='web',
        choices=QUALITY_PRESETS.keys(),
        help='Quality preset (default: web)'
    )

    parser.add_argument(
        '--formats', '-f',
        type=str,
        default='png,mp4,gif',
        help='Comma-separated list of output formats (default: png,mp4,gif)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        help='Output directory (default: examples/outputs/)'
    )

    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # List examples
    if args.list:
        print("Available examples:")
        for name in EXAMPLE_GENERATORS.keys():
            print(f"  - {name}")
        print()
        print("Quality presets:")
        for name, config in QUALITY_PRESETS.items():
            res = config['resolution']
            print(f"  - {name:10} : {res[0]}x{res[1]}, {config['fps']}fps, {config['max_duration']}s max - {config['description']}")
        return

    if not MORPHOGEN_AVAILABLE:
        print("Error: Morphogen is not available. Please install Morphogen first.")
        return 1

    # Parse formats
    formats = args.formats.split(',')

    # Create generator
    generator = OutputGenerator(
        output_dir=args.output_dir,
        preset=args.preset
    )

    # Generate examples
    if args.all:
        for example_name in EXAMPLE_GENERATORS.keys():
            generate_example(example_name, generator, formats, args.seed)
    elif args.example:
        generate_example(args.example, generator, formats, args.seed)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
