# Claude Context - Morphogen Project

## Project Overview

**Morphogen** (formerly Kairo) is a universal, deterministic computation platform that unifies multiple computational domains: audio synthesis, physics simulation, circuit design, geometry, and optimization—all in one type system, scheduler, and language.

- **Version**: 0.11.0
- **Status**: Production-Ready (40 computational domains)
- **Language**: Python-based runtime with MLIR compilation target
- **Philosophy**: Computation = Composition across domains

## Quick Start Workflow

**New to the project? Start here:**

1. **Install and verify**
   ```bash
   cd morphogen
   pip install -e ".[dev,audio,viz]"
   pytest tests/test_field_operations.py -v  # Should pass
   ```

2. **Explore project structure**
   ```bash
   ./scripts/reveal.sh 1 morphogen/stdlib/     # See all 40 domains
   ./scripts/reveal.sh 2 docs/architecture/overview.md  # Architecture overview
   ```

3. **Run an example**
   ```bash
   morphogen run examples/01_hello_heat.kairo
   # Or for Python examples:
   python examples/field_ops/heat_diffusion.py
   ```

4. **Understand a domain**
   ```bash
   ./scripts/reveal.sh 1 morphogen/stdlib/audio.py        # See operators
   pytest tests/test_audio_basic.py -v                    # Verify tests pass
   cat examples/audio/karplus_strong.py                   # See example
   ```

5. **Make a change and test**
   ```bash
   # Edit a file, then run its tests
   pytest tests/test_<domain>.py -v
   ```

## Key Capabilities

### Production-Ready Domains (40+)
- **Audio/DSP**: Synthesis, filters, effects, physical modeling
- **Physics**: RigidBody dynamics, fluid simulation, field operations
- **Agents**: Particle systems, boids, N-body simulations
- **Graph**: Network analysis, path algorithms, community detection
- **Signal**: FFT, STFT, filtering, spectral analysis
- **Vision**: Edge detection, feature extraction, morphology
- **Chemistry**: Molecular dynamics, quantum chem, kinetics, thermodynamics
- **Procedural**: Noise, terrain, color palettes, image processing
- **Infrastructure**: Sparse linear algebra, integrators, I/O storage

### Cross-Domain Integration
All domains share:
- **Type system** with physical units (m, s, K, Hz, etc.)
- **Multirate scheduler** (audio @ 48kHz, control @ 60Hz, physics @ 240Hz)
- **MLIR compiler** (6 custom dialects → LLVM/GPU)
- **Deterministic execution** (bit-exact reproducibility)

## Installation & Setup

### Quick Start
```bash
# Clone and install
git clone https://github.com/scottsen/morphogen.git
cd morphogen
pip install -e .

# With optional dependencies
pip install -e ".[dev]"      # Development tools (pytest, black, mypy)
pip install -e ".[audio]"    # Audio I/O (sounddevice, soundfile)
pip install -e ".[video]"    # Video export (imageio, ffmpeg)
pip install -e ".[viz]"      # Visualization (matplotlib, pillow)
pip install -e ".[io]"       # All I/O features

# Verify installation
morphogen version
pytest tests/ -v
```

### Optional Dependencies
- **audio**: Real-time playback/recording (`audio.play()`, `audio.record()`)
- **video**: MP4/GIF export (`visual.video()`)
- **viz**: Advanced visualization (matplotlib integration)
- **dev**: Testing, linting, type checking (pytest, black, mypy, ruff)
- **mlir**: MLIR compilation (experimental, requires separate installation)

## Project Structure

```
morphogen/
├── morphogen/               # Main package
│   ├── ast/                 # AST nodes and visitors
│   ├── lexer/               # Tokenizer
│   ├── parser/              # Recursive descent parser
│   ├── runtime/             # Python interpreter
│   ├── mlir/                # MLIR compilation (6 dialects)
│   ├── stdlib/              # Domain implementations (40 domains)
│   ├── cross_domain/        # Cross-domain type safety
│   ├── core/                # Domain registry
│   └── cli.py               # Command-line interface
├── tests/                   # 900+ tests across 63 files
├── examples/                # Working examples for all domains
│   ├── *.kairo              # Language examples
│   ├── agents/              # Agent simulations
│   ├── audio/               # Audio synthesis
│   ├── field_ops/           # Field operations
│   └── ...                  # 27 domain directories
├── docs/                    # Comprehensive documentation
│   ├── architecture/        # System design
│   ├── specifications/      # 19 domain specs
│   ├── guides/              # Implementation guides
│   ├── adr/                 # Architectural decisions
│   └── roadmap/             # Development roadmap
├── benchmarks/              # Performance benchmarks
├── scripts/                 # Utility scripts
└── archive/                 # Historical docs
```

## Testing

### Running Tests
```bash
# All tests (900+ tests)
pytest tests/ -v

# Specific domain
pytest tests/test_audio_basic.py -v
pytest tests/test_field_operations.py -v

# With coverage
pytest tests/ --cov=morphogen --cov-report=html
open htmlcov/index.html

# Parallel execution (faster)
pytest tests/ -n auto

# Test markers
pytest tests/ -m "not slow"           # Skip slow tests
pytest tests/ -m determinism          # Only determinism tests
pytest tests/ -m integration          # Only integration tests
```

### Test Structure
- `test_*_basic.py` - Core functionality for each domain
- `test_*_operations.py` - Operator tests
- `test_*_integration.py` - End-to-end tests
- `test_*_dialect.py` - MLIR compilation tests

### Key Test Markers
- `slow` - Tests taking >1 second
- `determinism` - Verifies reproducibility
- `integration` - Cross-domain tests
- `benchmark` - Performance tests

## Common Development Commands

### Language Development
```bash
# Run a Morphogen program
morphogen run examples/01_hello_heat.kairo

# Parse and show AST
morphogen parse examples/01_hello_heat.kairo

# Type check
morphogen check examples/01_hello_heat.kairo

# Generate MLIR IR (text-based)
morphogen mlir examples/01_hello_heat.kairo
```

### Code Quality
```bash
# Format code
black morphogen/ tests/

# Lint
ruff check morphogen/

# Type check
mypy morphogen/ --ignore-missing-imports

# Run all checks
black morphogen/ && ruff check morphogen/ && pytest tests/ -v
```

### Performance
```bash
# Run benchmarks
pytest benchmarks/ -v

# Profile a specific example
python -m cProfile -s cumtime examples/field_ops/heat_diffusion.py
```

## Performance Characteristics

### Current Implementation (Python/NumPy)
- **Field Operations** (256×256): ~10-50ms per timestep
- **Agent Systems** (1000 agents): ~5-20ms per timestep with spatial hashing
- **Audio Synthesis** (44.1kHz): Real-time capable for most operations
- **Parsing**: ~10-50ms for typical programs
- **Memory**: ~100MB base + domain-specific allocations

### Known Bottlenecks
- Large grids (>512×512) become slow without GPU acceleration
- Python interpreter overhead (MLIR compilation will address this)
- No parallelization yet (single-threaded execution)
- Field operations use iterative solvers (Jacobi, CG) which can be slow for large systems

### Optimization Tips
- Use smaller grid sizes during development (128×128 vs 512×512)
- Reduce iteration counts for diffusion/projection (20 iterations is often sufficient)
- Use spatial hashing for agent forces (`compute_pairwise_forces` with `radius` parameter)
- Profile with `cProfile` to identify bottlenecks
- Consider lowering time resolution (larger `dt`) if physics allows

## Troubleshooting

### Common Issues

**Import errors after installation**
```bash
# Ensure editable install
pip install -e .

# Verify package location
python -c "import morphogen; print(morphogen.__file__)"
```

**Audio playback not working**
```bash
# Install audio dependencies
pip install -e ".[audio]"

# Test audio backend
python -c "import sounddevice; print(sounddevice.query_devices())"
```

**Tests failing**
```bash
# Update dependencies
pip install -e ".[dev]"

# Check Python version (requires >=3.9)
python --version

# Run with verbose output
pytest tests/test_audio_basic.py -v -s
```

**MLIR import errors**
```bash
# MLIR is optional - system falls back to Python interpreter
# To use MLIR features, install separately:
pip install mlir -f https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest
```

**Visualization not displaying**
```bash
# Install visualization dependencies
pip install -e ".[viz]"

# For headless systems, export to file instead:
# output(vis, "output.png")  instead of display(vis)
```

### Getting Help
- Check `docs/troubleshooting.md` for detailed solutions
- Search issues: https://github.com/scottsen/morphogen/issues
- Review examples in `examples/` directory for working patterns

## Tools Available

### reveal.sh / reveal.py - Progressive File Explorer

**Purpose**: Explore large files incrementally to manage token usage and understand structure before diving into full content.

**Location**: `scripts/reveal.sh` (wrapper) and `scripts/reveal.py` (local Python implementation)

**Usage**:
```bash
# Level 0: Metadata only (filename, size, type, line count, hash)
./scripts/reveal.sh 0 morphogen/stdlib/audio.py

# Level 1: Structure (imports, classes, functions for Python files)
./scripts/reveal.sh 1 morphogen/stdlib/audio.py

# Level 2: Preview (representative sample with context)
./scripts/reveal.sh 2 docs/specifications/chemistry.md

# Level 3: Full content (with line numbers)
./scripts/reveal.sh 3 SPECIFICATION.md

# Direct Python usage
python scripts/reveal.py 1 morphogen/stdlib/audio.py
```

**When to Use reveal**:
- **Before reading large files**: Check structure at level 1 before committing to full read
- **Domain exploration**: Survey what's in a domain module (imports, classes, functions)
- **Documentation navigation**: Get markdown structure before reading 2000+ line specs
- **Large test files**: Preview test structure without loading all test code
- **Token conservation**: Get 80% of the information at 20% of the token cost

**File Type Support**:
- **Python**: AST analysis (imports, classes, functions, docstrings)
- **Markdown**: Heading hierarchy, code blocks, lists
- **Text**: Generic line/word counts

**Documentation**: See `scripts/README.md` for complete usage guide

## Working with Morphogen

### Common Tasks

**1. Adding a New Domain**
See `docs/guides/domain-implementation.md` for complete guide:
1. Create domain module in `morphogen/stdlib/`
2. Define operators (functions with type signatures)
3. Add to operator registry
4. Write tests in `tests/`
5. Document in `docs/specifications/`

**2. Understanding Existing Domains**
```bash
# Get domain structure first
./scripts/reveal.sh 1 morphogen/stdlib/audio.py

# See what tests exist
./scripts/reveal.sh 1 tests/test_audio_basic.py

# Check specification
./scripts/reveal.sh 2 docs/specifications/chemistry.md
```

**3. Running Tests**
```bash
# All tests
pytest tests/

# Specific domain
pytest tests/test_audio_basic.py -v

# With coverage
pytest tests/ --cov=morphogen --cov-report=html
```

**4. Exploring Documentation**
Start with `docs/README.md` for navigation, then:
- `docs/architecture/overview.md` - System architecture
- `ECOSYSTEM_MAP.md` - All domains mapped
- `SPECIFICATION.md` - Language reference
- `docs/architecture/domain-architecture.md` - Deep domain specs (2,266 lines)

## Development Workflow

### Current Branch
Working branch varies by task/session.

### Git Practices
```bash
# Always push to the claude/* branch for the current session
git add .
git commit -m "docs: Descriptive commit message"
git push -u origin <claude-branch-name>
```

### Before Making Changes
1. **Check structure** with `./scripts/reveal.sh 1 <file>` first
2. **Read selectively** using Read tool for specific files
3. **Search strategically** with Grep for patterns
4. **Test changes** with pytest
5. **Document updates** in relevant specs/guides

## Key Architectural Concepts

### Temporal Model
Programs use `flow` blocks for time evolution:
```morphogen
@state temp : Field2D<f32 [K]> = zeros((128, 128))

flow(dt=0.01, steps=1000) {
    temp = diffuse(temp, rate=0.1, dt)
    output colorize(temp, palette="fire")
}
```

### Deterministic RNG
All randomness explicit via RNG objects:
```morphogen
@state agents : Agents<Particle> = alloc(count=100, init=spawn_random)

fn spawn_random(id: u32, rng: RNG) -> Particle {
    return Particle {
        pos: rng.uniform_vec2(min=(0, 0), max=(100, 100))
    }
}
```

### Cross-Domain Composition
Domains work together seamlessly:
```morphogen
use fluid, acoustics, audio

@state flow : FluidNetwork1D = engine_exhaust(length=2.5m)
@state acoustic : AcousticField1D = waveguide_from_flow(flow)

flow(dt=0.1ms) {
    flow = flow.advance(engine_pulse(t))
    acoustic = acoustic.couple_from_fluid(flow)
    let sound = acoustic.to_audio(mic_position=1.5m)
    audio.play(sound)
}
```

## Strategic Context

### Professional Applications
- **Education**: Replace MATLAB for computational physics/engineering
- **Digital Twins**: Unified multi-physics simulation (automotive, aerospace)
- **Audio Production**: Physical modeling for virtual instruments
- **Scientific Computing**: Multi-domain research without tool fragmentation
- **Creative Coding**: Deterministic generative art + audio + physics

### Long-Term Vision
- GPU acceleration via MLIR GPU dialect
- JIT compilation for live performance
- Advanced optimizations (fusion, vectorization, polyhedral)
- Geometry/CAD integration (TiaCAD-inspired reference system)
- Symbolic math and control theory domains

## Sister Project: Philbrick

**Philbrick** is the analog/hybrid hardware counterpart to Morphogen (digital software). Both share the same four core operations (sum, integrate, nonlinearity, events) and compositional philosophy.

- **Morphogen**: Digital simulation of continuous phenomena
- **Philbrick**: Physical embodiment of continuous dynamics
- **Bridge**: Design in Morphogen → Build in Philbrick → Validate together

## Quick Reference

### Domain Cheat Sheet

**Field Operations** (`morphogen/stdlib/field.py`)
```python
field.alloc(shape, fill=0.0)                    # Create field
field.diffuse(field, rate, dt, iterations=20)    # Heat diffusion
field.advect(field, velocity, dt)                # Advection
field.project(velocity, iterations=50)           # Pressure projection
field.laplacian(field)                           # Laplacian operator
field.gradient(field)                            # Gradient
```

**Agent Operations** (`morphogen/stdlib/agents.py`)
```python
agents.alloc(count, properties={...})            # Create agents
agents.map(agents, property, func)               # Transform property
agents.filter(agents, property, condition)       # Filter agents
agents.compute_pairwise_forces(...)              # N-body forces
agents.sample_field(agents, field, property)     # Field coupling
```

**Audio Synthesis** (`morphogen/stdlib/audio.py`)
```python
audio.sine(freq, duration, amplitude=0.5)        # Sine wave
audio.string(excitation, freq, t60)              # Physical string
audio.lowpass(signal, cutoff)                    # Filter
audio.reverb(signal, mix=0.3, size=0.8)         # Reverb
audio.play(buffer)                               # Real-time playback
audio.save(buffer, "out.wav")                    # Export audio
```

**Visual Operations** (`morphogen/stdlib/visual.py`)
```python
visual.colorize(field, palette="viridis")        # Field → RGB
visual.agents(agents, width, height, ...)        # Render agents
visual.composite(layers, mode="add")             # Layer composition
visual.video(frames, "out.mp4", fps=30)         # Export video
visual.output(visual, "out.png")                 # Save image
```

**Graph/Network** (`morphogen/stdlib/graph.py`)
```python
graph.create_empty(directed=False)               # New graph
graph.add_edge(graph, source, target, weight)    # Add edge
graph.shortest_path(graph, source, target)       # Dijkstra
graph.degree_centrality(graph)                   # Centrality
graph.connected_components(graph)                # Components
```

**Optimization** (`morphogen/stdlib/optimization.py`)
```python
genetic.create_population(size, gene_length)     # GA population
genetic.evolve(population, fitness_func)         # Evolution step
optimization.cma_es(objective, x0, sigma0)       # CMA-ES
```

### Important Files
- `README.md` - Project overview and getting started
- `SPECIFICATION.md` - Complete language reference
- `ECOSYSTEM_MAP.md` - All domains mapped
- `STATUS.md` - Current implementation status
- `docs/architecture/overview.md` - System architecture
- `docs/architecture/domain-architecture.md` - Deep domain specs (2,266 lines)
- `docs/specifications/ambient-music.md` - Audio DSL specification
- `docs/DOMAIN_VALUE_ANALYSIS.md` - Strategic analysis

### Running Examples
```bash
# Heat diffusion
python examples/field_ops/heat_diffusion.py

# Audio synthesis
python examples/audio/karplus_strong.py

# Rigid body physics
python examples/rigidbody_physics/bouncing_balls.py

# Agent simulation
python examples/agents/boids.py
```

### Test Coverage
- **900+ tests** across 55 test files
- Full domain coverage for all production domains
- Integration tests for cross-domain composition

---

**Last Updated**: 2025-11-21
**Status**: v0.11.0 - Complete Multi-Domain Platform (40 domains, 500+ operators)
**Contributors**: Added comprehensive installation, testing, performance, and troubleshooting guides
