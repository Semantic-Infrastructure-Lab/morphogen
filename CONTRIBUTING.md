# Contributing to Morphogen

Thank you for your interest in contributing to Morphogen! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [High-Impact Areas](#high-impact-areas)
- [Contribution Workflow](#contribution-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Questions?](#questions)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

Morphogen is building toward something transformative: a universal platform where professional domains that have never talked before can seamlessly compose. Contributions welcome at all levels!

**Before contributing:**

1. **Explore the architecture** — Read [docs/architecture/overview.md](docs/architecture/overview.md) and [docs/architecture/domain-architecture.md](docs/architecture/domain-architecture.md)
2. **Understand the vision** — Check [README.md](README.md) and [docs/philosophy/](docs/philosophy/)
3. **Review existing work** — Look at [open issues](https://github.com/scottsen/morphogen/issues) and [pull requests](https://github.com/scottsen/morphogen/pulls)
4. **Check the status** — See [STATUS.md](STATUS.md) for current implementation status

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- (Optional) MLIR Python bindings for advanced features

### Installation

```bash
# Clone the repository
git clone https://github.com/scottsen/morphogen.git
cd morphogen

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Optional: Install I/O dependencies for audio/visual work
pip install soundfile sounddevice opencv-python pillow
```

### Verify Installation

```bash
# Run tests
pytest tests/ -v

# Check version
python -c "import morphogen; print(morphogen.__version__)"
```

## High-Impact Areas

### 1. Domain Expansion

Help implement new domains to expand Morphogen's capabilities:

- **Geometry/CAD integration** — TiaCAD-inspired reference system for CAD workflows
- **Chemistry & molecular dynamics** — Molecular simulation and analysis
- **Graph/network analysis** — Network algorithms and analysis tools
- **Neural operator support** — Physics-informed neural networks
- **Fluid dynamics** — CFD and flow simulation
- **Circuit simulation** — Electronic circuit modeling

**Resources:**
- [Domain Implementation Guide](docs/guides/domain-implementation.md)
- [Domain Architecture](docs/architecture/domain-architecture.md)
- [Existing domain specifications](docs/specifications/)

### 2. Core Infrastructure

Strengthen Morphogen's foundation:

- **MLIR lowering passes and optimization** — Improve compilation and performance
- **GPU acceleration** — Extend GPU support for field operations
- **Multi-GPU support** — Distributed execution across multiple GPUs
- **Cross-domain type checking** — Enhance type safety across domain boundaries
- **Unit validation** — Physical unit checking and conversion
- **Scheduler improvements** — Multi-rate scheduling and temporal semantics

**Resources:**
- [MLIR Dialects specification](docs/specifications/mlir-dialects.md)
- [Type System specification](docs/specifications/type-system.md)
- [Level 3 Type System](docs/specifications/level-3-type-system.md)

### 3. Professional Applications

Build real-world examples and workflows:

- **Engineering workflows** — CAD → FEA → optimization pipelines
- **Scientific computing** — Multi-physics simulations
- **Audio production** — Lutherie, timbre extraction, physical modeling
- **Creative coding** — Generative art, live visuals, procedural generation

**Resources:**
- [Examples directory](examples/)
- [Use cases](docs/use-cases/)
- [Cross-domain examples](docs/examples/)

### 4. Documentation & Education

Improve documentation and educational resources:

- **Tutorials** — Domain-specific tutorials
- **Professional field guides** — Best practices for specific industries
- **Implementation examples** — Working examples demonstrating features
- **Performance benchmarks** — Benchmarking and optimization guides
- **API documentation** — Docstring improvements and API docs

## Contribution Workflow

### 1. Fork and Branch

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/morphogen.git
cd morphogen
git remote add upstream https://github.com/scottsen/morphogen.git

# Create a feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write code following the [Code Style](#code-style) guidelines
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Commit

```bash
# Stage your changes
git add .

# Commit with descriptive message
git commit -m "feat(domain): Add new chemistry domain operators

- Implement molecular geometry optimization
- Add thermodynamic property calculations
- Include 15 comprehensive tests
- Update domain documentation"
```

**Commit message format:**
- Use conventional commits: `type(scope): description`
- Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`
- Scope: domain name, component, or area affected
- Keep first line under 72 characters
- Add detailed description in body if needed

### 4. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
```

**Pull request guidelines:**
- Provide clear description of changes
- Reference related issues (e.g., "Closes #123")
- Include screenshots/examples if applicable
- Ensure CI passes
- Request review from maintainers

## Code Style

### Python Code

- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints where applicable
- Maximum line length: 100 characters (soft limit)
- Use meaningful variable and function names
- Add docstrings for public functions and classes

**Example:**

```python
def diffuse(field: Field2D, rate: float, dt: float) -> Field2D:
    """Apply diffusion to a 2D field using finite differences.

    Args:
        field: Input field to diffuse
        rate: Diffusion coefficient (must be positive)
        dt: Time step

    Returns:
        Diffused field

    Raises:
        ValueError: If rate is negative
    """
    if rate < 0:
        raise ValueError(f"Diffusion rate must be positive, got {rate}")

    # Implementation...
    return diffused_field
```

### Morphogen Language Code

- Use clear, descriptive variable names
- Add comments for complex logic
- Follow existing examples in [examples/](examples/)
- Use appropriate physical units in annotations

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_field.py -v

# Run tests with coverage
pytest tests/ --cov=morphogen --cov-report=html
```

### Writing Tests

- Add tests for all new functionality
- Use descriptive test names: `test_<function>_<scenario>_<expected_result>`
- Test edge cases and error conditions
- Aim for >80% code coverage

**Example:**

```python
def test_diffuse_with_positive_rate_returns_smoothed_field():
    """Test that diffusion with positive rate smooths the field."""
    field = np.random.rand(64, 64)
    rate = 0.1
    dt = 0.01

    result = diffuse(field, rate, dt)

    # Check that variance decreased (smoothing effect)
    assert np.var(result) < np.var(field)

def test_diffuse_with_negative_rate_raises_value_error():
    """Test that negative diffusion rate raises ValueError."""
    field = np.random.rand(64, 64)

    with pytest.raises(ValueError, match="rate must be positive"):
        diffuse(field, rate=-0.1, dt=0.01)
```

## Documentation

### Code Documentation

- Add docstrings to all public functions, classes, and modules
- Use Google-style docstrings
- Include usage examples in docstrings where helpful

### Project Documentation

- Update relevant docs in `docs/` when adding features
- Follow existing documentation structure
- Use markdown for all documentation
- Add cross-references to related docs

**When to update documentation:**

- Adding a new domain → Update `docs/specifications/` and `STATUS.md`
- Adding language features → Update `SPECIFICATION.md`
- Architectural changes → Update `docs/architecture/`
- Breaking changes → Update `CHANGELOG.md`

## Questions?

- **General questions** — Open a [GitHub Discussion](https://github.com/scottsen/morphogen/discussions)
- **Bug reports** — Open an [issue](https://github.com/scottsen/morphogen/issues)
- **Feature requests** — Open an issue with the "enhancement" label
- **Security vulnerabilities** — See [SECURITY.md](SECURITY.md)

## Getting Help

Resources for contributors:

- [Architecture documentation](docs/architecture/)
- [Domain implementation guide](docs/guides/domain-implementation.md)
- [Specification](SPECIFICATION.md)
- [ADRs (Architectural Decision Records)](docs/adr/)
- [Existing issues](https://github.com/scottsen/morphogen/issues)

## Recognition

Contributors will be recognized in:
- Git commit history
- Release notes
- Future CONTRIBUTORS.md file

Thank you for contributing to Morphogen! Together we're building a platform that unifies domains that have never talked before.
