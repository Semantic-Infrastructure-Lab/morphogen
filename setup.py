"""Setup configuration for Morphogen."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="morphogen",
    version="0.12.0",
    author="Scott Sen",
    description="A language of creative determinism for simulation, sound, and visual form",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Multimedia :: Sound/Audio :: Sound Synthesis",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",    # For ndimage operations
        "pillow>=9.0.0",   # For visual output (MVP requirement)
        "pygame>=2.0.0",   # For interactive visualization
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "mypy>=1.0.0",
            "ruff>=0.1.0",
        ],
        "mlir": [
            "mlir-python-bindings>=17.0.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "pillow>=9.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "morphogen=morphogen.cli:main",
        ],
    },
)
