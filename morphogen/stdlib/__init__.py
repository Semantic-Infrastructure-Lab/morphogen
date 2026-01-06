"""Standard library implementations for Creative Computation DSL."""

# Core domains
from .field import field, Field2D, FieldOperations
from .visual import visual, Visual, VisualOperations
from .visual3d import visual3d, Visual3D, Camera3D, Light3D, Scene3D, Visual3DOperations

# Physics domains
from .acoustics import (
    acoustics, AcousticsOperations, PipeGeometry, WaveguideNetwork,
    ReflectionCoeff, FrequencyResponse, create_pipe, create_expansion_chamber
)
from . import rigidbody

# Fluid dynamics and multi-physics domains
from .fluid_network import fluid_network, FluidNetworkOperations, Fluid, Tube, Junction, FlowNet
from .thermal_ode import thermal_ode, ThermalODEOperations, ThermalSegment, WallTempModel, ThermalProfile
from .fluid_jet import fluid_jet, FluidJetOperations, Jet, JetArray
from .combustion_light import combustion_light, CombustionLightOperations, MixtureState, CombustionZone, SmokeIndex

# Audio analysis and instrument modeling domains
from .audio_analysis import audio_analysis, AudioAnalysisOperations, ModalModel, NoiseModel, ExcitationModel
from .instrument_model import instrument, InstrumentModelOperations, InstrumentModel, InstrumentType, SynthParams

# Chemistry domains
from . import molecular
from . import kinetics
from . import qchem
from . import transport
from . import multiphase
from . import thermo
from . import catalysis
from . import electrochem

# Base-level domains
from . import integrators
from . import io_storage
from . import sparse_linalg
from .flappy import flappy, Bird, Pipe, GameState, FlappyOperations
from .neural import neural, DenseLayer, MLP, NeuralOperations
from .genetic import genetic, Individual, Population, GeneticOperations

# Agent-based domains
from . import agents
from . import temporal

# Audio domains
from . import audio
from . import signal

# Geometry domains
from . import geometry
from . import graph

# Optimization domains
from . import optimization
from . import statemachine

# Terrain and vision domains
from . import terrain
from . import vision

# Procedural graphics domains
from .noise import noise, NoiseField2D, NoiseField3D, NoiseOperations
from .palette import palette, Palette, PaletteOperations
from .color import color, ColorOperations
from .image import image, Image, ImageOperations
from .cellular import cellular, CellularField2D, CellularField1D, CellularOperations, CARule

__all__ = [
    # Core domains
    "field", "Field2D", "FieldOperations",
    "visual", "Visual", "VisualOperations",
    "visual3d", "Visual3D", "Camera3D", "Light3D", "Scene3D", "Visual3DOperations",

    # Physics domains
    "acoustics", "AcousticsOperations", "PipeGeometry", "WaveguideNetwork",
    "ReflectionCoeff", "FrequencyResponse", "create_pipe", "create_expansion_chamber",
    "rigidbody",

    # Fluid dynamics and multi-physics domains
    "fluid_network", "FluidNetworkOperations", "Fluid", "Tube", "Junction", "FlowNet",
    "thermal_ode", "ThermalODEOperations", "ThermalSegment", "WallTempModel", "ThermalProfile",
    "fluid_jet", "FluidJetOperations", "Jet", "JetArray",
    "combustion_light", "CombustionLightOperations", "MixtureState", "CombustionZone", "SmokeIndex",

    # Audio analysis and instrument modeling domains
    "audio_analysis", "AudioAnalysisOperations", "ModalModel", "NoiseModel", "ExcitationModel",
    "instrument", "InstrumentModelOperations", "InstrumentModel", "InstrumentType", "SynthParams",

    # Chemistry domains
    "molecular",
    "kinetics",
    "qchem",
    "transport",
    "multiphase",
    "thermo",
    "catalysis",
    "electrochem",

    # Base-level domains
    "integrators",
    "io_storage",
    "sparse_linalg",
    "flappy", "Bird", "Pipe", "GameState", "FlappyOperations",
    "neural", "DenseLayer", "MLP", "NeuralOperations",
    "genetic", "Individual", "Population", "GeneticOperations",

    # Agent-based domains
    "agents",
    "temporal",

    # Audio domains
    "audio",
    "signal",

    # Geometry domains
    "geometry",
    "graph",

    # Optimization domains
    "optimization",
    "statemachine",

    # Terrain and vision domains
    "terrain",
    "vision",

    # Procedural graphics domains
    "noise", "NoiseField2D", "NoiseField3D", "NoiseOperations",
    "palette", "Palette", "PaletteOperations",
    "color", "ColorOperations",
    "image", "Image", "ImageOperations",
    "cellular", "CellularField2D", "CellularField1D", "CellularOperations", "CARule",
]
