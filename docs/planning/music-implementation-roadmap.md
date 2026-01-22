---
project: morphogen
type: roadmap
status: active
date: 2025-12-10
keywords:
  - implementation
  - music
  - roadmap
  - planning
beth_topics:
  - morphogen-music
  - implementation-plan
---

# 🎵 Music Stack Implementation Roadmap

**Detailed implementation plan for 7-layer music architecture**

Based on: **[ADR-013: Music Stack Consolidation](adr/013-music-stack-consolidation.md)**

Last Updated: 2025-12-10

---

## 📊 Overview

### Timeline

```
2025 Q1: Phase 1 (Feature Layer)
2025 Q2: Phase 2 (Symbolic Layer) + Phase 3 (Structural Layer)
2025 Q3: Phase 4 (Compositional Layer) + Phase 5 (RiffStack Integration)
2025 Q4: Optimization, Documentation, Release
```

### Current Status

**Existing (Complete)**:
- ✅ Layer 1 (Physical): Audio buffers, I/O
- ✅ Layer 2 (DSP): 45+ operations (oscillators, filters, effects)
- ⚠️ Layer 3 (Spectral): FFT, STFT exist (need HPSS)
- ⚠️ Layer 4 (Feature): Basic spectral features (need mel, chroma)

**To Implement**:
- ❌ Layer 4 (Feature): Mel, chroma, HPSS, SSM
- ❌ Layer 5 (Symbolic): Beat tracking, chord recognition, key detection
- ❌ Layer 6 (Structural): Segmentation, section labeling
- ❌ Layer 7 (Compositional): Voice-leading, progressions

---

## 🎯 Phase 1: Feature Layer (Q1 2025)

**Goal**: Complete Layer 4 (Perceptual/Musical Features)

**Dependencies**: Existing Layer 3 (FFT/STFT in `stdlib/audio.py`)

### 1.1 Create `stdlib/audio_features.py`

**New operations to implement**:

```python
# Mel-frequency spectrogram
def melspectrogram(
    signal: AudioBuffer,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    window_size: int = 2048,
    hop_size: int = 512
) -> np.ndarray:
    """
    Convert audio to mel-frequency spectrogram.

    Maps frequency to mel scale (perceptual).
    Returns: (n_mels, n_frames) array
    """
    pass

# Chroma features (pitch class representation)
def chroma(
    signal: AudioBuffer,
    n_chroma: int = 12,
    window_size: int = 4096,
    hop_size: int = 512
) -> np.ndarray:
    """
    Extract chroma (pitch class) features.

    12 bins representing C, C#, D, ..., B
    Returns: (12, n_frames) array
    """
    pass

# Harmonic-Percussive Source Separation
def hpss(
    signal: AudioBuffer,
    kernel_size: Tuple[int, int] = (17, 17),
    power: float = 2.0,
    margin: float = 1.0
) -> Tuple[AudioBuffer, AudioBuffer]:
    """
    Separate harmonic and percussive components.

    Returns: (harmonic, percussive) audio buffers
    """
    pass

# Self-Similarity Matrix (for structure)
def self_similarity_matrix(
    features: np.ndarray,
    metric: str = "cosine"
) -> np.ndarray:
    """
    Compute self-similarity matrix from features.

    Used for structural segmentation.
    Returns: (n_frames, n_frames) similarity matrix
    """
    pass
```

**Implementation approach**:
- Use `librosa` for reference implementations
- Optimize with NumPy/SciPy
- Match existing `stdlib/audio.py` patterns (decorators, types)
- Comprehensive docstrings

**Tests** (`tests/test_audio_features.py`):
```python
def test_melspectrogram():
    # Test mel conversion
    # Test shape (n_mels, n_frames)
    # Test frequency range
    pass

def test_chroma():
    # Test with known pitches
    # Test 12-bin output
    pass

def test_hpss():
    # Test separation quality
    # Test with harmonic+percussive signal
    pass

def test_ssm():
    # Test symmetry
    # Test known repetitions
    pass
```

**Documentation**:
- Add operations to `docs/DOMAINS.md`
- Add examples to `examples/audio_features/`
- Update `docs/MUSIC_DOCUMENTATION_INDEX.md`

**Deliverables**:
- [ ] `morphogen/stdlib/audio_features.py` (~500 lines)
- [ ] `morphogen/tests/test_audio_features.py` (~300 lines)
- [ ] `examples/audio_features/mel_visualization.py`
- [ ] `examples/audio_features/chroma_analysis.py`
- [ ] `examples/audio_features/hpss_demo.py`

**Estimated effort**: 2-3 weeks

---

### 1.2 Create MLIR `features` dialect

**File**: `morphogen/mlir/dialects/features.py`

**Types to define**:
```mlir
!features.mel
!features.chroma
!features.mfcc
!features.ssm
```

**Operations to define**:
```mlir
features.melspectrogram
features.chroma
features.hpss
features.ssm
features.spectral_centroid  # Already in audio.py, migrate
features.spectral_rolloff
features.spectral_flux
```

**Lowering pass** (`mlir/lowering/features_to_linalg.py`):
- Lower to `linalg` + `math` + `arith`
- Vectorization opportunities
- Match pattern in `audio_to_scf.py`

**Deliverables**:
- [ ] `morphogen/mlir/dialects/features.py` (~400 lines)
- [ ] `morphogen/mlir/lowering/features_to_linalg.py` (~500 lines)
- [ ] Unit tests for MLIR ops

**Estimated effort**: 2-3 weeks

---

### 1.3 Integration & Testing

**Integration tests**:
```python
def test_full_feature_extraction():
    audio = morphogen.audio.load("test.wav")
    mel = morphogen.audio_features.melspectrogram(audio)
    chroma = morphogen.audio_features.chroma(audio)
    harmonic, percussive = morphogen.audio_features.hpss(audio)
    ssm = morphogen.audio_features.self_similarity_matrix(chroma)

    assert mel.shape == (128, n_frames)
    assert chroma.shape == (12, n_frames)
    assert ssm.shape == (n_frames, n_frames)
```

**MLIR compilation test**:
```python
def test_features_mlir_compilation():
    # Compile feature extraction to LLVM
    # Verify correctness vs Python
    # Benchmark performance
    pass
```

**Deliverables**:
- [ ] Integration tests
- [ ] Performance benchmarks
- [ ] Documentation examples

**Estimated effort**: 1 week

---

**Phase 1 Total**: 5-7 weeks

---

## 🎼 Phase 2: Symbolic Layer (Q1-Q2 2025)

**Goal**: Implement Layer 5 (Music Theory - Notes, Chords, Beats)

**Dependencies**: Phase 1 (Feature Layer)

### 2.1 Create `stdlib/music_symbolic.py`

**New operations to implement**:

```python
# Beat tracking
def beat_track(
    signal: AudioBuffer,
    onset_envelope: Optional[np.ndarray] = None,
    start_bpm: float = 120.0,
    tightness: float = 100.0
) -> BeatGrid:
    """
    Detect beats in audio.

    Returns: BeatGrid with beat times and downbeats
    """
    pass

# Tempo estimation
def estimate_tempo(
    signal: AudioBuffer,
    start_bpm: float = 120.0,
    aggregate: Optional[Callable] = np.median
) -> Tuple[float, float]:
    """
    Estimate global tempo (BPM).

    Returns: (tempo, confidence)
    """
    pass

# Chord recognition
def chord_estimate(
    chroma: np.ndarray,
    beat_grid: Optional[BeatGrid] = None,
    model: str = "crema"  # or "deepchroma", "madmom"
) -> ChordSequence:
    """
    Estimate chords from chroma features.

    Returns: ChordSequence with chord labels and timings
    """
    pass

# Key detection
def key_estimate(
    chroma: np.ndarray,
    method: str = "krumhansl"  # or "temperley", "edma"
) -> Key:
    """
    Estimate musical key.

    Returns: Key (tonic + mode)
    """
    pass

# Onset detection
def onset_detect(
    signal: AudioBuffer,
    hop_size: int = 512,
    backtrack: bool = True
) -> np.ndarray:
    """
    Detect note onsets (attacks).

    Returns: Array of onset times (in seconds)
    """
    pass
```

**New types**:
```python
@dataclass
class BeatGrid:
    times: np.ndarray      # Beat times (seconds)
    downbeats: np.ndarray  # Downbeat indices
    tempo: float           # Average BPM
    confidence: float

@dataclass
class Chord:
    root: str              # "C", "D", "Eb", etc.
    quality: str           # "maj", "min", "dim", "aug", "sus", etc.
    extensions: List[str]  # ["7", "9", "11"], etc.
    bass: Optional[str]    # Slash chord bass note

@dataclass
class ChordSequence:
    chords: List[Chord]
    times: np.ndarray      # Chord change times
    beat_aligned: bool     # Whether aligned to beat grid

@dataclass
class Key:
    tonic: str             # "C", "D", "Eb", etc.
    mode: str              # "major", "minor", "dorian", etc.
    confidence: float
```

**Implementation approach**:
- Use `librosa` for beat tracking
- Integrate pre-trained chord recognition models (CREMA, DeepChroma)
- Implement Krumhansl-Schmuckler key detection
- Match existing Morphogen patterns

**Deliverables**:
- [ ] `morphogen/stdlib/music_symbolic.py` (~800 lines)
- [ ] `morphogen/tests/test_music_symbolic.py` (~400 lines)
- [ ] `examples/music_symbolic/beat_tracking_demo.py`
- [ ] `examples/music_symbolic/chord_recognition_demo.py`

**Estimated effort**: 4-5 weeks

---

### 2.2 Neural Model Integration (StableHLO)

**Goal**: Integrate neural models for beat/chord/section detection

**Create `morphogen/models/` directory**:

```
morphogen/models/
├── __init__.py
├── backbone.py         # Conformer/Transformer backbone
├── heads/
│   ├── __init__.py
│   ├── beat.py        # Beat detection head
│   ├── chord.py       # Chord recognition head
│   └── section.py     # Section labeling head
└── stablehlo_bridge.py  # StableHLO integration
```

**Example usage**:
```python
from morphogen.models import backbone, heads

# Feature extraction
mel = morphogen.audio_features.melspectrogram(audio)

# Neural backbone (Conformer)
features = backbone.conformer(mel)

# Specialized heads
beat_probs = heads.beat.predict(features)
chord_logits = heads.chord.predict(features)

# Decode to symbolic
beats = morphogen.music_symbolic.discretize_beats(beat_probs)
chords = morphogen.music_symbolic.decode_chords(chord_logits, beats)
```

**Models to integrate**:
- Beat tracking: `madmom` DBN model or `allin2015` model
- Chord recognition: CREMA or DeepChroma
- Optionally: Train custom models on proprietary datasets

**StableHLO integration**:
- Convert TensorFlow/PyTorch models to StableHLO
- MLIR compilation path for neural models
- GPU optimization

**Deliverables**:
- [ ] `morphogen/models/backbone.py` (~600 lines)
- [ ] `morphogen/models/heads/*.py` (~300 lines each)
- [ ] `morphogen/models/stablehlo_bridge.py` (~400 lines)
- [ ] Pre-trained model weights (checkpoint files)
- [ ] Model documentation and usage guide

**Estimated effort**: 6-8 weeks (can overlap with 2.1)

---

### 2.3 Create MLIR `symbolic` dialect

**File**: `morphogen/mlir/dialects/symbolic.py`

**Types**:
```mlir
!symbolic.pitch
!symbolic.note
!symbolic.chord
!symbolic.beat_grid
!symbolic.key
!symbolic.chord_sequence
```

**Operations**:
```mlir
symbolic.beat_track
symbolic.estimate_tempo
symbolic.chord_estimate
symbolic.key_estimate
symbolic.onset_detect
symbolic.discretize_beats
symbolic.decode_chords
```

**Deliverables**:
- [ ] `morphogen/mlir/dialects/symbolic.py` (~500 lines)
- [ ] `morphogen/mlir/lowering/symbolic_to_linalg.py` (~400 lines)

**Estimated effort**: 3-4 weeks

---

**Phase 2 Total**: 13-17 weeks (overlapping work)

---

## 🏛️ Phase 3: Structural Layer (Q2 2025)

**Goal**: Implement Layer 6 (Musical Form - Sections, Phrases)

**Dependencies**: Phase 1 (Feature Layer)

### 3.1 Create `stdlib/music_structural.py`

**Operations**:

```python
def segment_structure(
    ssm: np.ndarray,
    kernel_size: int = 65,
    method: str = "novelty"  # or "checkerboard", "laplacian"
) -> np.ndarray:
    """
    Find section boundaries from self-similarity matrix.

    Returns: Array of boundary frame indices
    """
    pass

def label_sections(
    boundaries: np.ndarray,
    features: np.ndarray,  # chroma, MFCC, etc.
    method: str = "clustering"  # or "sequence_model"
) -> SectionSequence:
    """
    Label sections (verse, chorus, bridge, etc.).

    Returns: SectionSequence with labels and timings
    """
    pass

def detect_novelty(
    features: np.ndarray,
    kernel_size: int = 9
) -> np.ndarray:
    """
    Detect structural novelty (change points).

    Returns: Novelty curve (time series)
    """
    pass

def find_repetitions(
    ssm: np.ndarray,
    min_length: int = 8
) -> List[Tuple[int, int, int, int]]:
    """
    Find repeated sections in SSM.

    Returns: List of (start1, end1, start2, end2) tuples
    """
    pass
```

**New types**:
```python
@dataclass
class Section:
    label: str             # "verse", "chorus", "bridge", "intro", etc.
    start_time: float
    end_time: float
    confidence: float

@dataclass
class SectionSequence:
    sections: List[Section]
    form: str              # "AABA", "verse-chorus", etc.

@dataclass
class Form:
    structure: str         # "AABA", "ABAC", etc.
    sections: Dict[str, List[Section]]  # "A" -> [Section, Section, ...]
```

**Implementation approach**:
- Use `librosa.segment` for segmentation
- Implement novelty-based and checkerboard methods
- K-means clustering for section labeling
- Repetition detection via diagonal stripe finding in SSM

**Deliverables**:
- [ ] `morphogen/stdlib/music_structural.py` (~600 lines)
- [ ] `morphogen/tests/test_music_structural.py` (~300 lines)
- [ ] `examples/music_structural/segmentation_demo.py`
- [ ] `examples/music_structural/form_analysis_demo.py`

**Estimated effort**: 3-4 weeks

---

### 3.2 Create MLIR `structural` dialect

**File**: `morphogen/mlir/dialects/structural.py`

**Types**:
```mlir
!structural.section
!structural.section_sequence
!structural.form
!structural.boundaries
```

**Operations**:
```mlir
structural.segment
structural.detect_novelty
structural.label_sections
structural.find_repetitions
structural.create_form
```

**Deliverables**:
- [ ] `morphogen/mlir/dialects/structural.py` (~400 lines)
- [ ] `morphogen/mlir/lowering/structural_to_linalg.py` (~300 lines)

**Estimated effort**: 2-3 weeks

---

**Phase 3 Total**: 5-7 weeks

---

## 🎨 Phase 4: Compositional Layer (Q3 2025)

**Goal**: Implement Layer 7 (Musical Intent - Voice-Leading, Progressions)

**Dependencies**: Phase 2 (Symbolic Layer)

### 4.1 Create `stdlib/music_compositional.py`

**Operations**:

```python
def voice_lead(
    progression: List[Chord],
    constraints: List[str] = ["smooth"],  # "common", "contrary", "range"
    num_voices: int = 4,
    ranges: Optional[Dict[str, Tuple[int, int]]] = None
) -> List[Voicing]:
    """
    Solve voice-leading constraints for chord progression.

    Uses constraint satisfaction (SAT/SMT solver).
    Returns: List of Voicing (concrete note assignments)
    """
    pass

def harmonize(
    melody: List[Note],
    style: str = "classical",  # "jazz", "pop", "baroque"
    num_voices: int = 4
) -> List[Voicing]:
    """
    Harmonize melody with chords.

    Returns: Voice-led harmonization
    """
    pass

def reharmonize(
    progression: List[Chord],
    style: str = "jazz",  # "modal", "chromatic", "diatonic"
    complexity: float = 0.5
) -> List[Chord]:
    """
    Substitute chords in progression.

    Returns: Reharmonized progression
    """
    pass

def generate_progression(
    key: Key,
    length: int = 4,
    style: str = "jazz",  # "classical", "pop", "modal"
    constraints: Optional[List[str]] = None
) -> List[Chord]:
    """
    Generate chord progression.

    Uses Markov chains or learned models.
    Returns: Chord progression
    """
    pass

def substitute_chord(
    chord: Chord,
    substitution_type: str = "tritone"  # "diatonic", "chromatic", "secondary_dominant"
) -> Chord:
    """
    Substitute single chord.

    Returns: Substitute chord
    """
    pass
```

**New types**:
```python
@dataclass
class Voicing:
    chord: Chord
    notes: List[Note]      # Concrete note assignments per voice
    inversion: int         # 0 = root position, 1 = first inversion, etc.
    doublings: List[str]   # Which notes are doubled

@dataclass
class Progression:
    chords: List[Chord]
    key: Key
    function: List[str]    # Roman numeral analysis: ["I", "IV", "V", "I"]

@dataclass
class VoiceLeadingConstraints:
    smooth: bool           # Minimize motion
    common: bool           # Maximize common tones
    contrary: bool         # Voices oppose bass
    range: Dict[str, Tuple[int, int]]  # Voice ranges (MIDI note numbers)
```

**Implementation approach**:
- Voice-leading: SAT solver (PySAT) or constraint programming (Python-constraint)
- Use Morphogen's group theory/topology for optimization
- Progression generation: Markov chains or LSTM
- Reference RiffStack's Harmony DSL vision

**Deliverables**:
- [ ] `morphogen/stdlib/music_compositional.py` (~1,000 lines)
- [ ] `morphogen/tests/test_music_compositional.py` (~500 lines)
- [ ] `examples/music_compositional/voice_leading_demo.py`
- [ ] `examples/music_compositional/reharmonization_demo.py`
- [ ] `examples/music_compositional/progression_generation.py`

**Estimated effort**: 6-8 weeks

---

### 4.2 Integrate RiffStack Harmony DSL Concepts

**Goal**: Import RiffStack's Harmony DSL design into compositional layer

**From RiffStack `docs/HARMONY_DSL_VISION.md`**:
- Root-motion operators (`+1`, `-3`, `T`, `5`, `=`)
- Chord color/mood (`lush`, `bright`, `dark`)
- Voice-leading policies (`smooth`, `common`, `contrary`)
- Rhythm modifiers (`sync`, `arp`, `stab`)

**Integration strategy**:
```python
# Harmony DSL operations
def move_root(chord: Chord, semitones: int) -> Chord:
    """Move chord root by semitones."""
    pass

def tritone_sub(chord: Chord) -> Chord:
    """Tritone substitution."""
    pass

def apply_mood(chord: Chord, mood: str) -> Voicing:
    """Apply voicing mood (lush, bright, dark)."""
    pass

def apply_rhythm_modifier(voicing: Voicing, modifier: str) -> Performance:
    """Apply rhythm modifier (arp, stab, roll)."""
    pass
```

**Deliverables**:
- [ ] Harmony DSL operations in `music_compositional.py`
- [ ] Examples from RiffStack docs working in Morphogen
- [ ] Migration guide from RiffStack concepts to Morphogen

**Estimated effort**: 3-4 weeks

---

### 4.3 Create MLIR `compositional` dialect

**File**: `morphogen/mlir/dialects/compositional.py`

**Types**:
```mlir
!compositional.progression
!compositional.voicing
!compositional.voice_leading
!compositional.constraints
```

**Operations**:
```mlir
compositional.voice_lead
compositional.harmonize
compositional.reharmonize
compositional.generate_progression
compositional.substitute
compositional.move_root
compositional.tritone_sub
```

**Deliverables**:
- [ ] `morphogen/mlir/dialects/compositional.py` (~600 lines)
- [ ] `morphogen/mlir/lowering/compositional_to_symbolic.py` (~500 lines)
- [ ] Integration tests

**Estimated effort**: 4-5 weeks

---

**Phase 4 Total**: 13-17 weeks

---

## 🔌 Phase 5: RiffStack Frontend Integration (Q3 2025)

**Goal**: RiffStack YAML/DSL compiles to Morphogen MLIR

**Dependencies**: Phase 4 (Compositional Layer)

### 5.1 RiffStack Parser → Morphogen IR

**Create in RiffStack**: `riffstack_core/frontend/`

```
riffstack_core/frontend/
├── __init__.py
├── parser.py           # YAML/Harmony DSL parser
├── ast.py              # Abstract Syntax Tree
├── codegen.py          # Generate Morphogen MLIR
└── validator.py        # Validate syntax/semantics
```

**Example workflow**:
```
RiffStack YAML
    ↓ (parser.py)
AST (Harmony DSL IR)
    ↓ (codegen.py)
Morphogen MLIR (compositional dialect)
    ↓ (Morphogen compiler)
Optimized executable
```

**Deliverables**:
- [ ] RiffStack parser for Harmony DSL
- [ ] AST representation
- [ ] Code generator targeting Morphogen MLIR
- [ ] Integration tests (YAML → MLIR → audio)

**Estimated effort**: 5-6 weeks

---

### 5.2 RiffStack CLI → Morphogen Runtime

**Update RiffStack CLI** to call Morphogen:

```python
# riffstack_core/cli.py

import morphogen

def play_patch(patch_path: str):
    # Parse YAML
    ast = riffstack.frontend.parser.parse(patch_path)

    # Generate Morphogen MLIR
    mlir_module = riffstack.frontend.codegen.generate_mlir(ast)

    # Compile with Morphogen
    compiled = morphogen.mlir.compiler.compile(mlir_module)

    # Execute
    morphogen.runtime.execute(compiled)
```

**Deliverables**:
- [ ] Updated RiffStack CLI
- [ ] Morphogen dependency in `requirements.txt`
- [ ] Migration guide for RiffStack users
- [ ] Performance comparison (old vs new backend)

**Estimated effort**: 3-4 weeks

---

### 5.3 Documentation & Examples

**Create**:
- [ ] Migration guide: RiffStack → Morphogen backend
- [ ] Tutorial: Writing Harmony DSL that compiles to Morphogen
- [ ] Examples: RiffStack YAML → Morphogen MLIR (side-by-side)
- [ ] Performance benchmarks

**Update**:
- [ ] RiffStack README (explain Morphogen relationship)
- [ ] Morphogen README (add music capabilities section)
- [ ] `MUSIC_DOCUMENTATION_INDEX.md` (integration complete)

**Estimated effort**: 2-3 weeks

---

**Phase 5 Total**: 10-13 weeks

---

## 📊 Summary Timeline

| Phase | Focus | Duration | Quarter |
|-------|-------|----------|---------|
| 1 | Feature Layer (mel, chroma, HPSS) | 5-7 weeks | Q1 2025 |
| 2 | Symbolic Layer (beat, chord, key) | 13-17 weeks | Q1-Q2 2025 |
| 3 | Structural Layer (segmentation) | 5-7 weeks | Q2 2025 |
| 4 | Compositional Layer (voice-leading) | 13-17 weeks | Q3 2025 |
| 5 | RiffStack Integration | 10-13 weeks | Q3 2025 |
| **Total** | | **46-61 weeks** | **Q1-Q3 2025** |

**Overlap opportunities**:
- Phase 1 & 2 can overlap (features → symbolic)
- Phase 2.1 & 2.2 can overlap (symbolic ops + neural models)
- Phase 4 & 5 can overlap (compositional ops + RiffStack integration)

**Realistic timeline with overlaps**: 30-40 weeks (~7-9 months)

---

## 🎯 Milestones

### M1: Feature Extraction Complete (End Q1)
- ✅ Mel, chroma, HPSS, SSM working
- ✅ MLIR `features` dialect
- ✅ Examples and documentation

### M2: Music Understanding Complete (End Q2)
- ✅ Beat tracking, chord recognition, key detection
- ✅ Structural segmentation
- ✅ Neural model integration (optional)
- ✅ MLIR `symbolic` + `structural` dialects

### M3: Music Generation Complete (Mid Q3)
- ✅ Voice-leading solver
- ✅ Progression generation
- ✅ Harmony DSL operations
- ✅ MLIR `compositional` dialect

### M4: RiffStack Integration Complete (End Q3)
- ✅ RiffStack compiles to Morphogen
- ✅ All examples working
- ✅ Migration guide published

### M5: Public Release (Q4)
- ✅ Documentation complete
- ✅ Performance optimized
- ✅ Tutorial series
- ✅ Blog post / announcement

---

## 📋 Dependencies & Risks

### External Dependencies

**Python packages**:
- `librosa` - Audio analysis
- `soundfile` - Audio I/O
- `numpy`, `scipy` - Numerical computing
- `scikit-learn` - Clustering for section labeling
- `pysat` or `python-constraint` - Constraint solving for voice-leading
- Optional: `madmom`, `CREMA-PDA` for neural models

**MLIR**:
- `mlir-python-bindings`
- `stablehlo` (for neural model integration)

### Risks

**High Risk**:
1. **Voice-leading constraint solving complexity**
   - Mitigation: Start with heuristic approaches, add SAT solver later
   - Fallback: Rule-based voice-leading

2. **Neural model integration (StableHLO)**
   - Mitigation: Phase 2.2 is optional, can use traditional methods
   - Fallback: Use `librosa` beat tracking instead of neural

**Medium Risk**:
3. **MLIR compilation performance**
   - Mitigation: Benchmark early, optimize incrementally
   - Fallback: Python-only implementation works

4. **RiffStack integration complexity**
   - Mitigation: Keep RiffStack minimal, focus on parser
   - Fallback: RiffStack as examples-only, not fully integrated

**Low Risk**:
5. **Documentation keeping up with implementation**
   - Mitigation: Document as we go, not at end

---

## 📈 Success Metrics

### Technical Metrics

- **Code Coverage**: >80% for all new modules
- **MLIR Compilation**: Successfully compile all 7 layers
- **Performance**: <10ms latency for real-time audio (synthesis)
- **Analysis Speed**: Process 1 hour of audio in <1 minute (analysis)

### User Metrics

- **Examples**: 20+ working examples across all layers
- **Documentation**: Complete API docs, 5+ tutorials
- **Tests**: 200+ unit tests, 50+ integration tests

---

## 🚀 Getting Started

**To begin Phase 1 today**:

1. Create feature branch:
   ```bash
   cd /home/scottsen/src/projects/morphogen
   git checkout -b feature/music-layer-4-features
   ```

2. Create stub file:
   ```bash
   touch morphogen/stdlib/audio_features.py
   ```

3. Start with `melspectrogram`:
   ```python
   # morphogen/stdlib/audio_features.py

   from morphogen.core.operator import operator, OpCategory
   from morphogen.stdlib.audio import AudioBuffer
   import numpy as np

   @operator("audio.melspectrogram", category=OpCategory.AUDIO)
   def melspectrogram(
       signal: AudioBuffer,
       n_mels: int = 128,
       # ...
   ) -> np.ndarray:
       """Convert audio to mel-frequency spectrogram."""
       # Implementation here
       pass
   ```

4. Write first test:
   ```bash
   touch morphogen/tests/test_audio_features.py
   ```

**First PR Goal**: `melspectrogram` working + tested + documented

---

## 📚 References

- **[Music Semantic Layers](architecture/MUSIC_SEMANTIC_LAYERS.md)** - Architecture
- **[ADR-013: Consolidation](adr/013-music-stack-consolidation.md)** - Decision rationale
- **[Music Documentation Index](MUSIC_DOCUMENTATION_INDEX.md)** - All docs
- **[Mathematical Music Frameworks](reference/mathematical-music-frameworks.md)** - Theory

---

**Maintainer**: TIA + Scott Senften
**Last Updated**: 2025-12-10
**Status**: Active roadmap
