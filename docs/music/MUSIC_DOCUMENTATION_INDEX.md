---
project: morphogen
type: index
status: active
date: 2025-12-10
keywords:
  - documentation
  - music
  - index
  - organization
beth_topics:
  - morphogen-music
  - documentation
---

# 🎵 Music Documentation Index

**Comprehensive guide to music-related documentation across Morphogen and RiffStack**

Last Updated: 2025-12-10

---

## 📚 Documentation Organization

### **Core Architecture**

These documents define the fundamental music architecture:

| Document | Location | Purpose | Status |
|----------|----------|---------|--------|
| **Music Semantic Layers** | `docs/architecture/MUSIC_SEMANTIC_LAYERS.md` | Defines 7-layer architecture (Physical → Compositional) | ✅ Complete |
| **ADR-013: Music Stack Consolidation** | `docs/adr/013-music-stack-consolidation.md` | Decision to consolidate music work in Morphogen | ✅ Proposed |
| **Mathematical Music Frameworks** | `docs/reference/mathematical-music-frameworks.md` | Group theory, topology, linear algebra for music | ✅ Complete |

**Read these first** to understand the overall architecture.

---

### **Implementation Documentation**

#### **Morphogen - Platform Implementation**

**Existing (Implemented)**:

| Document | Location | Content | Status |
|----------|----------|---------|--------|
| Audio Operations (stdlib) | `morphogen/stdlib/audio.py` | 60+ operations: oscillators, filters, effects, spectral | ✅ 3,105 lines |
| Audio Analysis (stdlib) | `morphogen/stdlib/audio_analysis.py` | Timbre analysis: pitch tracking, partials, modes | ✅ 696 lines |
| Audio MLIR Dialect | `morphogen/mlir/dialects/audio.py` | MLIR audio ops (synthesis-focused) | ✅ 541 lines |
| Audio Lowering | `morphogen/mlir/lowering/audio_to_scf.py` | MLIR lowering passes | ✅ 586 lines |

**To Be Created** (per ADR-013):

| Document | Location | Content | Status |
|----------|----------|---------|--------|
| Feature Extraction (stdlib) | `morphogen/stdlib/audio_features.py` | Mel, chroma, HPSS, SSM | ❌ TODO |
| Symbolic Music (stdlib) | `morphogen/stdlib/music_symbolic.py` | Beat tracking, chord recognition, key detection | ❌ TODO |
| Structural Analysis (stdlib) | `morphogen/stdlib/music_structural.py` | Segmentation, section labeling | ❌ TODO |
| Compositional (stdlib) | `morphogen/stdlib/music_compositional.py` | Voice-leading, progressions | ❌ TODO |
| Features MLIR Dialect | `morphogen/mlir/dialects/features.py` | Feature extraction ops | ❌ TODO |
| Symbolic MLIR Dialect | `morphogen/mlir/dialects/symbolic.py` | Music theory ops | ❌ TODO |
| Structural MLIR Dialect | `morphogen/mlir/dialects/structural.py` | Structure ops | ❌ TODO |
| Compositional MLIR Dialect | `morphogen/mlir/dialects/compositional.py` | Composition ops | ❌ TODO |

---

#### **RiffStack - Frontend & Language Design**

**Existing (Vision Documents)**:

| Document | Location | Content | Status |
|----------|----------|---------|--------|
| MLIR Architecture | `/home/scottsen/src/projects/riffstack/docs/MLIR_ARCHITECTURE.md` | 7-layer creative compiler vision | 📝 Design |
| Harmony DSL Vision | `/home/scottsen/src/projects/riffstack/docs/HARMONY_DSL_VISION.md` | Harmony language design (Layer 7) | 📝 Design |
| Harmony DSL Overview | `/home/scottsen/src/projects/riffstack/docs/HARMONY_DSL_OVERVIEW.md` | Multi-layer IR architecture | 📝 Design |
| Architecture Patterns | `/home/scottsen/src/projects/riffstack/docs/RIFFSTACK_ARCHITECTURE_PATTERNS.md` | Design patterns | 📝 Design |

**Implementation** (Minimal):

| Document | Location | Content | Status |
|----------|----------|---------|--------|
| Generators | `riffstack_core/audio/ops/generators.py` | 4 oscillators | ⚠️ Minimal |
| Processors | `riffstack_core/audio/ops/processors.py` | 7 basic operations | ⚠️ Minimal |
| Stack Machine | `riffstack_core/engine/stack_machine.py` | RPN interpreter | ✅ Works |
| Patch Loader | `riffstack_core/engine/patch_loader.py` | YAML loader | ✅ Works |

**Status**: RiffStack has excellent vision docs but minimal implementation. Per ADR-013, will transition to frontend for Morphogen.

---

### **Specifications & Reference**

| Document | Location | Content | Status |
|----------|----------|---------|--------|
| **Ambient Music Spec** | `morphogen/docs/specifications/ambient-music.md` | Generative ambient music (2,036 lines) | ✅ Complete |
| **Audio Synthesis Spec** | `morphogen/docs/specifications/audio-synthesis.md` | Synthesis domain specification | ✅ Complete |
| **Timbre Extraction Spec** | `morphogen/docs/specifications/timbre-extraction.md` | Physical modeling analysis | ✅ Complete |
| **ADR-009: Ambient Music** | `morphogen/docs/adr/009-ambient-music-generative-domains.md` | Generative audio domains | ✅ Complete |
| **Audio Visualization Ideas** | `morphogen/docs/reference/audio-visualization-ideas.md` | Cross-domain sonification | ✅ Complete |

---

### **Examples & Tutorials**

| Location | Content | Status |
|----------|---------|--------|
| `morphogen/examples/audio/` | Audio synthesis examples | ✅ Exists |
| `morphogen/examples/audio_visualizer/` | Real-time audio visualization | ✅ Exists |
| `morphogen/examples/cross_domain/` | Audio-reactive visuals | ✅ Exists |
| `riffstack/examples/*.yaml` | RiffStack YAML patches | ✅ 4 examples |

---

## 🗺️ Documentation by Use Case

### **I want to understand the overall architecture**

1. Start: **[Music Semantic Layers](architecture/MUSIC_SEMANTIC_LAYERS.md)**
   - Understand the 7 layers
   - See how analysis and synthesis fit together

2. Read: **[ADR-013: Consolidation](adr/013-music-stack-consolidation.md)**
   - Why Morphogen is the platform
   - How RiffStack integrates

3. Read: **[Mathematical Music Frameworks](reference/mathematical-music-frameworks.md)**
   - Group theory for chords
   - Topology for voice-leading
   - Linear algebra for analysis

---

### **I want to do music analysis (INPUT)**

**Current State**:
1. Check `morphogen/stdlib/audio.py` for:
   - Spectral analysis: `fft`, `stft`, `spectrum`
   - Spectral features: `spectral_centroid`, `spectral_rolloff`, `spectral_flux`

2. Check `morphogen/stdlib/audio_analysis.py` for:
   - Pitch tracking: `track_fundamental`
   - Harmonic analysis: `track_partials`, `analyze_modes`

**Coming Soon** (per ADR-013):
- Mel spectrograms, chroma features (Layer 4: Feature)
- Beat tracking, chord recognition (Layer 5: Symbolic)
- Structural segmentation (Layer 6: Structural)

---

### **I want to do music synthesis (OUTPUT)**

**Current State**:
1. Use `morphogen/stdlib/audio.py`:
   - 5 oscillators (sine, saw, square, triangle, noise)
   - 10+ filters (lowpass, highpass, bandpass, VCF, EQ)
   - 8 effects (delay, reverb, chorus, flanger, drive, limiter)
   - 3 envelopes (ADSR, AR, exponential)
   - Physical models (string, modal)

2. Examples in `morphogen/examples/audio/`

**Coming Soon** (per ADR-013):
- Higher-level composition layer (Layer 7: Compositional)
- Voice-leading, progression generation

---

### **I want to use the Harmony DSL (composition)**

**Current State**:
- Read vision: `riffstack/docs/HARMONY_DSL_VISION.md`
- Status: Design only, not implemented

**Future** (per ADR-013):
- Harmony DSL will be implemented in Morphogen Layer 7 (Compositional)
- RiffStack YAML frontend will compile to Morphogen MLIR

---

### **I want to understand MLIR compilation**

1. **Morphogen MLIR**:
   - Read `morphogen/mlir/compiler.py` (1,447 lines)
   - Check existing dialects: `morphogen/mlir/dialects/audio.py`
   - Lowering passes: `morphogen/mlir/lowering/`

2. **RiffStack MLIR Vision**:
   - Read `/home/scottsen/src/projects/riffstack/docs/MLIR_ARCHITECTURE.md`
   - Understand 7-layer compilation stack
   - Note: This vision will be implemented in Morphogen per ADR-013

---

### **I want to work on neural models**

**Current State**:
- Morphogen has `morphogen/stdlib/neural.py` (633 lines)
- General neural network operations (not music-specific)

**Coming Soon** (per ADR-013):
- `morphogen/models/backbone.py` - Conformer/Transformer for music
- `morphogen/models/heads/` - Beat, chord, section heads
- `morphogen/models/stablehlo_bridge.py` - StableHLO integration

**Reference**:
- User's proposed INPUT architecture (this session)
- StableHLO integration for music understanding

---

## 📋 Implementation Roadmap

See **[ADR-013: Music Stack Consolidation](adr/013-music-stack-consolidation.md)** for detailed phases.

**Quick Summary**:

```
Phase 1 (Q1 2025): Feature Layer
  └─ audio_features.py: mel, chroma, HPSS, SSM

Phase 2 (Q1-Q2 2025): Symbolic Layer
  └─ music_symbolic.py: beat tracking, chord recognition

Phase 3 (Q2 2025): Structural Layer
  └─ music_structural.py: segmentation, section labeling

Phase 4 (Q3 2025): Compositional Layer
  └─ music_compositional.py: voice-leading, progressions

Phase 5 (Q3 2025): RiffStack Integration
  └─ RiffStack frontend → Morphogen MLIR
```

---

## 🔍 Quick Reference

### **Finding Operations**

**Morphogen operations** are organized by domain in `morphogen/stdlib/`:

```python
# Audio synthesis & DSP (Layer 1-2)
from morphogen.stdlib import audio

audio.sine(440)              # Oscillator
audio.lowpass(signal, 1200)  # Filter
audio.reverb(signal, 0.5)    # Effect

# Spectral analysis (Layer 3)
audio.fft(signal)            # Fourier transform
audio.stft(signal)           # Short-time FT
audio.spectral_centroid(signal)  # "Brightness"

# Timbre analysis (Layer 4, partial)
from morphogen.stdlib import audio_analysis

audio_analysis.track_fundamental(signal, sr)  # Pitch tracking
audio_analysis.analyze_modes(signal, sr)      # Modal analysis
```

**Coming soon**:

```python
# Feature extraction (Layer 4)
from morphogen.stdlib import audio_features

audio_features.melspectrogram(signal)
audio_features.chroma(signal)
audio_features.hpss(signal)  # Harmonic-Percussive

# Symbolic music (Layer 5)
from morphogen.stdlib import music_symbolic

music_symbolic.beat_track(signal)
music_symbolic.chord_estimate(chroma, beats)
music_symbolic.key_estimate(chroma)

# Structural analysis (Layer 6)
from morphogen.stdlib import music_structural

music_structural.segment_structure(ssm)
music_structural.label_sections(sections, chords)

# Composition (Layer 7)
from morphogen.stdlib import music_compositional

music_compositional.voice_lead(progression, "smooth")
music_compositional.harmonize(melody, style="jazz")
```

---

### **MLIR Dialect Mapping**

| Semantic Layer | MLIR Dialect | Module | Status |
|----------------|--------------|--------|--------|
| 1. Physical | `physical` | `audio.py` (partial) | ⚠️ Partial |
| 2. DSP | `dsp` | `audio.py` (partial) | ⚠️ Partial |
| 3. Spectral | `spectral` | *To be created* | ❌ TODO |
| 4. Feature | `features` | *To be created* | ❌ TODO |
| 5. Symbolic | `symbolic` | *To be created* | ❌ TODO |
| 6. Structural | `structural` | *To be created* | ❌ TODO |
| 7. Compositional | `compositional` | *To be created* | ❌ TODO |

---

## 📞 Getting Help

### **Documentation Issues**

- Missing documentation: Check if it's in the roadmap (ADR-013)
- Unclear documentation: File an issue
- Documentation bugs: File PR

### **Implementation Questions**

- **Morphogen implementation**: Check `morphogen/stdlib/*.py`
- **MLIR questions**: Check `morphogen/mlir/`
- **RiffStack integration**: Check ADR-013 migration plan

### **Architectural Questions**

- Read: [Music Semantic Layers](architecture/MUSIC_SEMANTIC_LAYERS.md)
- Read: [ADR-013](adr/013-music-stack-consolidation.md)
- Context: This documentation session (divine-hero-1210)

---

## 🔄 Keeping This Index Updated

**When adding new documentation**:

1. Add entry to appropriate section above
2. Update status (❌ TODO → ⚠️ In Progress → ✅ Complete)
3. Add cross-references to related docs
4. Update roadmap if timeline changes

**When moving documentation**:

1. Update all paths in this index
2. Update cross-references in other docs
3. Leave breadcrumb at old location

---

## 📅 Version History

- **2025-12-10**: Initial index created (divine-hero-1210 session)
  - Documented existing Morphogen music work
  - Documented RiffStack vision
  - Created Music Semantic Layers architecture
  - Created ADR-013 for consolidation
  - Organized documentation structure

---

**Maintainer**: TIA + Scott Senften
**Last Review**: 2025-12-10
**Next Review**: After Phase 1 implementation
