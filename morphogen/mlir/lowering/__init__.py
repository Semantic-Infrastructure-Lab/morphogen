"""MLIR Lowering Passes for Morphogen

This package contains lowering passes that transform Morphogen's high-level
dialects into progressively lower-level representations:

Morphogen Dialects → SCF/Arith/Func → LLVM Dialect → LLVM IR → Native Code

Passes:
- FieldToSCFPass: Lower field operations to structured control flow (Phase 2) ✅
- TemporalToSCFPass: Lower temporal operations to SCF loops (Phase 3) ✅
- AgentToSCFPass: Lower agent operations to memref arrays and loops (Phase 4) ✅
- AudioToSCFPass: Lower audio operations to waveform generation loops (Phase 5) ✅
- SCFToLLVMPass: Lower SCF to LLVM dialect (Phase 6) ✅
"""

# Phase 2 passes
from .field_to_scf import FieldToSCFPass, create_field_to_scf_pass, MLIR_AVAILABLE

# Phase 3 passes
from .temporal_to_scf import TemporalToSCFPass, create_temporal_to_scf_pass

# Phase 4 passes
from .agent_to_scf import AgentToSCFPass, create_agent_to_scf_pass

# Phase 5 passes
from .audio_to_scf import AudioToSCFPass, create_audio_to_scf_pass

# Phase 6 passes
from .scf_to_llvm import SCFToLLVMPass, create_scf_to_llvm_pass, lower_to_llvm

__all__ = [
    # Phase 2
    "FieldToSCFPass",
    "create_field_to_scf_pass",

    # Phase 3
    "TemporalToSCFPass",
    "create_temporal_to_scf_pass",

    # Phase 4
    "AgentToSCFPass",
    "create_agent_to_scf_pass",

    # Phase 5
    "AudioToSCFPass",
    "create_audio_to_scf_pass",

    # Phase 6
    "SCFToLLVMPass",
    "create_scf_to_llvm_pass",
    "lower_to_llvm",

    # Availability
    "MLIR_AVAILABLE",
]
