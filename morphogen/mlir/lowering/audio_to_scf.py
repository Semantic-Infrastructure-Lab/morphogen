"""Audio-to-SCF Lowering Pass for Morphogen v0.7.0 Phase 5

This module implements the lowering pass that transforms Morphogen audio operations
into Structured Control Flow (SCF) loops with memref operations.

Transformation:
    morphogen.audio.* ops → scf.for loops + memref.load/store + arith/math ops

Examples:
    1. Buffer Creation:
        Input:  %buf = morphogen.audio.buffer.create %sr, %ch, %dur
        Output: %mem = memref.alloc(%num_samples) : memref<?xf32>

    2. Oscillator (sine):
        Input:  %osc = morphogen.audio.oscillator %buf, %waveform=0, %freq, %phase
        Output: scf.for %i = %c0 to %num_samples step %c1 {
                  %t = ... // i / sample_rate
                  %sample = math.sin(2π * freq * t + phase)
                  memref.store %sample, %buf[%i]
                }

    3. ADSR Envelope:
        Input:  %env = morphogen.audio.envelope %buf, %a, %d, %s, %r
        Output: scf.for with state machine for attack/decay/sustain/release

    4. Filter (biquad):
        Input:  %filt = morphogen.audio.filter %buf, %type, %cutoff, %Q
        Output: scf.for with IIR biquad filter state (x[n-1], x[n-2], y[n-1], y[n-2])

    5. Mix:
        Input:  %mix = morphogen.audio.mix %buf1, %buf2, %gain1, %gain2
        Output: scf.for summing scaled samples
"""

from __future__ import annotations
from typing import Any, Optional, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from ..context import MorphogenMLIRContext

# Import MLIR if available
try:
    from mlir import ir
    from mlir.dialects import builtin, arith, memref, scf, func, math as mlir_math
    MLIR_AVAILABLE = True
except ImportError:
    MLIR_AVAILABLE = False
    if TYPE_CHECKING:
        from mlir import ir
    else:
        ir = None


class AudioToSCFPass:
    """Lowering pass: Audio operations → SCF loops + memref.

    This pass traverses the MLIR module and replaces audio operations
    with nested scf.for loops operating on memref storage.

    Operations Lowered:
        - morphogen.audio.buffer.create → memref.alloc
        - morphogen.audio.oscillator → scf.for with waveform generation
        - morphogen.audio.envelope → scf.for with ADSR state machine
        - morphogen.audio.filter → scf.for with biquad IIR filter
        - morphogen.audio.mix → scf.for with summation

    Usage:
        >>> pass_obj = AudioToSCFPass(context)
        >>> pass_obj.run(module)
    """

    def __init__(self, context: MorphogenMLIRContext):
        """Initialize audio-to-SCF pass.

        Args:
            context: Morphogen MLIR context
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        self.context = context

    def run(self, module: Any) -> None:
        """Run lowering pass on module.

        Args:
            module: MLIR module to transform (in-place)
        """
        with self.context.ctx:
            # Walk through all operations in the module
            for op in module.body.operations:
                self._process_operation(op)

    def _process_operation(self, op: Any) -> None:
        """Process a single operation recursively.

        Args:
            op: MLIR operation to process
        """
        # Import audio dialect to check for audio ops
        from ..dialects.audio import AudioDialect

        # Check if this is an audio operation
        if AudioDialect.is_audio_op(op):
            op_name = AudioDialect.get_audio_op_name(op)
            if op_name == "morphogen.audio.buffer.create":
                self._lower_buffer_create(op)
            elif op_name == "morphogen.audio.oscillator":
                self._lower_oscillator(op)
            elif op_name == "morphogen.audio.envelope":
                self._lower_envelope(op)
            elif op_name == "morphogen.audio.filter":
                self._lower_filter(op)
            elif op_name == "morphogen.audio.mix":
                self._lower_mix(op)

        # Recursively process nested regions
        if hasattr(op, "regions"):
            for region in op.regions:
                for block in region.blocks:
                    for nested_op in block.operations:
                        self._process_operation(nested_op)

    def _lower_buffer_create(self, op: Any) -> None:
        """Lower morphogen.audio.buffer.create to memref.alloc.

        Input:
            %buf = morphogen.audio.buffer.create %sample_rate, %channels, %duration

        Output:
            %num_samples = sample_rate * duration * channels  // computed
            %mem = memref.alloc(%num_samples) : memref<?xf32>
            // Initialize to zeros
            scf.for %i = %c0 to %num_samples step %c1 {
              memref.store %c0_f32, %mem[%i]
            }
        """
        with self.context.ctx, ir.Location.unknown():
            operands = op.operands
            if len(operands) < 3:
                raise ValueError("audio.buffer.create requires 3 operands")

            sample_rate = operands[0]  # index type
            channels = operands[1]  # index type
            duration = operands[2]  # f32 type

            with ir.InsertionPoint(op):
                # Convert sample_rate to f32 for multiplication
                sample_rate_f32 = arith.IndexCastOp(ir.F32Type.get(), sample_rate).result

                # Compute num_samples = sample_rate * duration * channels
                sr_times_dur = arith.MulFOp(sample_rate_f32, duration).result
                channels_f32 = arith.IndexCastOp(ir.F32Type.get(), channels).result
                num_samples_f32 = arith.MulFOp(sr_times_dur, channels_f32).result

                # Convert back to index for allocation
                num_samples = arith.FPToSIOp(ir.IntegerType.get_signless(64), num_samples_f32).result
                num_samples_idx = arith.IndexCastOp(ir.IndexType.get(), num_samples).result

                # Allocate memref
                f32 = ir.F32Type.get()
                memref_type = ir.MemRefType.get([ir.ShapedType.get_dynamic_size()], f32)
                mem = memref.AllocOp(memref_type, [num_samples_idx], []).result

                # Initialize to zeros
                c0 = arith.ConstantOp(ir.IndexType.get(), 0).result
                c1 = arith.ConstantOp(ir.IndexType.get(), 1).result
                c0_f32 = arith.ConstantOp(f32, ir.FloatAttr.get(f32, 0.0)).result

                for_op = scf.ForOp(c0, num_samples_idx, c1)
                with ir.InsertionPoint(for_op.body):
                    i = for_op.induction_variable
                    memref.StoreOp(c0_f32, mem, [i])
                    scf.YieldOp([])

            # Replace uses
            op.results[0].replace_all_uses_with(mem)
            op.operation.erase()

    def _lower_oscillator(self, op: Any) -> None:
        """Lower morphogen.audio.oscillator to waveform generation loop.

        Input:
            %osc = morphogen.audio.oscillator %buffer, %waveform, %freq, %phase

        Output (sine wave example, waveform=0):
            %num_samples = memref.dim %buffer, %c0
            scf.for %i = %c0 to %num_samples step %c1 {
              // Compute time: t = i / sample_rate
              %i_f32 = arith.index_cast %i : index to f32
              %t = arith.divf %i_f32, %sample_rate_f32

              // Compute phase: 2π * freq * t + phase
              %omega_t = arith.mulf %freq, %t
              %two_pi_omega_t = arith.mulf %c_2pi, %omega_t
              %total_phase = arith.addf %two_pi_omega_t, %phase

              // Compute sample based on waveform type (0=sine, 1=square, 2=saw, 3=triangle)
              %sample = math.sin %total_phase  // for sine

              // Store to buffer
              memref.store %sample, %buffer[%i]
            }

        Note: For waveform selection, we use scf.if to switch between waveforms
        """
        with self.context.ctx, ir.Location.unknown():
            operands = op.operands
            if len(operands) < 4:
                raise ValueError("audio.oscillator requires 4 operands")

            buffer = operands[0]
            waveform = operands[1]  # index: 0=sine, 1=square, 2=saw, 3=triangle
            frequency = operands[2]  # f32
            phase = operands[3]  # f32

            with ir.InsertionPoint(op):
                f32 = ir.F32Type.get()

                # Get buffer size
                c0 = arith.ConstantOp(ir.IndexType.get(), 0).result
                c1 = arith.ConstantOp(ir.IndexType.get(), 1).result
                num_samples = memref.DimOp(buffer, c0).result

                # Constants
                sample_rate_f32 = arith.ConstantOp(f32, ir.FloatAttr.get(f32, 44100.0)).result
                two_pi = arith.ConstantOp(f32, ir.FloatAttr.get(f32, 2.0 * math.pi)).result

                # Generate waveform
                for_op = scf.ForOp(c0, num_samples, c1)
                with ir.InsertionPoint(for_op.body):
                    i = for_op.induction_variable

                    # t = i / sample_rate
                    i_f32 = arith.IndexCastOp(f32, i).result
                    t = arith.DivFOp(i_f32, sample_rate_f32).result

                    # total_phase = 2π * freq * t + phase
                    freq_times_t = arith.MulFOp(frequency, t).result
                    two_pi_freq_t = arith.MulFOp(two_pi, freq_times_t).result
                    total_phase = arith.AddFOp(two_pi_freq_t, phase).result

                    # Generate sample based on waveform type
                    # For simplicity in Phase 5, we implement sine wave
                    # TODO: Add scf.if branching for other waveform types
                    sample = mlir_math.SinOp(total_phase).result

                    # Store sample
                    memref.StoreOp(sample, buffer, [i])
                    scf.YieldOp([])

                # Create output memref (copy of buffer)
                output_mem = buffer

            # Replace uses
            op.results[0].replace_all_uses_with(output_mem)
            op.operation.erase()

    def _lower_envelope(self, op: Any) -> None:
        """Lower morphogen.audio.envelope to ADSR state machine loop.

        Input:
            %env = morphogen.audio.envelope %buffer, %attack, %decay, %sustain, %release

        Output:
            %num_samples = memref.dim %buffer, %c0
            %duration = %num_samples / %sample_rate  // total duration

            scf.for %i = %c0 to %num_samples step %c1 {
              // Compute time: t = i / sample_rate
              %t = ...

              // ADSR state machine
              %env_value = <compute based on t, attack, decay, sustain, release>

              // Apply envelope: buffer[i] *= env_value
              %original = memref.load %buffer[%i]
              %enveloped = arith.mulf %original, %env_value
              memref.store %enveloped, %buffer[%i]
            }
        """
        with self.context.ctx, ir.Location.unknown():
            operands = op.operands
            if len(operands) < 5:
                raise ValueError("audio.envelope requires 5 operands")

            buffer = operands[0]
            attack = operands[1]  # f32
            decay = operands[2]  # f32
            sustain = operands[3]  # f32 (level, 0.0 to 1.0)
            release = operands[4]  # f32

            with ir.InsertionPoint(op):
                f32 = ir.F32Type.get()

                # Get buffer size
                c0 = arith.ConstantOp(ir.IndexType.get(), 0).result
                c1 = arith.ConstantOp(ir.IndexType.get(), 1).result
                num_samples = memref.DimOp(buffer, c0).result

                # Constants
                sample_rate_f32 = arith.ConstantOp(f32, ir.FloatAttr.get(f32, 44100.0)).result
                c0_f32 = arith.ConstantOp(f32, ir.FloatAttr.get(f32, 0.0)).result
                c1_f32 = arith.ConstantOp(f32, ir.FloatAttr.get(f32, 1.0)).result

                # Compute duration
                num_samples_f32 = arith.IndexCastOp(f32, num_samples).result
                duration = arith.DivFOp(num_samples_f32, sample_rate_f32).result

                # Compute attack end time, decay end time, release start time
                attack_end = attack
                ad_sum = arith.AddFOp(attack, decay).result
                decay_end = ad_sum
                release_start = arith.SubFOp(duration, release).result

                # Apply envelope
                for_op = scf.ForOp(c0, num_samples, c1)
                with ir.InsertionPoint(for_op.body):
                    i = for_op.induction_variable

                    # t = i / sample_rate
                    i_f32 = arith.IndexCastOp(f32, i).result
                    t = arith.DivFOp(i_f32, sample_rate_f32).result

                    # ADSR state machine using nested scf.if
                    # if t < attack: env = t / attack
                    t_lt_attack = arith.CmpFOp(arith.CmpFPredicate.OLT, t, attack_end).result
                    env_attack = arith.DivFOp(t, attack).result

                    # else if t < attack + decay: env = 1.0 - (1.0 - sustain) * (t - attack) / decay
                    t_lt_decay_end = arith.CmpFOp(arith.CmpFPredicate.OLT, t, decay_end).result
                    t_minus_attack = arith.SubFOp(t, attack_end).result
                    one_minus_sustain = arith.SubFOp(c1_f32, sustain).result
                    decay_factor = arith.DivFOp(t_minus_attack, decay).result
                    decay_amount = arith.MulFOp(one_minus_sustain, decay_factor).result
                    env_decay = arith.SubFOp(c1_f32, decay_amount).result

                    # else if t < duration - release: env = sustain
                    t_lt_release_start = arith.CmpFOp(arith.CmpFPredicate.OLT, t, release_start).result
                    env_sustain = sustain

                    # else: env = sustain * (1.0 - (t - (duration - release)) / release)
                    t_minus_release_start = arith.SubFOp(t, release_start).result
                    release_factor = arith.DivFOp(t_minus_release_start, release).result
                    one_minus_release_factor = arith.SubFOp(c1_f32, release_factor).result
                    env_release = arith.MulFOp(sustain, one_minus_release_factor).result

                    # Nested if-else to select envelope value
                    # Simplified: Use attack envelope (full ADSR requires complex nested scf.if)
                    # For Phase 5, we use a simplified linear envelope
                    env_value = scf.IfOp(t_lt_attack, [f32], hasElse=True)
                    with ir.InsertionPoint(env_value.then_block):
                        scf.YieldOp([env_attack])
                    with ir.InsertionPoint(env_value.else_block):
                        # Nested if for decay/sustain/release
                        env_value_2 = scf.IfOp(t_lt_decay_end, [f32], hasElse=True)
                        with ir.InsertionPoint(env_value_2.then_block):
                            scf.YieldOp([env_decay])
                        with ir.InsertionPoint(env_value_2.else_block):
                            env_value_3 = scf.IfOp(t_lt_release_start, [f32], hasElse=True)
                            with ir.InsertionPoint(env_value_3.then_block):
                                scf.YieldOp([env_sustain])
                            with ir.InsertionPoint(env_value_3.else_block):
                                scf.YieldOp([env_release])
                            scf.YieldOp(env_value_3.results)
                        scf.YieldOp(env_value_2.results)

                    final_env = env_value.results[0]

                    # Apply envelope to buffer sample
                    original = memref.LoadOp(buffer, [i]).result
                    enveloped = arith.MulFOp(original, final_env).result
                    memref.StoreOp(enveloped, buffer, [i])
                    scf.YieldOp([])

                output_mem = buffer

            # Replace uses
            op.results[0].replace_all_uses_with(output_mem)
            op.operation.erase()

    def _lower_filter(self, op: Any) -> None:
        """Lower morphogen.audio.filter to biquad IIR filter loop.

        Input:
            %filt = morphogen.audio.filter %buffer, %filter_type, %cutoff, %resonance

        Output:
            // Allocate state variables for IIR filter
            %x1 = memref.alloca() : memref<1xf32>  // x[n-1]
            %x2 = memref.alloca() : memref<1xf32>  // x[n-2]
            %y1 = memref.alloca() : memref<1xf32>  // y[n-1]
            %y2 = memref.alloca() : memref<1xf32>  // y[n-2]

            // Initialize state to 0
            memref.store %c0, %x1[%c0]
            memref.store %c0, %x2[%c0]
            memref.store %c0, %y1[%c0]
            memref.store %c0, %y2[%c0]

            // Compute biquad coefficients (a0, a1, a2, b1, b2) from cutoff and Q
            %coeffs = <compute_biquad_coeffs(filter_type, cutoff, resonance)>

            scf.for %i = %c0 to %num_samples step %c1 {
              %x_n = memref.load %buffer[%i]
              %x_n_1 = memref.load %x1[%c0]
              %x_n_2 = memref.load %x2[%c0]
              %y_n_1 = memref.load %y1[%c0]
              %y_n_2 = memref.load %y2[%c0]

              // y[n] = a0*x[n] + a1*x[n-1] + a2*x[n-2] - b1*y[n-1] - b2*y[n-2]
              %y_n = <biquad computation>

              // Update state
              memref.store %x_n, %x1[%c0]
              memref.store %x_n_1, %x2[%c0]
              memref.store %y_n, %y1[%c0]
              memref.store %y_n_1, %y2[%c0]

              // Store filtered sample
              memref.store %y_n, %buffer[%i]
            }
        """
        with self.context.ctx, ir.Location.unknown():
            operands = op.operands
            if len(operands) < 4:
                raise ValueError("audio.filter requires 4 operands")

            buffer = operands[0]
            filter_type = operands[1]  # index: 0=lowpass, 1=highpass, 2=bandpass
            cutoff = operands[2]  # f32
            resonance = operands[3]  # f32 (Q factor)

            with ir.InsertionPoint(op):
                f32 = ir.F32Type.get()

                # Get buffer size
                c0 = arith.ConstantOp(ir.IndexType.get(), 0).result
                c1 = arith.ConstantOp(ir.IndexType.get(), 1).result
                num_samples = memref.DimOp(buffer, c0).result

                # Constants
                c0_f32 = arith.ConstantOp(f32, ir.FloatAttr.get(f32, 0.0)).result
                sample_rate_f32 = arith.ConstantOp(f32, ir.FloatAttr.get(f32, 44100.0)).result

                # Allocate state variables (on stack)
                state_type = ir.MemRefType.get([1], f32)
                x1_mem = memref.AllocaOp(state_type, [], []).result
                x2_mem = memref.AllocaOp(state_type, [], []).result
                y1_mem = memref.AllocaOp(state_type, [], []).result
                y2_mem = memref.AllocaOp(state_type, [], []).result

                # Initialize state to 0
                memref.StoreOp(c0_f32, x1_mem, [c0])
                memref.StoreOp(c0_f32, x2_mem, [c0])
                memref.StoreOp(c0_f32, y1_mem, [c0])
                memref.StoreOp(c0_f32, y2_mem, [c0])

                # Compute biquad coefficients (simplified lowpass for Phase 5)
                # Full implementation would compute a0, a1, a2, b1, b2 based on filter_type
                # For now, use simple lowpass: y[n] = α*x[n] + (1-α)*y[n-1]
                # where α = 2π * cutoff / sample_rate

                # Simplified single-pole lowpass coefficient
                two_pi = arith.ConstantOp(f32, ir.FloatAttr.get(f32, 2.0 * math.pi)).result
                omega = arith.MulFOp(two_pi, cutoff).result
                omega_over_sr = arith.DivFOp(omega, sample_rate_f32).result

                # Clamp alpha to [0, 1]
                c1_f32 = arith.ConstantOp(f32, ir.FloatAttr.get(f32, 1.0)).result
                alpha_clamped = omega_over_sr  # Simplified, should clamp
                one_minus_alpha = arith.SubFOp(c1_f32, alpha_clamped).result

                # Apply filter
                for_op = scf.ForOp(c0, num_samples, c1)
                with ir.InsertionPoint(for_op.body):
                    i = for_op.induction_variable

                    # Load current sample and previous output
                    x_n = memref.LoadOp(buffer, [i]).result
                    y_n_1 = memref.LoadOp(y1_mem, [c0]).result

                    # Simple lowpass: y[n] = α*x[n] + (1-α)*y[n-1]
                    alpha_x_n = arith.MulFOp(alpha_clamped, x_n).result
                    one_minus_alpha_y_n_1 = arith.MulFOp(one_minus_alpha, y_n_1).result
                    y_n = arith.AddFOp(alpha_x_n, one_minus_alpha_y_n_1).result

                    # Update state
                    memref.StoreOp(y_n, y1_mem, [c0])

                    # Store filtered sample
                    memref.StoreOp(y_n, buffer, [i])
                    scf.YieldOp([])

                output_mem = buffer

            # Replace uses
            op.results[0].replace_all_uses_with(output_mem)
            op.operation.erase()

    def _lower_mix(self, op: Any) -> None:
        """Lower morphogen.audio.mix to summation loop.

        Input:
            %mix = morphogen.audio.mix %buf1, %buf2, %gain1, %gain2

        Output:
            %num_samples = memref.dim %buf1, %c0
            %output = memref.alloc(%num_samples) : memref<?xf32>

            scf.for %i = %c0 to %num_samples step %c1 {
              %s1 = memref.load %buf1[%i]
              %s2 = memref.load %buf2[%i]
              %scaled1 = arith.mulf %s1, %gain1
              %scaled2 = arith.mulf %s2, %gain2
              %mixed = arith.addf %scaled1, %scaled2
              memref.store %mixed, %output[%i]
            }
        """
        with self.context.ctx, ir.Location.unknown():
            operands = op.operands

            # Get num_buffers from attribute
            if "num_buffers" not in op.operation.attributes:
                raise ValueError("audio.mix requires num_buffers attribute")

            num_buffers = int(op.operation.attributes["num_buffers"])

            if len(operands) < num_buffers * 2:
                raise ValueError(f"audio.mix requires {num_buffers * 2} operands")

            # Extract buffers and gains (interleaved in operands)
            buffers = []
            gains = []
            for i in range(num_buffers):
                buffers.append(operands[i * 2])
                gains.append(operands[i * 2 + 1])

            with ir.InsertionPoint(op):
                f32 = ir.F32Type.get()

                # Get buffer size from first buffer
                c0 = arith.ConstantOp(ir.IndexType.get(), 0).result
                c1 = arith.ConstantOp(ir.IndexType.get(), 1).result
                num_samples = memref.DimOp(buffers[0], c0).result

                # Allocate output buffer
                memref_type = ir.MemRefType.get([ir.ShapedType.get_dynamic_size()], f32)
                output_mem = memref.AllocOp(memref_type, [num_samples], []).result

                # Mix buffers
                c0_f32 = arith.ConstantOp(f32, ir.FloatAttr.get(f32, 0.0)).result

                for_op = scf.ForOp(c0, num_samples, c1)
                with ir.InsertionPoint(for_op.body):
                    i = for_op.induction_variable

                    # Initialize accumulator to 0
                    accum = c0_f32

                    # Sum all scaled samples
                    for buf, gain in zip(buffers, gains):
                        sample = memref.LoadOp(buf, [i]).result
                        scaled = arith.MulFOp(sample, gain).result
                        accum = arith.AddFOp(accum, scaled).result

                    # Store mixed sample
                    memref.StoreOp(accum, output_mem, [i])
                    scf.YieldOp([])

            # Replace uses
            op.results[0].replace_all_uses_with(output_mem)
            op.operation.erase()


def create_audio_to_scf_pass(context: MorphogenMLIRContext) -> AudioToSCFPass:
    """Factory function to create audio-to-SCF pass.

    Args:
        context: Morphogen MLIR context

    Returns:
        AudioToSCFPass instance
    """
    return AudioToSCFPass(context)
