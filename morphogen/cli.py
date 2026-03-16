"""Command-line interface for Creative Computation DSL."""

import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Creative Computation DSL v0.2.2 - A typed, deterministic simulation language"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a DSL program")
    run_parser.add_argument("file", type=Path, help="DSL source file to run")
    run_parser.add_argument(
        "--profile",
        choices=["low", "medium", "high"],
        default="medium",
        help="Execution profile (default: medium)"
    )
    run_parser.add_argument(
        "--param",
        action="append",
        metavar="KEY=VALUE",
        help="Override parameter values"
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global RNG seed for deterministic execution"
    )
    run_parser.add_argument(
        "--steps",
        type=int,
        help="Maximum number of timesteps to run"
    )

    # Check command
    check_parser = subparsers.add_parser("check", help="Type-check a DSL program")
    check_parser.add_argument("file", type=Path, help="DSL source file to check")
    check_parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict type checking"
    )

    # Parse command
    parse_parser = subparsers.add_parser("parse", help="Parse and display AST")
    parse_parser.add_argument("file", type=Path, help="DSL source file to parse")
    parse_parser.add_argument(
        "--format",
        choices=["tree", "json", "graphviz"],
        default="tree",
        help="Output format for AST"
    )

    # MLIR command
    mlir_parser = subparsers.add_parser("mlir", help="Lower to MLIR and display")
    mlir_parser.add_argument("file", type=Path, help="DSL source file to lower")
    mlir_parser.add_argument(
        "--dialect",
        choices=["all", "linalg", "scf", "arith"],
        default="all",
        help="MLIR dialect to display"
    )

    # Version command
    version_parser = subparsers.add_parser("version", help="Show version information")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "check":
        cmd_check(args)
    elif args.command == "parse":
        cmd_parse(args)
    elif args.command == "mlir":
        cmd_mlir(args)
    elif args.command == "version":
        cmd_version(args)
    else:
        parser.print_help()
        sys.exit(1)


def _apply_params(context, param_list):
    """Parse and apply key=value parameter overrides to the execution context."""
    for param_str in param_list:
        key, value = param_str.split("=", 1)
        try:
            value = float(value)
            if value.is_integer():
                value = int(value)
        except ValueError:
            pass  # Keep as string
        context.set_config(key, value)


def _has_visual_output(step_blocks):
    """Return True if any step block contains visual.output or visual.display calls."""
    for step_block in step_blocks:
        step_str = str(step_block)
        if 'visual.output' in step_str or 'visual.display' in step_str:
            return True
    return False


def _run_steps(runtime, step_blocks, max_steps):
    """Execute step blocks up to max_steps times; return the step count reached."""
    step_count = 0
    try:
        while step_count < max_steps:
            if step_count % 10 == 0:
                print(f"Step {step_count + 1}...")
            for step_block in step_blocks:
                runtime.execute_step(step_block)
            step_count += 1
    except KeyboardInterrupt:
        print(f"\nInterrupted after {step_count} steps")
    return step_count


def cmd_run(args):
    """Run a DSL program."""
    if not args.file.exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    print(f"Running {args.file} with profile={args.profile}, seed={args.seed}")

    try:
        from morphogen.parser.parser import parse
        from morphogen.ast.visitors import TypeChecker
        from morphogen.runtime.runtime import Runtime, ExecutionContext
        from morphogen.ast.nodes import Step

        source = args.file.read_text()
        program = parse(source)

        checker = TypeChecker()
        checker.visit(program)
        if checker.errors:
            print("\nType errors found:")
            for error in checker.errors:
                print(f"  - {error}")
            print("\nContinuing execution anyway...")

        context = ExecutionContext(global_seed=args.seed)
        context.set_config("profile", args.profile)
        if args.param:
            _apply_params(context, args.param)

        runtime = Runtime(context)

        init_statements = [s for s in program.statements if not isinstance(s, Step)]
        step_blocks = [s for s in program.statements if isinstance(s, Step)]

        print("Initializing...")
        for stmt in init_statements:
            runtime.execute_statement(stmt)

        if step_blocks:
            if len(step_blocks) > 1:
                print("Warning: Multiple step blocks found, executing all sequentially")
            max_steps = args.steps if args.steps else float('inf')
            if max_steps == float('inf') and not _has_visual_output(step_blocks):
                print("Error: Infinite loop without visual output. Use --steps or add visual.output()")
                sys.exit(1)
            step_count = _run_steps(runtime, step_blocks, max_steps)
            print(f"\nExecution completed ({step_count} steps)")
        else:
            print("No step blocks found, executing program once")

        print("Done!")

    except Exception as e:
        print(f"Runtime error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cmd_check(args):
    """Type-check a DSL program."""
    if not args.file.exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    print(f"Type-checking {args.file}...")

    try:
        from morphogen.parser.parser import parse
        from morphogen.ast.visitors import TypeChecker

        source = args.file.read_text()
        program = parse(source)

        checker = TypeChecker()
        checker.visit(program)

        if checker.errors:
            print("\nType errors found:")
            for error in checker.errors:
                print(f"  - {error}")
            sys.exit(1)
        else:
            print("✓ No type errors found")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_parse(args):
    """Parse and display AST."""
    if not args.file.exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    try:
        from morphogen.parser.parser import parse
        from morphogen.ast.visitors import ASTPrinter

        source = args.file.read_text()
        program = parse(source)

        if args.format == "tree":
            printer = ASTPrinter()
            print(printer.visit(program))
        elif args.format == "json":
            print("JSON output not yet implemented")
        elif args.format == "graphviz":
            print("Graphviz output not yet implemented")

    except Exception as e:
        print(f"Parse error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_mlir(args):
    """Lower to MLIR and display."""
    if not args.file.exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    print(f"Lowering {args.file} to MLIR...\n")

    try:
        from morphogen.parser.parser import parse
        from morphogen.mlir.compiler import MLIRCompiler
        from morphogen.mlir.optimizer import optimize_module, create_default_pipeline

        # Parse source file
        source = args.file.read_text()
        program = parse(source)

        # Compile to MLIR
        compiler = MLIRCompiler()
        ir_module = compiler.compile_program(program)

        # Verify module
        try:
            ir_module.verify()
        except ValueError as e:
            print(f"Warning: IR verification failed: {e}")

        # Optimize (Phase 5)
        print("Applying optimizations...")
        pipeline = create_default_pipeline()
        optimized_module = pipeline.optimize(ir_module)

        # Display MLIR
        print("\n" + "=" * 60)
        print("MLIR IR (optimized)")
        print("=" * 60)
        print(str(optimized_module))
        print("=" * 60)

        print("\n✓ MLIR compilation successful")

    except Exception as e:
        print(f"MLIR compilation error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cmd_version(args):
    """Show version information."""
    from morphogen import __version__
    print(f"Morphogen v{__version__}")
    print("Universal deterministic computation platform unifying audio, physics, circuits, and optimization")


if __name__ == "__main__":
    main()
