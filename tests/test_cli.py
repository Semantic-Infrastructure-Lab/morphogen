"""Unit tests for CLI interface."""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from io import StringIO

from morphogen.cli import (
    main,
    cmd_run,
    cmd_check,
    cmd_parse,
    cmd_mlir,
    cmd_version
)


class TestArgumentParsing:
    """Tests for command-line argument parsing."""

    def test_no_command_shows_help(self, capsys):
        """Test that no command shows help and exits."""
        with patch('sys.argv', ['morphogen']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert 'usage:' in captured.out.lower() or 'usage:' in captured.err.lower()

    def test_run_command_parsing(self):
        """Test parsing run command with various options."""
        test_args = [
            'morphogen', 'run', 'test.morph',
            '--profile', 'high',
            '--seed', '123',
            '--steps', '100',
            '--param', 'rate=0.5',
            '--param', 'size=64'
        ]

        with patch('sys.argv', test_args):
            with patch('morphogen.cli.cmd_run') as mock_cmd:
                main()

                # Verify cmd_run was called
                assert mock_cmd.called
                args = mock_cmd.call_args[0][0]

                # Verify arguments were parsed correctly
                assert args.file == Path('test.morph')
                assert args.profile == 'high'
                assert args.seed == 123
                assert args.steps == 100
                assert args.param == ['rate=0.5', 'size=64']

    def test_check_command_parsing(self):
        """Test parsing check command."""
        test_args = ['morphogen', 'check', 'test.morph', '--strict']

        with patch('sys.argv', test_args):
            with patch('morphogen.cli.cmd_check') as mock_cmd:
                main()

                args = mock_cmd.call_args[0][0]
                assert args.file == Path('test.morph')
                assert args.strict is True

    def test_parse_command_parsing(self):
        """Test parsing parse command with format options."""
        for fmt in ['tree', 'json', 'graphviz']:
            test_args = ['morphogen', 'parse', 'test.morph', '--format', fmt]

            with patch('sys.argv', test_args):
                with patch('morphogen.cli.cmd_parse') as mock_cmd:
                    main()

                    args = mock_cmd.call_args[0][0]
                    assert args.file == Path('test.morph')
                    assert args.format == fmt

    def test_mlir_command_parsing(self):
        """Test parsing mlir command with dialect options."""
        for dialect in ['all', 'linalg', 'scf', 'arith']:
            test_args = ['morphogen', 'mlir', 'test.morph', '--dialect', dialect]

            with patch('sys.argv', test_args):
                with patch('morphogen.cli.cmd_mlir') as mock_cmd:
                    main()

                    args = mock_cmd.call_args[0][0]
                    assert args.file == Path('test.morph')
                    assert args.dialect == dialect

    def test_version_command_parsing(self):
        """Test parsing version command."""
        test_args = ['morphogen', 'version']

        with patch('sys.argv', test_args):
            with patch('morphogen.cli.cmd_version') as mock_cmd:
                main()
                assert mock_cmd.called

    def test_default_values(self):
        """Test that default values are set correctly."""
        test_args = ['morphogen', 'run', 'test.morph']

        with patch('sys.argv', test_args):
            with patch('morphogen.cli.cmd_run') as mock_cmd:
                main()

                args = mock_cmd.call_args[0][0]
                assert args.profile == 'medium'  # default profile
                assert args.seed == 42  # default seed
                assert args.steps is None  # no default steps
                assert args.param is None  # no default params


class TestCmdRun:
    """Tests for the run command."""

    def test_run_file_not_found(self, capsys):
        """Test running with non-existent file."""
        args = Mock()
        args.file = Path('/nonexistent/file.morph')

        with pytest.raises(SystemExit) as exc_info:
            cmd_run(args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert 'File not found' in captured.err

    def test_run_simple_program(self, tmp_path, capsys):
        """Test running a simple program."""
        # Create a minimal test program
        test_file = tmp_path / "test.morph"
        test_file.write_text("x = 42")

        args = Mock()
        args.file = test_file
        args.profile = 'medium'
        args.seed = 42
        args.steps = None
        args.param = None

        # Mock the parser and runtime
        with patch('morphogen.parser.parser.parse') as mock_parse, \
             patch('morphogen.ast.visitors.TypeChecker') as mock_checker, \
             patch('morphogen.runtime.runtime.Runtime') as mock_runtime:

            # Setup mocks
            mock_program = Mock()
            mock_program.statements = []  # No step blocks
            mock_parse.return_value = mock_program

            mock_checker_instance = Mock()
            mock_checker_instance.errors = []
            mock_checker.return_value = mock_checker_instance

            # Run command
            cmd_run(args)

            # Verify parse was called with correct source
            mock_parse.assert_called_once()
            source = mock_parse.call_args[0][0]
            assert source == "x = 42"

            # Verify type checker was used
            assert mock_checker_instance.visit.called

    def test_run_with_type_errors_continues(self, tmp_path, capsys):
        """Test that run continues even with type errors."""
        test_file = tmp_path / "test.morph"
        test_file.write_text("x = 42")

        args = Mock()
        args.file = test_file
        args.profile = 'medium'
        args.seed = 42
        args.steps = None
        args.param = None

        with patch('morphogen.parser.parser.parse') as mock_parse, \
             patch('morphogen.ast.visitors.TypeChecker') as mock_checker, \
             patch('morphogen.runtime.runtime.Runtime'):

            mock_program = Mock()
            mock_program.statements = []
            mock_parse.return_value = mock_program

            # Simulate type errors
            mock_checker_instance = Mock()
            mock_checker_instance.errors = ["Type error 1", "Type error 2"]
            mock_checker.return_value = mock_checker_instance

            cmd_run(args)

            # Verify error messages were printed
            captured = capsys.readouterr()
            assert 'Type errors found' in captured.out
            assert 'Type error 1' in captured.out
            assert 'Continuing execution anyway' in captured.out

    def test_run_program_with_use_statement(self, tmp_path):
        """Run should support programs that begin with a use statement."""
        test_file = tmp_path / "test_use.morph"
        test_file.write_text("use field\nx = 42\n")

        args = Mock()
        args.file = test_file
        args.profile = 'medium'
        args.seed = 42
        args.steps = None
        args.param = None

        cmd_run(args)

    def test_run_flow_program_reports_flow_execution(self, tmp_path, capsys):
        """Run should report flow blocks instead of saying no step blocks were found."""
        test_file = tmp_path / "test_flow.morph"
        test_file.write_text("flow(steps=2) { x = 1 }\n")

        args = Mock()
        args.file = test_file
        args.profile = 'medium'
        args.seed = 42
        args.steps = None
        args.param = None

        cmd_run(args)

        captured = capsys.readouterr()
        assert 'Executing 1 flow block' in captured.out
        assert 'No step blocks found' not in captured.out

    def test_run_overrides_flow_steps_with_cli_steps(self, tmp_path):
        """CLI --steps should override flow block step counts."""
        from morphogen.ast.nodes import Flow, Program, Literal

        test_file = tmp_path / "test_flow_override.morph"
        test_file.write_text("flow(steps=5) { x = 1 }\n")

        flow = Flow(dt=None, steps=Literal(5), substeps=None, body=[])
        program = Program([flow])

        args = Mock()
        args.file = test_file
        args.profile = 'medium'
        args.seed = 42
        args.steps = 1
        args.param = None

        with patch('morphogen.parser.parser.parse', return_value=program), \
             patch('morphogen.ast.visitors.TypeChecker') as mock_checker, \
             patch('morphogen.runtime.runtime.Runtime'):
            mock_checker_instance = Mock()
            mock_checker_instance.errors = []
            mock_checker.return_value = mock_checker_instance

            cmd_run(args)

        assert flow.steps.value == 1

    def test_run_with_parameters(self, tmp_path):
        """Test running with parameter overrides."""
        test_file = tmp_path / "test.morph"
        test_file.write_text("x = 42")

        args = Mock()
        args.file = test_file
        args.profile = 'high'
        args.seed = 100
        args.steps = None
        args.param = ['rate=0.5', 'size=128', 'name=test']

        with patch('morphogen.parser.parser.parse') as mock_parse, \
             patch('morphogen.ast.visitors.TypeChecker') as mock_checker, \
             patch('morphogen.runtime.runtime.ExecutionContext') as mock_context_class, \
             patch('morphogen.runtime.runtime.Runtime'):

            mock_program = Mock()
            mock_program.statements = []
            mock_parse.return_value = mock_program

            mock_checker_instance = Mock()
            mock_checker_instance.errors = []
            mock_checker.return_value = mock_checker_instance

            mock_context = Mock()
            mock_context_class.return_value = mock_context

            cmd_run(args)

            # Verify context was configured with parameters
            config_calls = [call[0] for call in mock_context.set_config.call_args_list]
            assert ('profile', 'high') in config_calls
            assert ('rate', 0.5) in config_calls
            assert ('size', 128) in config_calls
            assert ('name', 'test') in config_calls

    def test_run_with_step_blocks(self, tmp_path, capsys):
        """Test running program with step blocks."""
        from morphogen.ast.nodes import Step

        test_file = tmp_path / "test.morph"
        test_file.write_text("step { x = 42 }")

        args = Mock()
        args.file = test_file
        args.profile = 'medium'
        args.seed = 42
        args.steps = 5  # Run for 5 steps
        args.param = None

        with patch('morphogen.parser.parser.parse') as mock_parse, \
             patch('morphogen.ast.visitors.TypeChecker') as mock_checker, \
             patch('morphogen.runtime.runtime.Runtime') as mock_runtime_class:

            # Create a real Step instance
            step_block = Step(body=[])

            mock_program = Mock()
            mock_program.statements = [step_block]
            mock_parse.return_value = mock_program

            mock_checker_instance = Mock()
            mock_checker_instance.errors = []
            mock_checker.return_value = mock_checker_instance

            mock_runtime = Mock()
            mock_runtime_class.return_value = mock_runtime

            cmd_run(args)

            # Verify execute_step was called 5 times
            assert mock_runtime.execute_step.call_count == 5

    def test_run_infinite_loop_error(self, tmp_path, capsys):
        """Test that infinite loops without output cause error."""
        from morphogen.ast.nodes import Step

        test_file = tmp_path / "test.morph"
        test_file.write_text("step { x = x + 1 }")

        args = Mock()
        args.file = test_file
        args.profile = 'medium'
        args.seed = 42
        args.steps = None  # No step limit
        args.param = None

        with patch('morphogen.parser.parser.parse') as mock_parse, \
             patch('morphogen.ast.visitors.TypeChecker') as mock_checker, \
             patch('morphogen.runtime.runtime.Runtime'):

            step_block = Step(body=[])

            mock_program = Mock()
            mock_program.statements = [step_block]
            mock_parse.return_value = mock_program

            mock_checker_instance = Mock()
            mock_checker_instance.errors = []
            mock_checker.return_value = mock_checker_instance

            with pytest.raises(SystemExit) as exc_info:
                cmd_run(args)

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert 'Infinite loop' in captured.out

    def test_run_keyboard_interrupt(self, tmp_path, capsys):
        """Test graceful handling of keyboard interrupt."""
        from morphogen.ast.nodes import Step

        test_file = tmp_path / "test.morph"
        test_file.write_text("step { x = x + 1 }")

        args = Mock()
        args.file = test_file
        args.profile = 'medium'
        args.seed = 42
        args.steps = 100
        args.param = None

        with patch('morphogen.parser.parser.parse') as mock_parse, \
             patch('morphogen.ast.visitors.TypeChecker') as mock_checker, \
             patch('morphogen.runtime.runtime.Runtime') as mock_runtime_class:

            step_block = Step(body=[])

            mock_program = Mock()
            mock_program.statements = [step_block]
            mock_parse.return_value = mock_program

            mock_checker_instance = Mock()
            mock_checker_instance.errors = []
            mock_checker.return_value = mock_checker_instance

            mock_runtime = Mock()
            # Simulate keyboard interrupt after 3 steps
            mock_runtime.execute_step.side_effect = [None, None, None, KeyboardInterrupt()]
            mock_runtime_class.return_value = mock_runtime

            cmd_run(args)

            captured = capsys.readouterr()
            assert 'Interrupted' in captured.out

    def test_run_runtime_error(self, tmp_path, capsys):
        """Test handling of runtime errors."""
        test_file = tmp_path / "test.morph"
        test_file.write_text("x = 42")

        args = Mock()
        args.file = test_file
        args.profile = 'medium'
        args.seed = 42
        args.steps = None
        args.param = None

        with patch('morphogen.parser.parser.parse') as mock_parse:
            # Simulate parse error
            mock_parse.side_effect = Exception("Parse failed!")

            with pytest.raises(SystemExit) as exc_info:
                cmd_run(args)

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert 'Runtime error' in captured.err
            assert 'Parse failed!' in captured.err


class TestCmdCheck:
    """Tests for the check command."""

    def test_check_program_with_use_statement(self, tmp_path, capsys):
        """Check should handle a program that imports domains via use."""
        test_file = tmp_path / "test_use.morph"
        test_file.write_text("use field\nx = 42\n")

        args = Mock()
        args.file = test_file
        args.strict = False

        cmd_check(args)

        captured = capsys.readouterr()
        assert 'No type errors found' in captured.out

    def test_check_file_not_found(self, capsys):
        """Test check with non-existent file."""
        args = Mock()
        args.file = Path('/nonexistent/file.morph')

        with pytest.raises(SystemExit) as exc_info:
            cmd_check(args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert 'File not found' in captured.err

    def test_check_no_errors(self, tmp_path, capsys):
        """Test successful type checking."""
        test_file = tmp_path / "test.morph"
        test_file.write_text("x = 42")

        args = Mock()
        args.file = test_file
        args.strict = False

        with patch('morphogen.parser.parser.parse') as mock_parse, \
             patch('morphogen.ast.visitors.TypeChecker') as mock_checker:

            mock_program = Mock()
            mock_parse.return_value = mock_program

            mock_checker_instance = Mock()
            mock_checker_instance.errors = []
            mock_checker.return_value = mock_checker_instance

            cmd_check(args)

            captured = capsys.readouterr()
            assert '✓ No type errors found' in captured.out

    def test_check_with_errors(self, tmp_path, capsys):
        """Test type checking with errors."""
        test_file = tmp_path / "test.morph"
        test_file.write_text("x : i32 = 'string'")

        args = Mock()
        args.file = test_file
        args.strict = False

        with patch('morphogen.parser.parser.parse') as mock_parse, \
             patch('morphogen.ast.visitors.TypeChecker') as mock_checker:

            mock_program = Mock()
            mock_parse.return_value = mock_program

            mock_checker_instance = Mock()
            mock_checker_instance.errors = ["Type mismatch: expected i32, got string"]
            mock_checker.return_value = mock_checker_instance

            with pytest.raises(SystemExit) as exc_info:
                cmd_check(args)

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert 'Type errors found' in captured.out
            assert 'Type mismatch' in captured.out

    def test_check_parse_error(self, tmp_path, capsys):
        """Test handling of parse errors during checking."""
        test_file = tmp_path / "test.morph"
        test_file.write_text("invalid syntax @#$")

        args = Mock()
        args.file = test_file
        args.strict = False

        with patch('morphogen.parser.parser.parse') as mock_parse:
            mock_parse.side_effect = Exception("Syntax error")

            with pytest.raises(SystemExit) as exc_info:
                cmd_check(args)

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert 'Error:' in captured.err


class TestCmdParse:
    """Tests for the parse command."""

    def test_parse_file_not_found(self, capsys):
        """Test parse with non-existent file."""
        args = Mock()
        args.file = Path('/nonexistent/file.morph')
        args.format = 'tree'

        with pytest.raises(SystemExit) as exc_info:
            cmd_parse(args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert 'File not found' in captured.err

    def test_parse_tree_format(self, tmp_path, capsys):
        """Test parsing with tree format."""
        test_file = tmp_path / "test.morph"
        test_file.write_text("x = 42")

        args = Mock()
        args.file = test_file
        args.format = 'tree'

        with patch('morphogen.parser.parser.parse') as mock_parse, \
             patch('morphogen.ast.visitors.ASTPrinter') as mock_printer_class:

            mock_program = Mock()
            mock_parse.return_value = mock_program

            mock_printer = Mock()
            mock_printer.visit.return_value = "AST Tree Output"
            mock_printer_class.return_value = mock_printer

            cmd_parse(args)

            captured = capsys.readouterr()
            assert 'AST Tree Output' in captured.out

    def test_parse_json_format(self, tmp_path, capsys):
        """Test parsing with JSON format (not yet implemented)."""
        test_file = tmp_path / "test.morph"
        test_file.write_text("x = 42")

        args = Mock()
        args.file = test_file
        args.format = 'json'

        with patch('morphogen.parser.parser.parse') as mock_parse:
            mock_program = Mock()
            mock_parse.return_value = mock_program

            cmd_parse(args)

            captured = capsys.readouterr()
            assert 'not yet implemented' in captured.out

    def test_parse_graphviz_format(self, tmp_path, capsys):
        """Test parsing with Graphviz format (not yet implemented)."""
        test_file = tmp_path / "test.morph"
        test_file.write_text("x = 42")

        args = Mock()
        args.file = test_file
        args.format = 'graphviz'

        with patch('morphogen.parser.parser.parse') as mock_parse:
            mock_program = Mock()
            mock_parse.return_value = mock_program

            cmd_parse(args)

            captured = capsys.readouterr()
            assert 'not yet implemented' in captured.out

    def test_parse_error(self, tmp_path, capsys):
        """Test handling of parse errors."""
        test_file = tmp_path / "test.morph"
        test_file.write_text("invalid @#$")

        args = Mock()
        args.file = test_file
        args.format = 'tree'

        with patch('morphogen.parser.parser.parse') as mock_parse:
            mock_parse.side_effect = Exception("Syntax error at line 1")

            with pytest.raises(SystemExit) as exc_info:
                cmd_parse(args)

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert 'Parse error' in captured.err


class TestCmdMlir:
    """Tests for the MLIR command."""

    def test_mlir_file_not_found(self, capsys):
        """Test MLIR with non-existent file."""
        args = Mock()
        args.file = Path('/nonexistent/file.morph')
        args.dialect = 'all'

        with pytest.raises(SystemExit) as exc_info:
            cmd_mlir(args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert 'File not found' in captured.err

    def test_mlir_compilation_success(self, tmp_path, capsys):
        """Test successful MLIR compilation."""
        test_file = tmp_path / "test.morph"
        test_file.write_text("x = 42")

        args = Mock()
        args.file = test_file
        args.dialect = 'all'

        with patch('morphogen.parser.parser.parse') as mock_parse, \
             patch('morphogen.mlir.compiler.MLIRCompiler') as mock_compiler_class, \
             patch('morphogen.mlir.optimizer.create_default_pipeline') as mock_pipeline_fn:

            mock_program = Mock()
            mock_parse.return_value = mock_program

            mock_module = Mock()
            mock_module.verify.return_value = None
            mock_module.__str__ = lambda self: "MLIR IR Content"

            mock_compiler = Mock()
            mock_compiler.compile_program.return_value = mock_module
            mock_compiler_class.return_value = mock_compiler

            mock_pipeline = Mock()
            mock_pipeline.optimize.return_value = mock_module
            mock_pipeline_fn.return_value = mock_pipeline

            cmd_mlir(args)

            captured = capsys.readouterr()
            assert 'MLIR IR' in captured.out
            assert '✓ MLIR compilation successful' in captured.out

    def test_mlir_verification_warning(self, tmp_path, capsys):
        """Test MLIR compilation with verification warning."""
        test_file = tmp_path / "test.morph"
        test_file.write_text("x = 42")

        args = Mock()
        args.file = test_file
        args.dialect = 'all'

        with patch('morphogen.parser.parser.parse') as mock_parse, \
             patch('morphogen.mlir.compiler.MLIRCompiler') as mock_compiler_class, \
             patch('morphogen.mlir.optimizer.create_default_pipeline') as mock_pipeline_fn:

            mock_program = Mock()
            mock_parse.return_value = mock_program

            mock_module = Mock()
            mock_module.verify.side_effect = ValueError("Verification failed")
            mock_module.__str__ = lambda self: "MLIR IR Content"

            mock_compiler = Mock()
            mock_compiler.compile_program.return_value = mock_module
            mock_compiler_class.return_value = mock_compiler

            mock_pipeline = Mock()
            mock_pipeline.optimize.return_value = mock_module
            mock_pipeline_fn.return_value = mock_pipeline

            cmd_mlir(args)

            captured = capsys.readouterr()
            assert 'Warning: IR verification failed' in captured.out

    def test_mlir_compilation_error(self, tmp_path, capsys):
        """Test handling of MLIR compilation errors."""
        test_file = tmp_path / "test.morph"
        test_file.write_text("x = 42")

        args = Mock()
        args.file = test_file
        args.dialect = 'all'

        with patch('morphogen.parser.parser.parse') as mock_parse, \
             patch('morphogen.mlir.compiler.MLIRCompiler') as mock_compiler_class:

            mock_program = Mock()
            mock_parse.return_value = mock_program

            mock_compiler = Mock()
            mock_compiler.compile_program.side_effect = Exception("Compilation failed")
            mock_compiler_class.return_value = mock_compiler

            with pytest.raises(SystemExit) as exc_info:
                cmd_mlir(args)

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert 'MLIR compilation error' in captured.err


class TestCmdVersion:
    """Tests for the version command."""

    def test_version_output(self, capsys):
        """Test version command output."""
        args = Mock()

        # Morphogen is the current module name (renamed from kairo/Creative Computation DSL)
        try:
            import morphogen
            patch_target = 'morphogen.__version__'
        except ImportError:
            pytest.skip("morphogen module not importable")

        with patch(patch_target, '0.2.2'):
            cmd_version(args)
            captured = capsys.readouterr()
            assert 'v0.2.2' in captured.out
            assert 'Morphogen' in captured.out or 'Creative Computation DSL' in captured.out


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_parameter_parsing_integer(self, tmp_path):
        """Test that integer parameters are parsed correctly."""
        test_file = tmp_path / "test.morph"
        test_file.write_text("x = 42")

        args = Mock()
        args.file = test_file
        args.profile = 'medium'
        args.seed = 42
        args.steps = None
        args.param = ['count=100']

        with patch('morphogen.parser.parser.parse') as mock_parse, \
             patch('morphogen.ast.visitors.TypeChecker') as mock_checker, \
             patch('morphogen.runtime.runtime.ExecutionContext') as mock_context_class, \
             patch('morphogen.runtime.runtime.Runtime'):

            mock_program = Mock()
            mock_program.statements = []
            mock_parse.return_value = mock_program

            mock_checker_instance = Mock()
            mock_checker_instance.errors = []
            mock_checker.return_value = mock_checker_instance

            mock_context = Mock()
            mock_context_class.return_value = mock_context

            cmd_run(args)

            # Find the call with 'count'
            config_calls = mock_context.set_config.call_args_list
            count_call = [call for call in config_calls if call[0][0] == 'count'][0]

            # Should be integer, not float
            assert count_call[0][1] == 100
            assert isinstance(count_call[0][1], int)

    def test_parameter_parsing_float(self, tmp_path):
        """Test that float parameters are parsed correctly."""
        test_file = tmp_path / "test.morph"
        test_file.write_text("x = 42")

        args = Mock()
        args.file = test_file
        args.profile = 'medium'
        args.seed = 42
        args.steps = None
        args.param = ['rate=0.5']

        with patch('morphogen.parser.parser.parse') as mock_parse, \
             patch('morphogen.ast.visitors.TypeChecker') as mock_checker, \
             patch('morphogen.runtime.runtime.ExecutionContext') as mock_context_class, \
             patch('morphogen.runtime.runtime.Runtime'):

            mock_program = Mock()
            mock_program.statements = []
            mock_parse.return_value = mock_program

            mock_checker_instance = Mock()
            mock_checker_instance.errors = []
            mock_checker.return_value = mock_checker_instance

            mock_context = Mock()
            mock_context_class.return_value = mock_context

            cmd_run(args)

            config_calls = mock_context.set_config.call_args_list
            rate_call = [call for call in config_calls if call[0][0] == 'rate'][0]

            assert rate_call[0][1] == 0.5
            assert isinstance(rate_call[0][1], float)

    def test_parameter_parsing_string(self, tmp_path):
        """Test that string parameters remain as strings."""
        test_file = tmp_path / "test.morph"
        test_file.write_text("x = 42")

        args = Mock()
        args.file = test_file
        args.profile = 'medium'
        args.seed = 42
        args.steps = None
        args.param = ['name=test']

        with patch('morphogen.parser.parser.parse') as mock_parse, \
             patch('morphogen.ast.visitors.TypeChecker') as mock_checker, \
             patch('morphogen.runtime.runtime.ExecutionContext') as mock_context_class, \
             patch('morphogen.runtime.runtime.Runtime'):

            mock_program = Mock()
            mock_program.statements = []
            mock_parse.return_value = mock_program

            mock_checker_instance = Mock()
            mock_checker_instance.errors = []
            mock_checker.return_value = mock_checker_instance

            mock_context = Mock()
            mock_context_class.return_value = mock_context

            cmd_run(args)

            config_calls = mock_context.set_config.call_args_list
            name_call = [call for call in config_calls if call[0][0] == 'name'][0]

            assert name_call[0][1] == 'test'
            assert isinstance(name_call[0][1], str)

    def test_multiple_step_blocks_warning(self, tmp_path, capsys):
        """Test warning when multiple step blocks are present."""
        from morphogen.ast.nodes import Step

        test_file = tmp_path / "test.morph"
        test_file.write_text("step { } step { }")

        args = Mock()
        args.file = test_file
        args.profile = 'medium'
        args.seed = 42
        args.steps = 1
        args.param = None

        with patch('morphogen.parser.parser.parse') as mock_parse, \
             patch('morphogen.ast.visitors.TypeChecker') as mock_checker, \
             patch('morphogen.runtime.runtime.Runtime') as mock_runtime_class:

            step_block1 = Step(body=[])
            step_block2 = Step(body=[])

            mock_program = Mock()
            mock_program.statements = [step_block1, step_block2]
            mock_parse.return_value = mock_program

            mock_checker_instance = Mock()
            mock_checker_instance.errors = []
            mock_checker.return_value = mock_checker_instance

            mock_runtime = Mock()
            mock_runtime_class.return_value = mock_runtime

            cmd_run(args)

            captured = capsys.readouterr()
            assert 'Multiple step blocks found' in captured.out


class TestIntegrationWithRealFiles:
    """Integration tests using real example files."""

    def test_parse_real_example(self):
        """Test parsing a real example file."""
        examples_dir = Path(__file__).parent.parent / "examples"
        example_file = examples_dir / "01_hello_heat.morph"

        if not example_file.exists():
            pytest.skip("Example file not found")

        args = Mock()
        args.file = example_file
        args.format = 'tree'

        # This should not raise an exception
        # We're just testing that the parsing pipeline works
        with patch('morphogen.ast.visitors.ASTPrinter') as mock_printer_class:
            mock_printer = Mock()
            mock_printer.visit.return_value = "AST Tree"
            mock_printer_class.return_value = mock_printer

            cmd_parse(args)

            # Verify ASTPrinter was instantiated and used
            assert mock_printer_class.called
            assert mock_printer.visit.called

    def test_check_real_example(self):
        """Test type-checking a real example file."""
        examples_dir = Path(__file__).parent.parent / "examples"
        example_file = examples_dir / "01_hello_heat.morph"

        if not example_file.exists():
            pytest.skip("Example file not found")

        args = Mock()
        args.file = example_file
        args.strict = False

        # This should parse and type-check without errors
        # (assuming the example file is valid)
        try:
            cmd_check(args)
        except SystemExit as e:
            # If it exits, it should be with code 0 (success) or 1 (type errors)
            # We don't want unexpected exceptions
            assert e.code in [0, 1]
