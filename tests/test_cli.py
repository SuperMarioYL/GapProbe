"""tests/test_cli.py — CLI surface tests (validation, --version)."""

from __future__ import annotations

from typer.testing import CliRunner

from gapprobe import __version__
from gapprobe.cli import app

runner = CliRunner()


class TestLayerValidation:
    def test_unknown_layer_exits_2(self):
        result = runner.invoke(
            app, ["probe", "--endpoint", "http://localhost:8080", "--layer", "tokenizer"]
        )
        assert result.exit_code == 2
        assert "Unknown --layer 'tokenizer'" in result.output
        assert "chat-template" in result.output
        assert "sampling" in result.output

    def test_error_does_not_blame_endpoint(self):
        result = runner.invoke(
            app, ["probe", "--endpoint", "http://localhost:8080", "--layer", "tokenizer"]
        )
        # The old failure mode reported "no probes ran — check endpoint".
        assert "check endpoint" not in result.output


class TestVersionFlag:
    def test_version_prints_package_version(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert f"gapprobe {__version__}" in result.output

    def test_version_flag_works_before_subcommands(self):
        # is_eager: `gapprobe --version probe ...` must still short-circuit.
        result = runner.invoke(app, ["--version", "probe"])
        assert result.exit_code == 0
        assert f"gapprobe {__version__}" in result.output


class TestBenchCommand:
    def test_bench_beats_random_guess(self):
        result = runner.invoke(app, ["bench"])
        assert result.exit_code == 0
        assert "20/20" in result.output
