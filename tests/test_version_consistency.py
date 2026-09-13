"""tests/test_version_consistency.py — every version surface must agree.

All checks are in-process (tomllib / json / CliRunner); nothing shells out to
a binary, so no availability guards are needed.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from typer.testing import CliRunner

from gapprobe import __version__
from gapprobe.cli import app

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "0.2.0"
RELEASES = "https://github.com/SuperMarioYL/GapProbe/releases/tag"

runner = CliRunner()


def test_pyproject_version_matches():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == EXPECTED


def test_package_dunder_version_matches():
    assert __version__ == EXPECTED


def test_site_json_content_version_matches():
    site = json.loads((REPO_ROOT / "web" / "site.json").read_text(encoding="utf-8"))
    assert site["meta"]["content_version"] == EXPECTED
    assert site["footer"]["tag"].startswith(f"{EXPECTED} ·")


def test_readme_badges_match():
    for readme in ("README.md", "README.zh-CN.md"):
        text = (REPO_ROOT / readme).read_text(encoding="utf-8")
        assert f"`v{EXPECTED}`" in text


def test_demo_results_version_matches():
    demo = json.loads((REPO_ROOT / "docs" / "demo-results.json").read_text(encoding="utf-8"))
    assert demo["version"] == EXPECTED


def test_changelog_has_current_section_and_link_refs():
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [{EXPECTED}] —" in changelog
    assert f"[{EXPECTED}]: {RELEASES}/v{EXPECTED}" in changelog
    assert f"[0.1.0]: {RELEASES}/v0.1.0" in changelog
    assert (
        f"[Unreleased]: https://github.com/SuperMarioYL/GapProbe/compare/v{EXPECTED}...HEAD"
        in changelog
    )


def test_cli_version_flag_matches():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert f"gapprobe {EXPECTED}" in result.output
