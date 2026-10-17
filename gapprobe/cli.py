"""gapprobe/cli.py — Typer CLI entry point."""

from __future__ import annotations

import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from gapprobe import __version__

app = typer.Typer(
    name="gapprobe",
    help="Capability-gap diagnostician for locally-served CN open-weights models.",
    no_args_is_help=True,
)
console = Console()

_VALID_LAYERS = ("chat-template", "chat_template", "sampling")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"gapprobe {__version__}")
        raise typer.Exit()


@app.callback()
def root(
    version: bool = typer.Option(
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Capability-gap diagnostician for locally-served CN open-weights models."""


@app.command()
def probe(
    endpoint: str = typer.Option(..., "--endpoint", "-e", help="Local server URL (e.g. http://localhost:8080)"),
    layer: Optional[str] = typer.Option(
        None, "--layer", "-l", help="Probe only one layer: chat-template | sampling"
    ),
):
    """Probe a local model server and emit a capability-gap verdict."""
    from gapprobe.probe import ProbeEngine
    from gapprobe.registry import get_entry
    from gapprobe.layers.chat_template import ChatTemplateLayer
    from gapprobe.layers.sampling import SamplingLayer
    from gapprobe.verdict import assemble_verdict

    console.print(f"[dim]GapProbe v{__version__} — probing {endpoint}[/dim]")

    if layer is not None and layer not in _VALID_LAYERS:
        console.print(
            f"[red]✗ Unknown --layer '{layer}'. Valid: chat-template | sampling[/red]"
        )
        raise typer.Exit(2)

    with ProbeEngine(endpoint) as engine:
        try:
            model_key = engine.detect_model()
        except Exception as exc:
            console.print(f"[red]✗ Model detection failed: {exc}[/red]")
            raise typer.Exit(1)

        entry = get_entry(model_key)
        if not entry:
            console.print(f"[red]✗ Model '{model_key}' not in registry[/red]")
            raise typer.Exit(1)

        console.print(f"[green]✓ Detected: {engine.model_id} (registry: {model_key})[/green]")

        layer_verdicts = []
        fix_hints = []

        if layer in (None, "chat-template", "chat_template"):
            console.print("[dim]Probing chat-template layer…[/dim]")
            ct = ChatTemplateLayer().run(engine, entry, model_key)
            layer_verdicts.append(ct)
            fix_hints.append(entry.fix_hint_chat_template)
            _print_layer_table(ct)

        if layer in (None, "sampling"):
            console.print("[dim]Probing sampling layer…[/dim]")
            sp = SamplingLayer().run(engine, entry, model_key)
            layer_verdicts.append(sp)
            fix_hints.append(entry.fix_hint_sampling)
            _print_layer_table(sp)

    # Assemble cross-layer verdict
    if layer_verdicts and all(v.inconclusive for v in layer_verdicts):
        # Every probe died at the HTTP level: prescribing template/sampler
        # fixes would misattribute a server failure to the model config.
        suggested_fix = "resolve server errors, then re-run gapprobe probe"
    else:
        suggested_fix = " | ".join(h for h, v in zip(fix_hints, layer_verdicts) if v.is_misconfigured)
    if not suggested_fix:
        suggested_fix = "no fix needed — configuration looks correct"

    verdict = assemble_verdict(
        model=engine.model_id,
        endpoint=endpoint,
        layer_verdicts=layer_verdicts,
        suggested_fix=suggested_fix,
    )

    console.print()
    console.print(f"[bold cyan]━━━ Verdict ━━━[/bold cyan]")
    console.print(f"[bold]{verdict.one_liner}[/bold]")


@app.command()
def bench():
    """Run the 20-case attribution bench."""
    from gapprobe.cases import run_bench

    correct, total, details = run_bench()
    pct = correct / total * 100 if total else 0

    table = Table(title="GapProbe 20-Case Attribution Bench")
    table.add_column("Case")
    table.add_column("Predicted")
    table.add_column("Expected")
    table.add_column("✓/✗")
    for case_id, predicted, expected in details:
        ok = "✓" if predicted == expected else "✗"
        table.add_row(case_id, predicted or "none", expected or "none", ok)

    console.print(table)
    console.print()
    console.print(
        f"[bold {'green' if pct > 50 else 'red'}]"
        f"Attribution accuracy: {correct}/{total} ({pct:.0f}%)"
        f" — {'beats' if pct > 50 else 'FAILS'} random guess (>50%)[/bold {'green' if pct > 50 else 'red'}]"
    )


def _print_layer_table(verdict):
    from gapprobe.verdict import LayerVerdict

    table = Table(title=f"Layer: {verdict.layer}")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("divergence_score", f"{verdict.divergence_score:.3f}")
    table.add_row("est_capability_loss", f"{verdict.est_capability_loss_pct:.1f}%")
    table.add_row("evidence", "\n".join(verdict.evidence[:3]))
    console.print(table)


def main():
    app()


if __name__ == "__main__":
    main()
