"""
CLI entry point using Click + Rich.
"""
# Load .env into os.environ before any LangChain imports so that
# LANGCHAIN_TRACING_V2 / LANGCHAIN_API_KEY are picked up for LangSmith tracing.
from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="auto-sdet")
def cli():
    """Auto-SDET: Autonomous Unit Test Generation Agent."""
    pass


@cli.command()
@click.argument("target", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--max-retries",
    default=3,
    type=int,
    show_default=True,
    help="Maximum self-healing retry attempts.",
)
@click.option(
    "--model",
    default="deepseek-chat",
    show_default=True,
    help="DeepSeek model name.",
)
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(path_type=Path),
    help="Output directory for generated test file. [default: same as target]",
)
@click.option("--verbose", is_flag=True, default=False, help="Show detailed node logs.")
def test(target: Path, max_retries: int, model: str, output_dir: Path | None, verbose: bool):
    """
    Generate unit tests for TARGET source file.

    Example:
        auto-sdet test src/calculator.py --max-retries 5
    """
    # ── Validate target file ────────────────────────────
    target = target.resolve()
    if not target.suffix == ".py":
        console.print("[bold red]Error:[/] Target must be a .py file.")
        sys.exit(1)

    # ── Determine output path ───────────────────────────
    if output_dir is None:
        output_dir = target.parent
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"test_{target.name}"

    # ── Display banner ──────────────────────────────────
    console.print(Panel.fit(
        f"[bold cyan]Auto-SDET[/]  |  target: [bold]{target.name}[/]\n"
        f"output: {output_path}\n"
        f"max retries: {max_retries}  |  model: {model}",
        border_style="cyan",
    ))

    # ── Run the LangGraph agent ─────────────────────────
    try:
        from auto_sdet.graph.graph import run_agent

        result = run_agent(
            target_path=target,
            output_path=output_path,
            max_retries=max_retries,
            model_name=model,
            verbose=verbose,
        )

        # ── Handle result ───────────────────────────────
        if result["status"] == "done":
            er = result.get("execution_result")
            cov_str = f"  |  Coverage: [bold cyan]{er.coverage_pct}%[/]" if er and er.coverage_pct is not None else ""
            console.print(f"\n[bold green]✓[/] Tests generated: {escape(str(output_path))}{cov_str}")
        else:
            console.print(f"\n[bold red]✗[/] Failed after {result['retry_count']} retries.")
            if result.get("execution_result"):
                er = result['execution_result']
                console.print(f"[dim]Last error:[/]\n{escape((er.stdout + er.stderr)[:500])}")
            sys.exit(1)

    except Exception as e:
        console.print(f"[bold red]Error:[/] {escape(str(e))}")
        if verbose:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    cli()
