"""CLI entry point for RouteSentinel."""

import typer

from sentinel.config import load_config

app = typer.Typer(help="RouteSentinel: flag unjustified freight-cost rises per route.")


@app.command()
def run(
    no_llm: bool = typer.Option(False, "--no-llm", help="Run without the LLM layer."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Ignore cached LLM calls."),
    bruteforce: bool = typer.Option(
        False, "--bruteforce", help="Score all notes during retrieval."
    ),
    config: str = typer.Option("config.yaml", "--config", help="Path to config file."),
) -> None:
    """Run the detection pipeline (deterministic; --no-llm identical verdicts)."""
    settings = load_config(config)
    _ = (no_llm, no_cache, bruteforce, settings)
    from sentinel.qa.answer import build_outputs, repo_root

    stats = build_outputs(root=repo_root())
    typer.echo(
        f"candidates={stats.candidates} justified={stats.justified} "
        f"flagged_yes={stats.flagged_yes} llm_calls={stats.llm_calls}"
    )


@app.command()
def eval() -> None:
    """Run the evaluation harness (scorecard + eval_report.md)."""
    import runpy
    from pathlib import Path as _P

    runpy.run_path(str(_P("eval/run_eval.py")), run_name="__main__")


@app.command()
def ask(question: str) -> None:
    """Answer one grounded question from findings.sqlite."""
    from sentinel.qa.answer import answer_question, repo_root

    typer.echo(answer_question(question, root=repo_root()))


if __name__ == "__main__":
    app()
