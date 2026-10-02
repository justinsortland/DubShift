"""DubShift command-line interface."""

import typer

from dubshift import __version__

app = typer.Typer(
    name="dubshift",
    help="Replace the spoken voice in educational videos while preserving "
    "the original video's timing and visual content.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """DubShift: voice replacement for educational videos."""


@app.command()
def version() -> None:
    """Print the installed DubShift version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
