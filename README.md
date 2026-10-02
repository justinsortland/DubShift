# DubShift

DubShift replaces the spoken voice in educational videos while preserving
the original video's timing and visual content.

**Status:** bootstrap only. The CLI exists and runs, but it does not yet
download videos, process audio, or convert voices — that comes in later
milestones.

## Requirements

- Python 3.12+
- [FFmpeg](https://ffmpeg.org/) (required by a later milestone; not used yet)

## Install (local development)

```bash
# from the project root
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

## Run

```bash
dubshift --help
dubshift version
```

If you don't want to install the package, you can also run it directly:

```bash
python -m dubshift.cli --help
```

## Configuration

Copy `.env.example` to `.env` and fill in any values you need. `.env` is
gitignored and should never be committed. No API keys are required yet —
this milestone only sets up the CLI skeleton.

```bash
cp .env.example .env
```

## Tests

```bash
pytest
```

## Project layout

```
src/dubshift/      # application package
  cli.py           # Typer CLI entry point
tests/             # pytest test suite
```

## Voice usage

DubShift is designed around voices you own or have permission to use.
It does not bundle or distribute third-party character or celebrity
voice models.

## Roadmap

See `CLAUDE.md` for the full architecture and milestone plan. Upcoming
milestones add YouTube downloading, audio extraction, speech-to-speech
voice conversion, and dubbed video output.
