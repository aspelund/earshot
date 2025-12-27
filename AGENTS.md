# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains the Python STT/TTS servers and audio pipeline logic.
- `tests/` holds pytest-based unit and async tests (e.g., `tests/test_segmentor.py`).
- `scripts/` provides runnable helpers for servers, clients, and diagnostics.
- `rust-earshot/` is the Rust client (GUI + audio I/O) with its own `Cargo.toml`.
- `config*.yaml` are runtime configs; `models/` holds shared model assets.
- `docs/`, `README.md`, and `TESTING.md` document usage and test flows.

## Build, Test, and Development Commands
- Python setup: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
- Install models: `python scripts/install_models.py`.
- Run servers: `bash scripts/run_whisper_server.sh` (STT) and `bash scripts/run_tts_server.sh` (TTS).
- Local pipeline dev: `bash scripts/dev_run.sh`.
- Tests: `pytest tests/` (or per-file: `pytest tests/test_segmentor.py -v`).
- Rust client: `cd rust-earshot && cargo build --release && ./run.sh`.

## Coding Style & Naming Conventions
- Follow existing conventions; keep changes consistent with nearby code.
- Python: 4-space indentation, `snake_case` functions/vars, `PascalCase` classes.
- Rust: use `rustfmt` conventions (`cargo fmt` before larger refactors).
- File naming: Python tests use `tests/test_*.py`.

## Testing Guidelines
- Framework: `pytest` with async tests in `tests/`.
- Prefer focused unit tests for segmenting/VAD logic; add regression tests for bugs.
- Run relevant tests before PRs; see `TESTING.md` for manual checklists.

## Commit & Pull Request Guidelines
- Commit messages in history are short, imperative statements (e.g., “Add …”, “Improve …”).
- PRs should describe scope, list key commands run, and link related issues.
- Include screenshots or short clips for GUI changes (`rust-earshot/`).

## Configuration & Runtime Notes
- Primary config: `config.yaml`; use `config.server.yaml` for network audio sources.
- Models are shared between Python and Rust via `models/` (see `rust-earshot/README.md`).
