# Repository Guidelines

## Project Structure & Module Organization

This repository combines a Python audio-emotion backend with report artifacts. The application code lives in `backend/`: `main.py` exposes the FastAPI service, `train.py` selects encoder/classifier training experiments, `core/` contains model and utility code, `features/` handles audio and dataset processing, and `training/` contains reusable training/evaluation logic. Tests are in `backend/tests/` and are written as standalone Python modules. Trained checkpoints, metrics, and plots are stored under `backend/models/`; report figures are in root `assets/`. Runtime data is intentionally excluded: `data/` and `backend/uploads/` are ignored.

## Build, Test, and Development Commands

Run backend commands from `backend/` so relative `data/` and `models/` paths resolve correctly.

- `pip install -r requirements.txt` installs FastAPI, audio, ML, and plotting dependencies.
- `python main.py` starts the MoodWave API on `0.0.0.0:8000` with reload enabled.
- `python train.py --encoder mel --classifier cnn-bigru --split strict --epochs 80` trains the current CNN-BiGRU multitask model and writes artifacts to `models/`.
- `python train.py --encoder muq --classifier bigru --split strict` trains the MuQ sequence model.
- `python train.py --encoder openl3 --classifier mlp --split paper --eval-only` evaluates an existing checkpoint for the paper split.
- `python tests/test_model.py` runs a single standalone test module.

## Coding Style & Naming Conventions

Use Python with 4-space indentation. Match the existing naming pattern: `snake_case` for modules, functions, and variables; `PascalCase` for model classes such as `MoodCNNBiGRU`; uppercase constants for shared audio settings like `MEL_SAMPLE_RATE`. Keep imports grouped as standard library, third-party packages, then local modules. No formatter, linter, type-checker, or pre-commit configuration is present, so preserve the local style in edited files.

## Testing Guidelines

Tests live in `backend/tests/` and follow `test_*.py` filenames with `test_*` functions using plain `assert`. Each file also has a `__main__` block, so run focused checks directly with `python tests/<file>.py`. There is no recorded coverage target or test runner configuration in the repository.

## Commit & Pull Request Guidelines

Git history currently uses short, freeform subjects such as `fixing model issue` and `change architechure to CNN-Bigru`; no strict convention is enforced. Keep new commit subjects concise and focused on the changed model, training path, or API behavior. No PR template is present; include a brief change summary, commands run, affected artifacts in `backend/models/` or `assets/`, and screenshots or plots when visual results change.
