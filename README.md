# vpp-controller

## Project structure

Place your source code under `src/vpp_controller/`.

Place your datasets and other input/output files under `data/`.

Example layout:

```text
vpp-controller/
    data/
    src/
        vpp_controller/
            __init__.py
            paths.py
            config.py
```

## Global variables and shared settings

Define shared global variables (for example, folder paths, global parameters) in `src/vpp_controller/paths.py` or `src/vpp_controller/config.py`.

Avoid scattering hardcoded paths in multiple modules. Keep them centralized in one of these files so they are easy to update and reuse.

## Quick start with uv

This project uses `pyproject.toml`, so `uv` can create the environment and install dependencies in one step.

```bash
# 1) Create/sync the local .venv from pyproject.toml
uv sync

# 2) Run the project
uv run python main.py

# 3) Run tests
uv run pytest -q
```

If your machine does not have a compatible Python version yet (`>=3.13`), install one first:

```bash
uv python install 3.13
uv sync
```
