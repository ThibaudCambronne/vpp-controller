from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vpp_controller.paths import OUTPUT_DIR
from vpp_controller.runner import DayOptimizationResult


def _to_jsonable(value):
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def save_day_optimization_result(
    day_opt_results: DayOptimizationResult,
    battCap: float,
    opVersion: int,
    output_dir: Path = OUTPUT_DIR,
):
    """
    Save optimization outputs in a reusable format.

    - variables -> compressed NPZ for efficient numeric storage
    - status/objective/diagnostics/duals -> JSON for readability
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    run_tag = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    variables_path = (
        output_dir / f"day_opt_variables_battCap{battCap}_v_{opVersion}.npz"
    )
    np.savez_compressed(variables_path, **day_opt_results.variables)

    metadata = {
        "status": day_opt_results.status,
        "objective_value": day_opt_results.objective_value,
        "diagnostics": _to_jsonable(day_opt_results.diagnostics),
        "variables_file": variables_path.name,
        "variable_shapes": {
            name: list(np.asarray(values).shape)
            for name, values in day_opt_results.variables.items()
        },
        "duals": {
            group: [np.asarray(arr).tolist() for arr in dual_list]
            for group, dual_list in day_opt_results.duals.items()
        },
    }

    metadata_path = output_dir / f"day_opt_metadata_battCap{battCap}_v_{opVersion}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved metadata to: {metadata_path}")
    print(f"Saved variables to: {variables_path}")

    return metadata_path, variables_path


def load_day_optimization_result(metadata_path: str | Path):
    """
    Load a saved optimization result bundle.

    Returns a dictionary with metadata and variables loaded from the NPZ file.
    """
    metadata_path = Path(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    variables_path = metadata_path.with_name(metadata["variables_file"])

    with np.load(variables_path, allow_pickle=False) as variables_file:
        variables = {key: variables_file[key] for key in variables_file.files}

    return {
        "status": metadata.get("status"),
        "objective_value": metadata.get("objective_value"),
        "diagnostics": metadata.get("diagnostics", {}),
        "duals": metadata.get("duals", {}),
        "variables": variables,
        "variable_shapes": metadata.get("variable_shapes", {}),
    }


__all__ = [
    "load_day_optimization_result",
    "save_day_optimization_result",
]
