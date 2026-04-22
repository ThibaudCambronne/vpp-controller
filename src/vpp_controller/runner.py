from __future__ import annotations

import ast
import importlib
from dataclasses import dataclass
from typing import Any, Dict, Tuple, cast

import cvxpy as cp
import numpy as np
import pandas as pd

from .optimization import formulate_vpp_problem

REQUIRED_PROVIDER_FUNCTIONS = (
    "get_network_topology_and_parameters",
    "get_daily_node_demand",
    "get_daily_price_curve",
)


@dataclass(frozen=True)
class DayOptimizationResult:
    """Structured outputs for a solved day-level optimization."""

    status: str
    objective_value: float | None
    variables: Dict[str, np.ndarray]
    duals: Dict[str, list[np.ndarray]]
    diagnostics: Dict[str, Any]


# TODO: Define unit of battery capacity
def run_day_optimization(
    topology_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    price_df: pd.DataFrame,
    total_battery_capacity: float,
    # day: str,
    # total_battery_capacity: float,
    # network_ieee_case: str,
) -> DayOptimizationResult:
    """
    Run optimization for one day.
    """

    # providers = _load_provider_module("vpp_controller.data_sources")

    # topology_df = pd.read_csv(f"data/{network_ieee_case}.csv")
    # print(topology_df)
    # providers.get_network_topology_and_parameters(network_ieee_case)
    # demand_df = providers.get_daily_node_demand(day, network_ieee_case)

    # price_series = providers.get_daily_price_curve(day, network_ieee_case)
    
    # Get price_series from price_df column "$/MW"
    price_series = price_df["$/MW"].to_numpy(dtype=float)
    
    model_inputs = _build_model_inputs(
        topology_df=topology_df,
        demand_df=demand_df,
        price_series=price_series,
        total_battery_capacity=total_battery_capacity,
    )

    formulation = formulate_vpp_problem(**model_inputs)
    solve_formulation_problem(formulation.problem)

    duals = {
        group: [np.array(con.dual_value) for con in group_constraints]
        for group, group_constraints in formulation.constraints.items()
    }

    variables = {
        name: np.array(var.value) if hasattr(var, "value") else np.array(var)
        for name, var in formulation.variables.items()
    }

    diagnostics = {
        "solver": formulation.problem.solver_stats.solver_name,
        "solve_time": formulation.problem.solver_stats.solve_time,
        "num_iters": formulation.problem.solver_stats.num_iters,
        "dimensions": formulation.dimensions,
    }

    return DayOptimizationResult(
        status=formulation.problem.status,
        objective_value=cast(float | None, formulation.problem.value),
        variables=variables,
        duals=duals,
        diagnostics=diagnostics,
    )


def _load_provider_module(module_name: str):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        required = ", ".join(REQUIRED_PROVIDER_FUNCTIONS)
        raise NotImplementedError(
            f"Missing provider module '{module_name}'. Add it with functions: {required}."
        ) from exc

    missing = [fn for fn in REQUIRED_PROVIDER_FUNCTIONS if not hasattr(module, fn)]
    if missing:
        missing_str = ", ".join(missing)
        raise NotImplementedError(
            f"Provider module '{module_name}' is missing functions: {missing_str}."
        )

    return module


def _build_model_inputs(
    *,
    topology_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    price_series: pd.Series | np.ndarray,
    total_battery_capacity: float,
) -> Dict[str, Any]:
    node_col = _detect_node_column(topology_df)
    nodes = topology_df[node_col].to_numpy(dtype=int)
    n_nodes = len(nodes)

    r_matrix = _parse_vector_column(topology_df["r"], n_nodes)
    x_matrix = _parse_vector_column(topology_df["x"], n_nodes)
    i_max_matrix = _parse_vector_column(topology_df["I_max"], n_nodes)

    edges: list[Tuple[int, int]] = []
    r_list: list[float] = []
    x_list: list[float] = []
    i_max_list: list[float] = []

    for i in range(n_nodes):
        for j in range(n_nodes):
            if r_matrix[i, j] > 0.0 or x_matrix[i, j] > 0.0 or i_max_matrix[i, j] > 0.0:
                edges.append((int(nodes[i]), int(nodes[j])))
                r_list.append(float(r_matrix[i, j]))
                x_list.append(float(x_matrix[i, j]))
                i_max_list.append(float(i_max_matrix[i, j]))

    A = np.zeros((n_nodes, n_nodes))
    for i, j in edges:
        A[int(i), int(j)] = 1.0

    rho = {int(i): 0 for i in nodes}
    for i, j in edges:
        rho[int(j)] = int(i)

    hourly_l_P, hourly_l_Q = _extract_hourly_demand(demand_df, n_nodes=n_nodes)

    price = np.asarray(price_series, dtype=float).reshape(-1)
    if price.shape[0] != hourly_l_P.shape[1]:
        raise ValueError("Price curve length must match number of time steps.")

    c = np.zeros_like(hourly_l_P)
    c[0, :] = price

    s_max = topology_df["s_max"].to_numpy(dtype=float)
    if s_max.shape[0] != n_nodes:
        raise ValueError("s_max length does not match number of nodes.")

    v_min = float(topology_df["v_min"].iloc[0])
    v_max = float(topology_df["v_max"].iloc[0])

    return {
        "N": list(nodes),
        "E": edges,
        "T": list(range(hourly_l_P.shape[1])),
        "rho": rho,
        # "A": A,
        "l_P": hourly_l_P,
        "l_Q": hourly_l_Q,
        "c": c,
        "s_max": s_max,
        "r": np.array(r_list),
        "x": np.array(x_list),
        "I_max": np.array(i_max_list),
        "v_min": v_min,
        "v_max": v_max,
        "eta_batt": 0.95,
        "alpha": 2.0,
        "delta_t": 1.0,
        "e_0": 0.0,
        "e_batt_max": float(total_battery_capacity),
        "mu_P": 200.0,
        "mu_Q": 80.0,
        "v_0": 1.0,
    }


def _detect_node_column(topology_df: pd.DataFrame):
    candidate_names = ("node", "bus", "i", "Unnamed: 0")
    for candidate in candidate_names:
        if candidate in topology_df.columns:
            return candidate

    first_col = topology_df.columns[0]
    if pd.api.types.is_integer_dtype(topology_df[first_col]):
        return first_col

    raise ValueError(
        "Could not infer node index column in topology table. "
        "Expected one of: node, bus, i, Unnamed: 0."
    )


def _parse_vector_column(col: pd.Series, width: int) -> np.ndarray:
    rows = []
    for item in col:
        if isinstance(item, str):
            vector = ast.literal_eval(item)
        else:
            vector = item
        arr = np.asarray(vector, dtype=float)
        if arr.shape != (width,):
            raise ValueError("Topology vector column has invalid width.")
        rows.append(arr)
    return np.vstack(rows)


def _extract_hourly_demand(
    demand_df: pd.DataFrame, *, n_nodes: int
) -> Tuple[np.ndarray, np.ndarray]:
    required_cols = {"node", "hour", "l_P", "l_Q"}
    if not required_cols.issubset(set(demand_df.columns)):
        missing = required_cols.difference(set(demand_df.columns))
        raise ValueError(f"Demand table missing required columns: {sorted(missing)}")

    max_hour = int(demand_df["hour"].max())
    n_time = max_hour + 1

    l_P = np.zeros((n_nodes, n_time), dtype=float)
    l_Q = np.zeros((n_nodes, n_time), dtype=float)

    for _, row in demand_df.iterrows():
        node = int(row["node"])
        hour = int(row["hour"])
        l_P[node, hour] = float(row["l_P"])
        l_Q[node, hour] = float(row["l_Q"])

    return l_P, l_Q


def solve_formulation_problem(problem: cp.Problem) -> None:
    # This model is conic; prioritize conic-capable solvers that are commonly installed.
    preferred = ["CLARABEL", "ECOS", "SCS"]
    installed = set(cp.installed_solvers())

    for solver_name in preferred:
        if solver_name in installed:
            problem.solve(solver=solver_name, verbose=False)
            return

    installed_list = ", ".join(sorted(installed)) if installed else "none"
    raise cp.SolverError(
        "No suitable conic solver found. Install one of CLARABEL, ECOS, or SCS. "
        f"Installed solvers: {installed_list}."
    )
