import numpy as np
import pandas as pd

from vpp_controller.optimization import formulate_vpp_problem
from vpp_controller.paths import DATA_DIR
from vpp_controller.runner import (
    DayOptimizationResult,
    build_model_inputs,
    solve_formulation_problem,
)


def test_hw3():
    """Test the optimization to see if we get the same as hw 3 when we run 1 timestep with no battery."""

    topology_df = pd.read_csv(DATA_DIR / "homework3bus.csv")
    print(topology_df.columns)

    # price series is just the price profile for node 0
    price_series = np.array([topology_df["c"].iloc[0]])
    demand_df = pd.DataFrame(
        {
            "node": topology_df["node"].to_numpy(dtype=int),
            "hour": np.zeros(len(topology_df), dtype=int),
            "l_P": topology_df["l_P"].to_numpy(dtype=float),
            "l_Q": topology_df["l_Q"].to_numpy(dtype=float),
        }
    )

    model_inputs = build_model_inputs(
        topology_df=topology_df,
        demand_df=demand_df,
        price_series_root_node=price_series,
        total_battery_capacity=0,
    )

    # add generation cost at all nodes (right now it only has price for node 0)
    model_inputs["c"][:, 0] = topology_df["c"]

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

    # Check results against expected values from hw 3
    assert np.isclose(formulation.problem.value, 299.79)

    assert np.isclose(
        variables["p_{i,t}"][:, 0],
        np.array([1.573, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.934, 0.0, 0.0, 0.0]),
        atol=0.001,
    ).all()

    assert np.isclose(
        variables["q_{i,t}"][:, 0],
        np.array([0.988, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.208, 0.0, 0.0, 0.0]),
        atol=0.001,
    ).all()

    assert np.isclose(
        variables["s_{i,t}"][:, 0],
        np.array([1.858, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.281, 0.0, 0.0, 0.0]),
        atol=0.001,
    ).all()

    # Node   ||  0   |  1   |  2   |  3   |  4   |  5   |  6   |  7   |  8   |  9   |  10  |  11  |  12  |
    # Voltage||1.000 |0.967 |0.963 |0.963 |0.962 |0.960 |0.957 |0.957 |0.957 |0.964 |0.955 |0.954 |0.953 |
    assert np.isclose(
        variables["V_{i,t}"][:, 0],
        np.array(
            [
                1.000,
                0.967,
                0.963,
                0.963,
                0.962,
                0.960,
                0.957,
                0.957,
                0.957,
                0.964,
                0.955,
                0.954,
                0.953,
            ]
        ),
        atol=0.001,
    ).all()

    result = DayOptimizationResult(
        status=formulation.problem.status,
        objective_value=formulation.problem.value,
        variables=variables,
        duals=duals,
        diagnostics=diagnostics,
    )
