import numpy as np
import pandas as pd

from tests.test_three_node_case import run_test_case_and_solve
from vpp_controller.optimization import formulate_vpp_problem
from vpp_controller.paths import DATA_NETWORKS_DIR
from vpp_controller.runner import (
    build_model_inputs,
)


def test_hw3():
    """Test the optimization to see if we get the same as hw 3 when we run 1 timestep with no battery."""

    topology_df = pd.read_csv(DATA_NETWORKS_DIR / "homework3bus.csv")
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
    result = run_test_case_and_solve(
        formulation, eta_ch=0.95, eta_dis=0.95, delta_t=1.0
    )

    variables = result.variables

    # Check results against expected values from hw 3
    assert np.isclose(result.objective_value, 299.79)

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
        )
        ** 2,
        atol=0.001,
    ).all()
