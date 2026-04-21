import cvxpy as cp
import numpy as np

from vpp_controller.optimization import VPPFormulation, formulate_vpp_problem
from vpp_controller.runner import DayOptimizationResult, solve_formulation_problem


def test_1():
    """Test case with 3 nodes, 2 edges, and 2 time steps.
    Here, we make a problem where the optimal placement for the battery is on node 2.
    We need to put all of the available capacity there to be able to meet the demand at timestep 2,
    because of congestion on the line between node 1 and node 2.

    The test checks that the battery is indeed placed at node 2, and that the cycle constraint is satisfied.
    """
    T = 2
    hours = np.arange(T)
    n_set = [0, 1, 2]
    e_set = [(0, 1), (1, 2)]
    adjacency = np.zeros((3, 3))
    adjacency[0, 1] = 1
    adjacency[1, 2] = 1

    rho = {0: -1, 1: 0, 2: 1}

    l_p = np.array([[0, 0], [1, 1], [0, 1]])
    l_q = 0 * l_p

    c0 = np.array(
        [10.0, 20.0],
    )
    c = np.vstack([c0, np.zeros_like(c0), np.zeros_like(c0)])

    I_max = np.array([2.0, 1.2])

    battery_duration = 4.0  # hours
    max_energy_capacity = 1 * battery_duration

    formulation = formulate_vpp_problem(
        N=n_set,
        E=e_set,
        T=hours,
        rho=rho,
        A=adjacency,
        l_P=l_p,
        l_Q=l_q,
        c=c,
        s_max=np.array([10.0, 0.0, 0.0]),
        r=np.array([0.015, 0.03]),
        x=np.array([0.025, 0.04]),
        I_max=I_max,
        v_min=0.95,
        v_max=1.05,
        eta_batt=0.95,
        alpha=battery_duration,
        delta_t=1.0,
        e_0=0.0,
        e_batt_max=max_energy_capacity,
        mu_P=200.0,
        mu_Q=80.0,
        v_0=1.0,
    )

    result = run_test_case_and_solve(formulation)


def run_test_case_and_solve(formulation: VPPFormulation) -> DayOptimizationResult:
    solve_formulation_problem(formulation.problem)

    result = DayOptimizationResult(
        status=formulation.problem.status,
        objective_value=formulation.problem.value,
        variables={
            name: np.array(var.value) if hasattr(var, "value") else np.array(var)
            for name, var in formulation.variables.items()
        },
        duals={
            group: [np.array(con.dual_value) for con in group_constraints]
            for group, group_constraints in formulation.constraints.items()
        },
        diagnostics={
            "solver": formulation.problem.solver_stats.solver_name,
            "solve_time": formulation.problem.solver_stats.solve_time,
            "num_iters": formulation.problem.solver_stats.num_iters,
            "dimensions": formulation.dimensions,
        },
    )

    assert result.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}, (
        f"3-node case did not solve to optimality: {result.status}"
    )
    assert result.objective_value is not None

    for key in (
        "p_{i,t}",
        "q_{i,t}",
        "s_{i,t}",
        "P_{ij,t}",
        "Q_{ij,t}",
        "L_{ij,t}",
        "V_{i,t}",
        "delta^P_{i,t}",
        "delta^Q_{i,t}",
        "P^{batt}_{j,t}",
        "Q^{batt}_{j,t}",
        "S^{batt}_{j,t}",
        "e_{j,t}",
        "P^{batt}_{j,max}",
        "e^{batt}_{j,max}",
    ):
        assert key in result.variables

    T = formulation.dimensions["|T|"]

    e = result.variables["e_{j,t}"]
    assert e.shape == (3, T + 1), "Unexpected SOC dimensions for 3-node test case."

    cycle_gap = np.max(np.abs(e[:, 0] - e[:, -1]))
    assert cycle_gap <= 1e-5, "Battery cycle constraint failed in 3-node test case."

    v = result.variables["V_{i,t}"]
    assert np.min(v) >= 0.95**2 - 1e-5, (
        "Voltage lower bound violated in 3-node test case."
    )
    assert np.max(v) <= 1.05**2 + 1e-5, (
        "Voltage upper bound violated in 3-node test case."
    )

    return result
