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

    Model:      |0| --(3 MW)--> |1| --(0.5 MW)--> |2|
    demand t1:   0 MW            1 MW              0 MW
    demand t2:   0 MW            0.5 MW            1 MW
    """
    T = 2
    hours = np.arange(T)
    n_set = [0, 1, 2]
    e_set = [(0, 1), (1, 2)]

    rho = {0: 0, 1: 0, 2: 1}

    l_p = np.array([[0, 0], [1, 0.5], [0, 1]])
    l_q = 0 * l_p

    c0 = np.array(
        [10.0, 20.0],
    )
    c = np.vstack([c0, np.zeros_like(c0), np.zeros_like(c0)])

    I_max = np.array([3.0, 0.5])

    battery_duration = 2.0  # hours
    max_energy_capacity = 0.7 * battery_duration

    eta_ch = 0.95
    eta_dis = 0.95

    formulation = formulate_vpp_problem(
        N=n_set,
        E=e_set,
        T=hours,
        rho=rho,
        l_P=l_p,
        l_Q=l_q,
        c=c,
        s_max=np.array([10.0, 0.0, 0.0]),
        r=np.array([0.015, 0.03]),
        x=np.array([0.025, 0.04]),
        I_max=I_max,
        v_min=0.95,
        v_max=1.05,
        eta_ch=eta_ch,
        eta_dis=eta_dis,
        alpha=battery_duration,
        delta_t=1.0,
        e_0=0.0,
        e_batt_max=max_energy_capacity,
        mu_P=200.0,
        mu_Q=80.0,
        v_0=1.0,
    )

    result = run_test_case_and_solve(
        formulation,
        eta_ch=eta_ch,
        eta_dis=eta_dis,
        delta_t=1.0,
    )

    # Check that battery is placed at node 2
    P_batt_j_max = result.variables["P^{batt}_{j,max}"]
    assert P_batt_j_max[2] > 0.45, (
        "Battery of around 0.5 MW should be placed at node 2."
    )

    # Check that node battery charges around 0.5 MW at t=0 and discharges at t=1.
    P_ch = result.variables["P^{ch}_{j,t}"]
    P_dis = result.variables["P^{dis}_{j,t}"]
    P_batt = result.variables["P^{batt}_{j,t}"]
    assert P_ch[2, 0] > 0.45, "Battery should charge around 0.5 MW at t=0."
    assert P_dis[2, 1] > 0.42, "Battery should discharge around 0.5 MW at t=1."
    assert np.allclose(P_batt, P_dis - P_ch, atol=1e-6), (
        "Net battery injection must satisfy P_batt = P_dis - P_ch."
    )

    # Check that there is only generation at node 0 that is above 1.5 MW at t=0
    # to supply 1MW of demand of node 0, 0.5 MW of battery charge for node 2, around 0.2 MW of battery charge for node 1 because
    # electricity is cheaper at t=0 than t=1
    s = result.variables["s_{i,t}"]
    assert s[0, 0] > 1.7, "Generation at node 0 should be around 1.5 MW at t=0."
    assert s[0, 1] > 0.6, "Generation at node 0 should be around 1 MW at t=1."
    assert np.all(s[1:, :] < 1e-5), "There should be no generation at nodes 1 and 2."

    # assert that there is congestion on the second edge at t=1,
    # because we need to charge the battery there to meet the demand at t=2
    assert (result.duals["thermal_limits"][2] >= 1).all(), (
        "There should be congestion on the line between node 1 and node 2 at t=1."
    )


def run_test_case_and_solve(
    formulation: VPPFormulation,
    *,
    eta_ch: float,
    eta_dis: float,
    delta_t: float,
) -> DayOptimizationResult:
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
        "P^{ch}_{j,t}",
        "P^{dis}_{j,t}",
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

    assert np.isclose(e[0, :], 0.0).all(), (
        "Battery SOC at node 0 should be zero in 3-node test case."
    )

    cycle_gap = np.max(np.abs(e[:, 0] - e[:, -1]))
    assert cycle_gap <= 1e-5, "Battery cycle constraint failed in 3-node test case."

    P_ch = result.variables["P^{ch}_{j,t}"]
    P_dis = result.variables["P^{dis}_{j,t}"]
    assert np.min(P_ch) >= -1e-7, "Charging power must be nonnegative."
    assert np.min(P_dis) >= -1e-7, "Discharging power must be nonnegative."

    soc_residual = (
        e[:, 1:] - e[:, :-1] - (eta_ch * P_ch - (1.0 / eta_dis) * P_dis) * delta_t
    )
    assert np.max(np.abs(soc_residual)) <= 1e-5, (
        "Battery energy dynamics equation residual is too large."
    )

    v = result.variables["V_{i,t}"]
    assert np.min(v) >= 0.95**2 - 1e-5, (
        "Voltage lower bound violated in 3-node test case."
    )
    assert np.max(v) <= 1.05**2 + 1e-5, (
        "Voltage upper bound violated in 3-node test case."
    )

    return result
