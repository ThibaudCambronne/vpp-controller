from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import cvxpy as cp
import numpy as np


@dataclass(frozen=True)
class VPPFormulation:
    """Container for a formulated VPP optimization problem."""

    problem: cp.Problem
    variables: Dict[str, Any]
    constraints: Dict[str, List[cp.Constraint]]
    dimensions: Dict[str, int]


def formulate_vpp_problem(
    N: Sequence[int],
    E: Sequence[Tuple[int, int]],
    T: np.ndarray,
    rho: Mapping[int, int],
    l_P: np.ndarray,
    l_Q: np.ndarray,
    c: np.ndarray,
    s_max: np.ndarray,
    r: np.ndarray,
    x: np.ndarray,
    I_max: np.ndarray,
    v_min: float,
    v_max: float,
    eta_batt: float,
    alpha: float,
    delta_t: float,
    e_0: float,
    e_batt_max: float,
    mu_P: float,
    mu_Q: float,
    v_0: float = 1.0,
) -> VPPFormulation:
    """
    Build the convex OPF + battery placement model with LaTeX-aligned symbols.

    Variable keys in the returned dictionary intentionally match paper notation:
    - p_{i,t}, q_{i,t}, s_{i,t}, P_{ij,t}, Q_{ij,t}, L_{ij,t}, V_{i,t}
    - delta^P_{i,t}, delta^Q_{i,t}
    - P^{batt}_{j,t}, Q^{batt}_{j,t}, S^{batt}_{j,t}, e_{j,t}
    - P^{batt}_{j,max}, e^{batt}_{j,max}
    """

    node_ids = list(N)
    edge_ids = list(E)
    time_ids = list(T)

    n_nodes = len(node_ids)
    n_edges = len(edge_ids)
    n_time = len(time_ids)

    _validate_inputs(
        n_nodes=n_nodes,
        n_edges=n_edges,
        n_time=n_time,
        rho=rho,
        l_P=l_P,
        l_Q=l_Q,
        c=c,
        s_max=s_max,
        r=r,
        x=x,
        I_max=I_max,
        v_0=v_0,
    )

    node_to_idx = {node: idx for idx, node in enumerate(node_ids)}
    edge_to_idx = {(i, j): idx for idx, (i, j) in enumerate(edge_ids)}

    # construct adjency matrix
    A = np.zeros((n_nodes, n_nodes))
    for i in node_ids:
        A[node_to_idx[rho[i]], node_to_idx[i]] = 1
    A[0, 0] = 0

    children_by_node: Dict[int, List[int]] = {node: [] for node in node_ids}
    parent_edge_by_node: Dict[int, int] = {}
    for edge_idx, (i, j) in enumerate(edge_ids):
        children_by_node[i].append(j)
        parent_edge_by_node[j] = edge_idx

    root_candidates = [node for node in node_ids if node not in parent_edge_by_node]
    if len(root_candidates) != 1:
        raise ValueError("Expected exactly one root node in radial feeder.")
    root_node = root_candidates[0]
    root_node_idx = node_to_idx[root_node]
    assert root_node_idx == 0, "Root node must be the first node."
    assert rho[root_node] == 0, "Root node must have rho value of 0."
    assert A[root_node_idx, root_node_idx] == 0, "Root node cannot have a self-loop."

    p = cp.Variable((n_nodes, n_time), nonneg=True, name="p_{i,t}")  # (5)
    q = cp.Variable((n_nodes, n_time), nonneg=True, name="q_{i,t}")  # (5)
    s = cp.Variable((n_nodes, n_time), nonneg=True, name="s_{i,t}")

    P = cp.Variable((n_nodes, n_nodes, n_time), name="P_{ij,t}")
    Q = cp.Variable((n_nodes, n_nodes, n_time), name="Q_{ij,t}")
    L = cp.Variable((n_nodes, n_nodes, n_time), nonneg=True, name="L_{ij,t}")
    V = cp.Variable((n_nodes, n_time), name="V_{i,t}")

    delta_P_pos = cp.Variable((n_nodes, n_time), nonneg=True, name="delta^P_{i,t,+}")
    delta_P_neg = cp.Variable((n_nodes, n_time), nonneg=True, name="delta^P_{i,t,-}")
    delta_Q_pos = cp.Variable((n_nodes, n_time), nonneg=True, name="delta^Q_{i,t,+}")
    delta_Q_neg = cp.Variable((n_nodes, n_time), nonneg=True, name="delta^Q_{i,t,-}")

    P_batt = cp.Variable((n_nodes, n_time), name="P^{batt}_{j,t}")
    Q_batt = cp.Variable((n_nodes, n_time), name="Q^{batt}_{j,t}")
    S_batt = cp.Variable((n_nodes, n_time), nonneg=True, name="S^{batt}_{j,t}")
    e = cp.Variable((n_nodes, n_time + 1), name="e_{j,t}")

    P_batt_max = cp.Variable(n_nodes, nonneg=True, name="P^{batt}_{j,max}")
    e_batt_max_by_node = cp.Variable(n_nodes, nonneg=True, name="e^{batt}_{j,max}")

    delta_P = delta_P_pos - delta_P_neg
    delta_Q = delta_Q_pos - delta_Q_neg

    constraints: Dict[str, List[cp.Constraint]] = {}

    constraints["voltage_bounds"] = []
    constraints["generator_apparent_power"] = []
    constraints["generation_apparent_power_cap"] = []
    constraints["battery_inverter_apparent_power"] = []
    constraints["battery_inverter_capacity"] = []
    constraints["battery_energy_limits"] = []
    constraints["battery_cycle"] = []
    constraints["battery_energy_dynamics"] = []
    constraints["battery_no_reactive_power"] = []
    for node in node_ids:
        j_idx = node_to_idx[node]  # node index in 0-based ordering
        for t_idx in range(n_time):
            # Voltage bounds at each node and time step (8)
            constraints["voltage_bounds"].append(V[j_idx, t_idx] >= v_min**2)
            constraints["voltage_bounds"].append(V[j_idx, t_idx] <= v_max**2)

            # Generator apparent power definition, relaxed (6)
            constraints["generator_apparent_power"].append(
                cp.norm(cp.hstack([p[j_idx, t_idx], q[j_idx, t_idx]]), 2)
                <= s[j_idx, t_idx]
            )

            # Generator capacity (7)
            constraints["generation_apparent_power_cap"].append(
                s[j_idx, t_idx] <= s_max[j_idx]
            )

            # Battery inverter apparent power definition, relaxed (13)
            constraints["battery_inverter_apparent_power"].append(
                cp.norm(cp.hstack([P_batt[j_idx, t_idx], Q_batt[j_idx, t_idx]]), 2)
                <= S_batt[j_idx, t_idx]
            )

            # TODO: test
            # Battery inverter apparent power definition, relaxed (13)
            # constraints["battery_no_reactive_power"].append(Q_batt[j_idx, t_idx] == 0.0)

            # Battery inverter capacity (14)
            constraints["battery_inverter_capacity"].append(
                S_batt[j_idx, t_idx] <= P_batt_max[j_idx]
            )

            # Battery energy limits (11)
            constraints["battery_energy_limits"].append(e[j_idx, t_idx] >= 0.0)
            constraints["battery_energy_limits"].append(
                e[j_idx, t_idx] <= e_batt_max_by_node[j_idx]
            )

        # Battery energy limits, at time T (11)
        constraints["battery_energy_limits"].append(e[j_idx, n_time] >= 0.0)
        constraints["battery_energy_limits"].append(
            e[j_idx, n_time] <= e_batt_max_by_node[j_idx]
        )

        # Battery cycle constraint (12)
        constraints["battery_cycle"].append(e[j_idx, 0] == e_0)
        constraints["battery_cycle"].append(e[j_idx, n_time] == e_0)

        # Battery energy dynamics (10)
        for t_idx in range(n_time):
            constraints["battery_energy_dynamics"].append(
                e[j_idx, t_idx + 1]
                == e[j_idx, t_idx] - eta_batt * P_batt[j_idx, t_idx] * delta_t
            )

    constraints["active_power_balance"] = [P[0, 0, :] == 0.0]
    constraints["reactive_power_balance"] = [Q[0, 0, :] == 0.0]
    constraints["voltage_balance"] = [V[0, :] == v_0**2]
    constraints["thermal_limits"] = []
    constraints["current_relation_relaxed"] = []
    for node in node_ids:
        j_idx = node_to_idx[node]
        i_idx = node_to_idx[rho[node]]

        if j_idx == root_node_idx:
            rij = 0
            xij = 0
            I_max_ij = 0
        else:
            edge_idx = edge_to_idx[(rho[node], node)]
            rij = r[edge_idx]
            xij = x[edge_idx]
            I_max_ij = I_max[edge_idx]

        # Active power balance constraint (1)
        constraints["active_power_balance"].append(
            P[i_idx, j_idx, :]
            == (
                l_P[j_idx, :]
                - p[j_idx, :]
                - P_batt[j_idx, :]
                + rij * L[i_idx, j_idx, :]
                + A[j_idx, :] @ P[j_idx, :, :]
                + delta_P[j_idx, :]
            )
        )

        # Reactive power balance constraint (2)
        constraints["reactive_power_balance"].append(
            Q[i_idx, j_idx, :]
            == (
                l_Q[j_idx, :]
                - q[j_idx, :]
                - Q_batt[j_idx, :]
                + xij * L[i_idx, j_idx, :]
                + A[j_idx, :] @ Q[j_idx, :, :]
                + delta_Q[j_idx, :]
            )
        )

        # Voltage balance constraint (3)
        constraints["voltage_balance"].append(
            V[j_idx, :]
            == V[i_idx, :]
            + (rij**2 + xij**2) * L[i_idx, j_idx, :]
            - 2.0 * (rij * P[i_idx, j_idx, :] + xij * Q[i_idx, j_idx, :])
        )

        # Thermal limit constraint (9)
        constraints["thermal_limits"].append(L[i_idx, j_idx, :] <= I_max_ij**2)

        # Current relation, relaxed (4)
        for t_idx in range(n_time):
            constraints["current_relation_relaxed"].append(
                L[i_idx, j_idx, t_idx]
                >= cp.quad_over_lin(
                    cp.hstack([P[i_idx, j_idx, t_idx], Q[i_idx, j_idx, t_idx]]),
                    V[i_idx, t_idx],
                )
            )

    # Battery energy-power link (15)
    constraints["battery_energy_power_link"] = []
    for node in node_ids:
        j_idx = node_to_idx[node]
        constraints["battery_energy_power_link"].append(
            e_batt_max_by_node[j_idx] == alpha * P_batt_max[j_idx]
        )

    # Battery total capacity constraint (16)
    constraints["battery_total_capacity"] = [cp.sum(e_batt_max_by_node) <= e_batt_max]  # type: ignore

    # No battery at root node
    constraints["no_battery_at_root"] = [e_batt_max_by_node[root_node_idx] == 0.0]

    generation_cost = cp.sum(cp.multiply(c, p))
    imbalance_cost = mu_P * cp.sum(delta_P_pos + delta_P_neg) + mu_Q * cp.sum(
        delta_Q_pos + delta_Q_neg
    )
    objective = cp.Minimize(generation_cost + imbalance_cost)

    all_constraints = [con for group in constraints.values() for con in group]
    problem = cp.Problem(objective, all_constraints)

    variables: Dict[str, Any] = {
        "p_{i,t}": p,
        "q_{i,t}": q,
        "s_{i,t}": s,
        "P_{ij,t}": P,
        "Q_{ij,t}": Q,
        "L_{ij,t}": L,
        "V_{i,t}": V,
        "delta^P_{i,t}": delta_P,
        "delta^Q_{i,t}": delta_Q,
        "delta^P_{i,t,+}": delta_P_pos,
        "delta^P_{i,t,-}": delta_P_neg,
        "delta^Q_{i,t,+}": delta_Q_pos,
        "delta^Q_{i,t,-}": delta_Q_neg,
        "P^{batt}_{j,t}": P_batt,
        "Q^{batt}_{j,t}": Q_batt,
        "S^{batt}_{j,t}": S_batt,
        "e_{j,t}": e,
        "P^{batt}_{j,max}": P_batt_max,
        "e^{batt}_{j,max}": e_batt_max_by_node,
    }

    dimensions = {
        "|N|": n_nodes,
        "|E|": n_edges,
        "|T|": n_time,
        "root_node": int(root_node),
    }

    return VPPFormulation(
        problem=problem,
        variables=variables,
        constraints=constraints,
        dimensions=dimensions,
    )


def _validate_inputs(
    n_nodes: int,
    n_edges: int,
    n_time: int,
    rho: Mapping[int, int],
    l_P: np.ndarray,
    l_Q: np.ndarray,
    c: np.ndarray,
    s_max: np.ndarray,
    r: np.ndarray,
    x: np.ndarray,
    I_max: np.ndarray,
    v_0: float,
) -> None:
    if n_nodes == 0:
        raise ValueError("N cannot be empty.")
    if n_edges == 0:
        raise ValueError("E cannot be empty.")
    if n_time == 0:
        raise ValueError("T cannot be empty.")
    if l_P.shape != (n_nodes, n_time):
        raise ValueError("l_P must have shape (|N|, |T|).")
    if l_Q.shape != (n_nodes, n_time):
        raise ValueError("l_Q must have shape (|N|, |T|).")
    if c.shape != (n_nodes, n_time):
        raise ValueError("c must have shape (|N|, |T|).")
    if s_max.shape != (n_nodes,):
        raise ValueError("s_max must have shape (|N|,).")
    if r.shape != (n_edges,):
        raise ValueError("r must have shape (|E|,).")
    if x.shape != (n_edges,):
        raise ValueError("x must have shape (|E|,).")
    if I_max.shape != (n_edges,):
        raise ValueError("I_max must have shape (|E|,).")
    if len(rho) != n_nodes:
        raise ValueError("rho must contain one entry per node.")
    if v_0 != 1.0:
        raise ValueError(
            "v_0 should most likely be 1.0 p.u.."
            "If it is not the case, think about why and review the code."
        )
