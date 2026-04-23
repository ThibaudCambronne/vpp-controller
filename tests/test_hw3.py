import cvxpy as cp
import numpy as np
import cvxpy as cp
import numpy as np
import pandas as pd

from vpp_controller.optimization import VPPFormulation, formulate_vpp_problem
from vpp_controller.runner import run_day_optimization, DayOptimizationResult, solve_formulation_problem
from vpp_controller.demand_data import (
    SEASON_DATES,
    build_season_base_demand,
    create_all_nodes_demand,
    create_node_demand,
    load_demand_data,
)

def test_hw3():
    topology_df = pd.read_csv(f"data/homework3bus.csv")
    print(topology_df)
    prices = c = [100.0, 0.0, 0.0, 150.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0, 0.0]
