import cvxpy as cp
import numpy as np
import cvxpy as cp
import numpy as np
import pandas as pd

from vpp_controller.optimization import VPPFormulation, formulate_vpp_problem
from vpp_controller.runner import run_day_optimization, DayOptimizationResult, solve_formulation_problem


def test_day():
    topology_df = pd.read_csv(f"data/homework3bus.csv")
    print(topology_df)
    price_df = pd.read_csv(f"data/0096WD_7_N001_fall_2025-10-10.csv")
    print(price_df)
    # run_day_optimization(
    #     day = "day",
    #     total_battery_capacity=20.0,
    #     network_ieee_case="homework3bus"        
    # )
