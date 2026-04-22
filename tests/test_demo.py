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

def test_day():
    topology_df = pd.read_csv(f"data/homework3bus.csv")
    print(topology_df)
    price_df = pd.read_csv(f"data/pricedf_0096WD_7_N001_fall_2025_10_10.csv")
    print(price_df)
    demandDf = create_all_nodes_demand(topology_df, "fall")
    print(demandDf)

    dayOptResults = run_day_optimization(
        topology_df=topology_df,
        demand_df=demandDf,
        price_df=price_df,
        total_battery_capacity=20.0
    )

    print(dayOptResults)

    # save dayOptResults to csv
    variables_df = pd.DataFrame(dayOptResults.variables)
    variables_df.to_csv("day_optimization_results.csv", index=False)
    