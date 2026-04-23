import pandas as pd

from vpp_controller.demand_data import (
    create_all_nodes_demand,
)
from vpp_controller.runner import run_day_optimization


def test_day():
    topology_df = pd.read_csv("data/homework3bus.csv")
    print(topology_df)
    price_df = pd.read_csv("data/pricedf_0096WD_7_N001_fall_2025_10_10.csv")
    print(price_df)
    demandDf = create_all_nodes_demand(topology_df, "fall")
    print(demandDf)

    dayOptResults = run_day_optimization(
        topology_df=topology_df,
        demand_df=demandDf,
        price_df_root_node=price_df,
        total_battery_capacity=20.0,
    )

    print(dayOptResults)

    # save dayOptResults to csv
    variables_df = pd.DataFrame(dayOptResults.variables)
    variables_df.to_csv("day_optimization_results.csv", index=False)
