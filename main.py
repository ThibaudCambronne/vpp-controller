import numpy as np
import pandas as pd

# import sys
# from pathlib import Path
# ROOT = Path(__file__).resolve().parent
# SRC = ROOT / "src"
# if str(SRC) not in sys.path:
#     sys.path.insert(0, str(SRC))
from src.vpp_controller.demand_data import create_all_nodes_demand
from src.vpp_controller.paths import DATA_NETWORKS_DIR, DATA_PRICES_DIR
from src.vpp_controller.results_format import save_day_optimization_result
from src.vpp_controller.runner import run_day_optimization

opVersion = 1


def main() -> None:
    topology_df = pd.read_csv(DATA_NETWORKS_DIR / "homework3bus no gen.csv")
    print(topology_df)

    price_df = pd.read_csv(
        DATA_PRICES_DIR / "pricedf_0096WD_7_N001_fall_2025_10_10.csv"
    )
    price_df = price_df.sort_values(by="OPR_HR").reset_index(drop=True)
    print(price_df)

    demand_df = create_all_nodes_demand(topology_df, "fall", factor=0.7)
    demand_df = demand_df.rename(columns={"P_demand": "l_P", "Q_demand": "l_Q"})
    demand_df["hour"] = pd.to_datetime(demand_df["timestamp"]).dt.hour
    demand_df = demand_df[["node", "hour", "l_P", "l_Q"]]

    print(demand_df)

    # create list from 0 to 20 iterating by 5
    batt_caps = list(np.arange(0, 21, 1))

    for batt_cap in batt_caps:
        print("\n" + "=" * 50)
        print(f"Running optimization with battery capacity: {batt_cap} kWh")

        batt_cap = float(batt_cap)

        dayOptResults = run_day_optimization(
            topology_df=topology_df,
            demand_df=demand_df,
            price_df_root_node=price_df,
            total_battery_capacity=batt_cap,
        )

        for key, value in dayOptResults.variables.items():
            print(f" {key}: {value.round(1)}")

        metadata_path, variables_path = save_day_optimization_result(
            dayOptResults, batt_cap, opVersion
        )


if __name__ == "__main__":
    main()
