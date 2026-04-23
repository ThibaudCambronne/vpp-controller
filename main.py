import pandas as pd

# import sys
# from pathlib import Path
# ROOT = Path(__file__).resolve().parent
# SRC = ROOT / "src"
# if str(SRC) not in sys.path:
#     sys.path.insert(0, str(SRC))
from src.vpp_controller.demand_data import create_all_nodes_demand
from src.vpp_controller.paths import DATA_DIR
from src.vpp_controller.runner import run_day_optimization


def main() -> None:
    topology_df = pd.read_csv(DATA_DIR / "homework3bus no gen.csv")
    print(topology_df)

    price_df = pd.read_csv(DATA_DIR / "pricedf_0096WD_7_N001_fall_2025_10_10.csv")
    price_df = price_df.sort_values(by="OPR_HR").reset_index(drop=True)
    print(price_df)

    demand_df = create_all_nodes_demand(topology_df, "fall")
    demand_df = demand_df.rename(columns={"P_demand": "l_P", "Q_demand": "l_Q"})
    demand_df["hour"] = pd.to_datetime(demand_df["timestamp"]).dt.hour
    demand_df = demand_df[["node", "hour", "l_P", "l_Q"]]

    day_opt_results = run_day_optimization(
        topology_df=topology_df,
        demand_df=demand_df,
        price_df=price_df,
        total_battery_capacity=20.0,
    )

    for key, value in day_opt_results.variables.items():
        print(f" {key}: {value.round(1)}")

    pass


if __name__ == "__main__":
    main()
