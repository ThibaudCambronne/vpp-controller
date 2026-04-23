import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.patches import Patch

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
# from src.vpp_controller.demand_data import create_all_nodes_demand
from src.vpp_controller.paths import OUTPUT_DIR, FIGURE_PATH
# from src.vpp_controller.runner import run_day_optimization
from src.results_format import load_day_optimization_result


def create_output_folder(date_str):
    OUTPATH = FIGURE_PATH / date_str
    OUTPATH.mkdir(parents=True, exist_ok=True)


def get_results_dict(json_path):
    results = load_day_optimization_result(json_path)
    return results


def plot_objective_vs_capacity(results_list,OUT_PATH):
    """Scatter/line plot: objective value (y) vs total battery capacity constraint (x).

    Args:
        results_list: list of results dicts, one per JSON file for a given date.
    """
    capacities = [sum(r['variables']['e^{batt}_{j,max}']) for r in results_list]
    objectives = [r['objective_value'] for r in results_list]

    pairs = sorted(zip(capacities, objectives))
    capacities_sorted, objectives_sorted = zip(*pairs)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(capacities_sorted, objectives_sorted, marker='o', linewidth=1.5)
    ax.set_xlabel('Total Battery Capacity Constraint (MWh)')
    ax.set_ylabel('Objective Value')
    ax.set_title('Objective Value vs Battery Capacity Constraint')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT_PATH / 'objective_vs_capacity.png', dpi=150, bbox_inches='tight')
    # plt.show()
    return


def plot_capacity_allocation(results_list,OUT_PATH):
    """Grouped bar chart: per-node share of total capacity, one group per node, one bar per scenario.

    Each bar shows e^{batt}_{j,max}[node] / sum(e^{batt}_{j,max}) so bars within a
    scenario sum to 1, letting you compare how allocation shifts across capacity levels.

    Args:
        results_list: list of results dicts, one per JSON file for a given date.
    """
    sorted_results = sorted(results_list, key=lambda r: sum(r['variables']['e^{batt}_{j,max}']))
    total_caps = [sum(r['variables']['e^{batt}_{j,max}']) for r in sorted_results]
    n_scenarios = len(sorted_results)
    n_nodes = len(sorted_results[0]['variables']['e^{batt}_{j,max}'])
    nodes = np.arange(n_nodes)

    bar_width = 0.8 / n_scenarios
    fig, ax = plt.subplots(figsize=(max(10, n_nodes * 1.5), 6))

    for i, (r, total) in enumerate(zip(sorted_results, total_caps)):
        fractions = [v / total for v in r['variables']['e^{batt}_{j,max}']]
        offsets = nodes + (i - n_scenarios / 2 + 0.5) * bar_width
        ax.bar(offsets, fractions, width=bar_width, label=f'Cap = {total:.1f}')

    ax.set_xlabel('Node')
    ax.set_ylabel('Fraction of Total Capacity')
    ax.set_title('Battery Capacity Allocation by Node')
    ax.set_xticks(nodes)
    ax.set_xticklabels([str(i) for i in range(n_nodes)])
    ax.legend(title='Total Capacity (MWh)', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT_PATH / 'capacity_allocation_by_node.png', dpi=150, bbox_inches='tight')
    # plt.show()
    return


def plot_battery_dispatch_3d(results_dict,OUT_PATH):
    """3D bar chart of battery charge/discharge by node (x) and hour (z), value on y-axis.

    Charging bars (positive P) are blue; discharging bars (negative P) are red.
    Assumes results['variables']['P^{batt}_{j,t}'] is a list of n_nodes lists,
    each of length n_hours, indexed as [node][hour].

    Args:
        results_dict: single results dict for the scenario to visualise.
    """
    P_batt = results_dict['variables']['P^{batt}_{j,t}']
    dispatch = np.array(P_batt)  # shape: (n_nodes, n_hours)
    n_nodes, n_hours = dispatch.shape

    node_idx, hour_idx = np.meshgrid(np.arange(n_nodes), np.arange(n_hours), indexing='ij')
    node_flat = node_idx.ravel().astype(float)
    hour_flat = hour_idx.ravel().astype(float)
    values_flat = dispatch.ravel()

    dx = 0.6
    dy = 0.6
    dz = np.abs(values_flat)
    # For negative values the bar grows downward: bottom = value, top = 0
    z_bottoms = np.where(values_flat >= 0, 0.0, values_flat)
    colors = ['steelblue' if v >= 0 else 'tomato' for v in values_flat]

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(
        node_flat - dx / 2,
        hour_flat - dy / 2,
        z_bottoms,
        dx, dy, dz,
        color=colors,
        alpha=0.85,
        zsort='average',
    )

    ax.set_xlabel('Node')
    ax.set_zlabel('Charge / Discharge (MW)')
    ax.set_ylabel('Hour')
    ax.set_title('Battery Dispatch by Node and Hour')
    ax.set_xticks(range(n_nodes))
    ax.set_yticks(range(0, n_hours, max(1, n_hours // 12)))

    legend_elements = [
        Patch(facecolor='steelblue', label='Charging (+)'),
        Patch(facecolor='tomato', label='Discharging (−)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    plt.tight_layout()
    total_cap = sum(results_dict['variables']['e^{batt}_{j,max}'])
    plt.savefig(OUT_PATH / f'battery_dispatch_3d_cap{total_cap:.0f}MWh.png', dpi=150, bbox_inches='tight')
    # plt.show()
    return


def main(date_string):
    OUT_PATH = FIGURE_PATH / date_string
    create_output_folder(OUT_PATH)
    json_files = find_files_by_date(date_string)
    results_list = []

    for file in json_files:
        results_dict = get_results_dict(file)
        plot_battery_dispatch_3d(results_dict,OUT_PATH)
        results_list.append(results_dict)
    plot_objective_vs_capacity(results_list,OUT_PATH)
    plot_capacity_allocation(results_list,OUT_PATH)
    return

def find_files_by_date(date_str):
    json_files = []
    # Find and print matching files
    for file_path in OUTPUT_DIR.rglob('*.json'):
        try:
            # Read file content as a string
            if date_str in file_path.read_text(encoding='utf-8'):
                json_files.append(file_path)
        except (UnicodeDecodeError, PermissionError):
            continue
    # json_paths = OUTPUT_DIR / f'day_opt_metadata_20260422_175358.json'
    return json_files


if __name__ == '__main__':
    date_string = "20260422_175358"
    main(date_string)