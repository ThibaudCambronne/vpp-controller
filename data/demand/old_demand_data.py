import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

## Read data and prepare it 
base_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_path, 'CISO-demand.csv')
df = pd.read_csv(file_path, sep=';')
df['UTC time'] = pd.to_datetime(df['UTC time'], format='mixed', dayfirst=True)
df = df[df['UTC time'].dt.year == 2025]
cols = ['UTC time', 'Hour', 'Time zone', 'Demand forecast', 'Demand', 'Net generation', 'Subregion PGAE']
df = df[cols]

## Filter for specific dates             
dates = ['2025-01-10','2025-04-10', '2025-07-10', '2025-10-10']     ### can be changed!
df_seasons = df[df['UTC time'].dt.date.isin(pd.to_datetime(dates).date)][['UTC time', 'Subregion PGAE']].reset_index(drop=True)


## Defining the functions

def create_node_demand(node_row, base_demand):
    """
    Returns: pd.DataFrame with columns: node, P_demand, Q_demand
    """
    node  = node_row['node']
    P_mean = node_row['l_P']
    Q_mean = node_row['l_Q']

    # Scale to node mean values
    p_scaled = (base_demand / base_demand.mean()) * P_mean
    q_scaled = (base_demand / base_demand.mean()) * Q_mean

    # Noise: 5% of mean demand, different seed for P and Q
    std_p = P_mean * 0.05
    std_q = Q_mean * 0.05

    rng_p = np.random.default_rng(seed=int(node))
    rng_q = np.random.default_rng(seed=int(node) + 10)

    p_noisy = p_scaled + rng_p.normal(loc=0, scale=std_p, size=len(p_scaled))
    q_noisy = q_scaled + rng_q.normal(loc=0, scale=std_q, size=len(q_scaled))

    return pd.DataFrame({
        'node':     node,
        'P_demand': p_noisy,
        'Q_demand': q_noisy
    })

SEASON_DATES = {
    'winter': '2025-01-10',
    'spring': '2025-04-10',
    'summer': '2025-07-10',
    'fall':   '2025-10-10'
}

def create_all_nodes_demand(nodes_df, season):
    """
    Returns: pd.DataFrame of all nodes with columns: timestamp, node, P_demand, Q_demand for a given season
    """
    if season not in SEASON_DATES:
        raise ValueError(f"Season must be one of: {list(SEASON_DATES.keys())}")

    # Select base demand for the given season
    date = SEASON_DATES[season]
    mask = df_seasons['UTC time'].dt.date == pd.Timestamp(date).date()
    df_day = df_seasons[mask].reset_index(drop=True)

    if df_day.empty:
        raise ValueError(f"No data found for date {date}")

    base_demand = df_day['Subregion PGAE'].values
    timestamps  = df_day['UTC time'].values

    # Iterate over all nodes and collect demand dfs
    all_nodes = []
    for _, node_row in nodes_df.iterrows():
        df_node = create_node_demand(node_row, base_demand)
        df_node.insert(0, 'timestamp', timestamps)
        all_nodes.append(df_node)

    return pd.concat(all_nodes, ignore_index=True)


### Test with artificial nodes_df
'''
nodes_df = pd.DataFrame({
    'node': [1, 2, 3, 4, 5],
    'l_P':  [3.2, 5.1, 2.8, 4.5, 1.9],  # MW
    'l_Q':  [1.5, 2.3, 1.2, 2.0, 0.9],  # MVAr
})

print(nodes_df)

# --- Example usage ---
df_demand = create_all_nodes_demand(nodes_df, season='spring')
print(df_demand.head(100))

'''

## Code to create plots for the report
'''
# Plot with the demand from all 4 seasons
fig, ax = plt.subplots(figsize=(14, 5))

for date in dates:
    mask = df_seasons['UTC time'].dt.date == pd.Timestamp(date).date()
    data = df_seasons[mask]
    ax.plot(data['UTC time'].dt.hour, data['Subregion PGAE'], linewidth=1.5, marker='o', markersize=3, label=date)

ax.set_title('Subregion PGAE – Demand during all Seasons')
ax.set_xlabel('Hour')
ax.set_ylabel('PGAE (GW)')
ax.set_xticks(range(0, 24))
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Plot with demand and noise on 10.04.2025
mask = df_seasons['UTC time'].dt.date == pd.Timestamp('2025-04-10').date()
df_day = df_seasons[mask].reset_index(drop=True)

base = df_day['Subregion PGAE'].values
std  = base.mean() * 0.05

rng   = np.random.default_rng(seed=1)
noisy = base + rng.normal(loc=0, scale=std, size=len(base))

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df_day['UTC time'].dt.hour, base,  color='steelblue', linewidth=1.5, label='Original')
ax.plot(df_day['UTC time'].dt.hour, noisy, color='tomato',    linewidth=1,   label='With Noise (5% of mean)', alpha=0.7)

ax.set_title('Subregion PGAE – Demand on 10. April 2025')
ax.set_xlabel('Hour')
ax.set_ylabel('Demand (GW)')
ax.set_xticks(range(0, 24))
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

'''

