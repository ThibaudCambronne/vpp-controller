import pandas as pd
import numpy as np

import pandapower as pp
import pandapower.networks.power_system_test_cases as power_system_test_cases


def findMeshed(edges):
    parent = {}

    def find(x):
        if parent.setdefault(x, x) != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        
        if root_x == root_y:
            return True  # cycle detected
        
        parent[root_y] = root_x
        return False

    cycle_found = False

    for u, v in edges:
        if union(u, v):
            cycle_found = True
            break

    if cycle_found:
        True
    else:
        False

def formatSystemDf(net):
    df = pd.DataFrame({'node':net.bus.name,
                       'l_P':net.load.p_mw,
                       'l_Q':net.load.q_mvar,
                       'v_min':net.bus.min_vm_pu,
                       'v_max':net.bus.max_vm_pu,
                       })
    df = df.infer_objects().fillna(0)
    if 0 not in list(df.node) and 0 in list(net.line.from_bus):
        df['node'] = df['node'] - 1

    lines = net.line
    lines['total_r'] = lines['r_ohm_per_km']*lines['length_km']
    lines['total_x'] = lines['x_ohm_per_km']*lines['length_km']
    lines['total_r'] = lines['total_r']/max(lines['total_r'])
    lines['total_x'] = lines['total_x']/max(lines['total_x'])
    lines['I_max'] = lines['max_i_ka']/max(lines['max_i_ka'])
    total_nodes = len(df)
    resistance_dict = {i:[0]*total_nodes for i in list(df['node'])}
    reactance_dict = {i:[0]*total_nodes for i in list(df['node'])}
    current_dict = {i:[0]*total_nodes for i in list(df['node'])}
    for idx,line in net.line.iterrows():
        r = line['total_r']
        x = line['total_x']
        i_max = line['I_max']
        first = line['from_bus']
        second = line['to_bus']
        resistance_dict[first][second] = r
        reactance_dict[first][second] = x
        current_dict[first][second] = i_max
    df['r'] = df['node'].map(resistance_dict)
    df['x'] = df['node'].map(reactance_dict)
    df['I_max'] = df['node'].map(current_dict)
    df['s_max'] = 0.
    gen = net.gen
    if gen.empty:
        return None
    df.loc[df['node']==0,'s_max'] = np.sqrt(max(pow(gen['p_mw'],2)+pow(gen[['min_q_mvar', 'max_q_mvar']].max(axis=1),2)))

    #We are removing all meshed networks here
    num_nodes = df['node'].nunique()
    edges = []
    for _, row in df.iterrows():
        parent = row['node']
        for child, val in enumerate(row['r']):
            if val != 0:  # adjust condition if needed
                edges.append((parent, child))
    num_edges = len(edges)
    # if num_edges == num_nodes - 1:
    #     print("Likely radial (if connected)")
    # else:
    #     print("Likely meshed or disconnected")
    if findMeshed(edges):
        return None
    else:
        # df.to_csv('../data/'+ str(f'{func.__name__}')+ '.csv')
        return df

testCaseFuncs = [power_system_test_cases.case4gs,power_system_test_cases.case5,power_system_test_cases.case6ww,power_system_test_cases.case9,
                 power_system_test_cases.case14,power_system_test_cases.case24_ieee_rts,power_system_test_cases.case30,
                 power_system_test_cases.case_ieee30,power_system_test_cases.case33bw,power_system_test_cases.case39,power_system_test_cases.case57,
                 power_system_test_cases.case89pegase,power_system_test_cases.case118,power_system_test_cases.case145,power_system_test_cases.case_illinois200,
                 power_system_test_cases.case300,power_system_test_cases.case1354pegase,power_system_test_cases.case1888rte,power_system_test_cases.case2848rte,
                 power_system_test_cases.case2869pegase,power_system_test_cases.case3120sp,power_system_test_cases.case6470rte,power_system_test_cases.case6495rte,
                 power_system_test_cases.case6515rte,power_system_test_cases.case9241pegase,power_system_test_cases.GBnetwork,power_system_test_cases.GBreducednetwork,
                 power_system_test_cases.iceland,]

if __name__ == '__main__':
    for func in testCaseFuncs:
        net = func()
        try:
            print(func.__name__)
            df = formatSystemDf(net)
            if df is None:
                print('empty???',func.__name__)
            else:
                pass
        except:
            print('bad',func.__name__)