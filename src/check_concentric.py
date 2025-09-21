import os
import numpy as np
from data_manager import load_obj

def is_concentric_distribution(expected_distance):
    for base in range(len(expected_distance)//2-1):
        is_consentric = False
        for reference in range(len(expected_distance)):
            if expected_distance[base][reference] < expected_distance[base][len(expected_distance)//2] and reference > len(expected_distance)//2:
                is_consentric = True
        if not is_consentric:
            # print(f"base: {base}")
            # print(f"center value: {expected_distance[base][len(expected_distance)//2]}")
            # print(f"row: {expected_distance[base]}")
            return False
    return True
def is_concentric_distribution_per_agent(expected_distance):
    # 各baseごとにTrue/Falseを返すリスト
    result = []
    center = len(expected_distance)//2
    for base in range(len(expected_distance)):
        is_consentric = False
        for reference in range(len(expected_distance)):
            if abs(reference - base) > abs(center - base) and expected_distance[base][reference] < expected_distance[base][center] and (base < center) != (reference < center):
                is_consentric = True
        result.append(is_consentric)
    return result

def check_concentric_distribution_time_series(distance):
    # distance: shape = (time, agent, agent)
    time_length = distance.shape[0]
    agent_length = distance.shape[1]
    counts = np.zeros((time_length, agent_length), dtype=int)
    for t in range(time_length):
        per_agent = is_concentric_distribution_per_agent(distance[t])
        counts[t] = per_agent
    return counts

# --- 設定 ---
DATA_DIR = os.environ.get('ONEDRIVE', os.path.expanduser('~/Downloads')) + "/SekiLabo/res_language/ilm/data"
PLOT_OBJS = "expected_distance"

# unique_argsの例（compare_simulations.pyやmain_exdi.pyを参考）
alpha_per_data=0.001
fr = 0.01
data_size = 100
alpha = data_size*alpha_per_data
unique_args = {
    "simulation_count": [ 1000000],
    "agents_count": [15],
    # "simulate_type":["markov_chain"],
    "simulate_type":["monte_carlo"],
    "agent": [ "BayesianInfiniteVariantsAgent"],
    "agents_arguments": [{
    # "alpha":1/7,  
    "alpha":alpha,
    "data_size":data_size,
    "nonzero_alpha":"evely"
},{
    # "alpha":1/7,
    "alpha":alpha,
    "data_size":data_size,
    "nonzero_alpha":"center"
} ],
    "network_args": [{
    "bidirectional_flow_rate": fr,
}, {
    "outward_flow_rate": fr,
},
],
}

setting_count = np.prod([len(unique_args[key]) for key in unique_args.keys()])
args = [{} for _ in range(setting_count)]
for i in range(setting_count):
    for ki, key in enumerate(unique_args.keys()):
        args[i][key] = unique_args[key][int(i // np.prod([len(unique_args[key]) for key in list(unique_args.keys())[ki+1:]])) % len(unique_args[key])]

for i in range(setting_count):
    setting_name = ''
    for key in unique_args.keys():
        setting_name += f"{key}_{args[i][key]}_"
    setting_name = setting_name.replace(":", "_")
    dir_path = DATA_DIR + '/raw/' + setting_name
    print(f"Checking: {setting_name}")
    true_count = 0
    total_count = 0
    agent_true_counts = None
    if os.path.exists(dir_path):
        simulation_version = 0
        max_versions = 204
        while simulation_version < max_versions:
            try:
                rec = load_obj(dir_path, PLOT_OBJS, number=simulation_version)
                distance = rec["distance"]
                time_counts = check_concentric_distribution_time_series(distance)
                if agent_true_counts is None:
                    agent_true_counts = np.zeros((time_counts.shape[0], time_counts.shape[1]), dtype=int)
                agent_true_counts += time_counts
                total_count += 1
            except Exception as e:
                pass
            simulation_version += 1
        print(f"concentric_distribution_count per agent per time: {agent_true_counts} / {total_count}")
    else:
        print(f"Data not found: {dir_path}")
print(DATA_DIR)