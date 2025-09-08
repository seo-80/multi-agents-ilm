import numpy as np
import matplotlib.pyplot as plt
import os
import pickle



import ilm
import data_manager

# 引数の定義
SAVE_RESULT = True
LOAD_RESULT = True
PLOT_RESULT = True
SAVE_STATES = False  # Set to True to save raw simulation data
SAVE_DISTANCES = True  # Set to True to save distance matrices
SAVE_EX_DISTANCE = True  # Set to True to save expected distance matrices
SAVE_KEYS = ["record", "expected_distance", "expected_oldness"]

PLOT_OBJS = "distance"  # Options: "distance" or "oldness"
PLOT_DISTANCE_FROM_ONE = False


DATA_DIR = os.path.dirname(__file__) + "/../data"
# 事前に環境変数ONEDRIVEを設定しておく
DATA_DIR = os.environ['ONEDRIVE'] + "/SekiLabo/res_language/ilm/data"
# 引数の定義

agents_count = 7
data_size = 1
variants_count = 2
alpha = 1
simulation_count = 10000
nonzero_alpha = "evely"  # "evely" or "center"
# if nonzero_alpha == "evely":
#     agents_arguments = [{"alpha": alpha, "data_size": data_size, "variants_count": variants_count} for _ in range(agents_count)]
# elif nonzero_alpha == "center":
#     agents_arguments = [{"alpha": 0, "data_size": data_size, "variants_count": variants_count} for _ in range(agents_count)]
#     agents_arguments[agents_count // 2]["alpha"] = alpha

agents_arguments = {
    "alpha":alpha,
    "data_size":data_size,
    "variants_count":variants_count,
    "nonzero_alpha":nonzero_alpha
}
defalut_args = {
    "simulation_count": 100000,
    "agent": "BayesianFiniteVariantsAgent",
    "agents_count": 7,
    "agents_arguments": agents_arguments,
    "network_args": {
        "bidirectional_flow_rate": 0.005,
    },
    "recorder": "data",
    "simulate_type": "markov_chain"
}

agents_arguments = [{
    # "alpha":1/7,
    "alpha":alpha,
    "data_size":data_size,
    "variants_count":variants_count,
    "nonzero_alpha":"evely"
} for alpha in np.linspace(0.0001, 0.01, 5)]
network_args = [{
    "outward_flow_rate": fr,
} for fr in np.linspace(0.01, 1, 5)]
unique_args = {
    "simulation_count": [ 100000],
    # "simulate_type":["monte_carlo"],
    "agent": [ "BayesianInfiniteVariantsAgent"],
    "agents_arguments": agents_arguments,
    "network_args": network_args,
}
#データ数100
alpha=0.1
fr = 0.01
unique_args = {
    "simulation_count": [ 1000000],
    "agents_count": [15],
    "simulate_type":["monte_carlo"],
    "agent": [ "BayesianInfiniteVariantsAgent"],
    "agents_arguments": [{
    # "alpha":1/7,
    "alpha":alpha,
    "data_size":100,
    "nonzero_alpha":"evely"
},{
    # "alpha":1/7,
    "alpha":alpha,
    "data_size":100,
    "nonzero_alpha":"center"
} ],
    "network_args": [{
    "bidirectional_flow_rate": fr,
}, {
    "outward_flow_rate": fr,
}, 
],
}
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

# unique_args = {
#     "simulation_count": [ 100],
#     "agent": [ "BayesianInfiniteVariantsAgent"],
#     "agents_arguments": [{
#     "alpha":1,
#     "data_size":data_size,
#     "variants_count":variants_count,
#     "nonzero_alpha":nonzero_alpha
# } ],
#     "agents_count": [3],
#     "network_args": [{
#     "outward_flow_rate": 0.1,
#     }],
#     "initial_states": [None, np.array([1, 1, 1])],
# }
setting_count = np.prod([len(unique_args[key]) for key in unique_args.keys()])
args = [defalut_args.copy() for _ in range(setting_count)]
for i in range(setting_count):
    for ki, key in enumerate(unique_args.keys()):
        args[i][key] = unique_args[key][int(i // np.prod([len(unique_args[key]) for key in list(unique_args.keys())[ki+1:]])) % len(unique_args[key])]




recorder = "data"


setting_count = len(args)

def comulative_average(data):
    cumsum = np.cumsum(data, axis=0)
    counts = np.arange(1, data.shape[0] + 1)[:, None, None]
    return cumsum / counts

plt_data = []
for i in range(setting_count):
    setting_name = ''
    for key in unique_args.keys():
            setting_name += f"{key}_{args[i][key]}_"
    setting_name = setting_name.replace(":","_")
    file_path = DATA_DIR + '/raw/' +setting_name +".pkl"
    dir_path = DATA_DIR + '/raw/' +setting_name
    if LOAD_RESULT and (os.path.exists(file_path) or os.path.exists(dir_path)):
        print('load', setting_name)
        rec = data_manager.load_obj(DATA_DIR + '/raw/' + setting_name, [PLOT_OBJS])
        print(rec.keys())
    else:
        print('simulate', setting_name)
        rec = ilm.simulate(
            args[i]
        )
        rec.compute_distance()
        rec.compute_oldness()
        if SAVE_RESULT:
            data_manager.save_obj(rec, DATA_DIR + '/raw/' + setting_name, SAVE_KEYS, style="separete")
    result = comulative_average(rec.distance)
    with open(DATA_DIR + '/raw/' + setting_name + "/comulative_average.pkl", 'wb') as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)



