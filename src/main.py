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
SAVE_KEYS = ["distance", "oldness", "expected_distance", "expected_oldness"]

PLOT_STYLE = "grid"  # Options: "grid" or "line"
PLOT_OBJS = "oldness"  # Options: "distance" or "oldness"


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
unique_args = {
    "simulation_count": [ 50],
    "agents_count": [15],
    "simulate_type":["monte_carlo"],
    "agent": [ "BayesianInfiniteVariantsAgent"],
    "agents_arguments": [{
    # "alpha":1/7,
    "alpha":0.01,
    "data_size":100,
    "nonzero_alpha":"evely"
},{
    # "alpha":1/7,
    "alpha":0.01,
    "data_size":100,
    "nonzero_alpha":"center"
} ],
    "network_args": [{
    "bidirectional_flow_rate": 0.01,
}, {
    "outward_flow_rate": 0.01,
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
        # if not SAVE_STATES:
        #     rec.__return_record = None
        # if not SAVE_DISTANCES:
        #     rec.__distance = None
        # if not SAVE_EX_DISTANCE:
        #     rec.__expected_distance = None
        data_manager.save_obj(rec, DATA_DIR + '/raw/' + setting_name, SAVE_KEYS, style="separete")





    if args[i]["agent"] == "BayesianFiniteVariantsAgent":
        if args[i]["simulate_type"] == "markov_chain":
            plt_data.append(rec.distance[-1]*variants_count)
        elif args[i]["simulate_type"] == "monte_carlo":
            plt_data.append(np.mean(rec.distance[args[i]["simulation_count"]//10:], axis=0))
        else:
            raise ValueError("simulate_type must be 'markov_chain' or 'monte_carlo'")
    elif args[i]["agent"] == "BayesianInfiniteVariantsAgent":
        if args[i]["simulate_type"] == "markov_chain":
            plt_data.append(np.sum(rec.distance, axis=0))
        elif args[i]["simulate_type"] == "monte_carlo":
            if PLOT_OBJS == "distance":
                plt_data.append(np.mean(rec.distance[args[i]["simulation_count"]//10:], axis=0))
            elif PLOT_OBJS == "expected_distance":
                plt_data.append(rec.expected_distance)
            elif PLOT_OBJS == "oldness":
                plt_data.append(np.mean(rec.oldness[args[i]["simulation_count"]//10:], axis=0))
    else:
        raise ValueError("agent must be 'BayesianFiniteVariantsAgent' or 'BayesianInfiniteVariantsAgent'")


def is_concentric_distribution(expected_distance):
    for base in range(len(expected_distance)//2-1):
        for reference in range(len(expected_distance)):
            if expected_distance[base][reference] < expected_distance[base][len(expected_distance)//2] and reference > len(expected_distance)//2:
                return True
    return False
print(plt_data)
# print(recs[0].distance)
if PLOT_STYLE == "grid":
    setting_counts = [len(unique_args[key]) for key in unique_args.keys() if len(unique_args[key])>1]
    fig, ax = plt.subplots(*setting_counts, figsize=(5, 5) if setting_count > 1 else (5, 5))
    
    if setting_count == 1:
        ax = np.array([ax])
    for i, j in np.ndindex(ax.shape):
        if PLOT_OBJS == "distance" or PLOT_OBJS == "expected_distance":
            ax[i, j].invert_yaxis()
            ax[i, j].pcolor(plt_data[i*ax.shape[1]+j])
            ax[i, j].set_aspect('equal')
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
        elif PLOT_OBJS == "oldness":
            ax[i, j].plot(plt_data[i*ax.shape[1]+j])
            ax[i, j].set_ylim(bottom=0)
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
        else:
            raise ValueError("invalid PLOT_OBJS")
    plt.show()
if PLOT_STYLE == "line":
    fig, ax = plt.subplots(setting_count)
    if setting_count == 1:
        ax = [ax]
    for i in range(setting_count):
        if PLOT_OBJS == "distance" or PLOT_OBJS == "expected_distance":
            ax[i].invert_yaxis()
            ax[i].pcolor(plt_data[i])
            ax[i].set_aspect('equal')
        elif PLOT_OBJS == "oldness":
            ax[i].plot(plt_data[i])
            ax[i].set_ylim(bottom=0)

        else:
            raise ValueError("invalid PLOT_OBJS")
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)
    plt.show()

# fig_num = 10
# fig, ax = plt.subplots(fig_num)
# for i in range(fig_num):
#     ax[i].invert_yaxis()
#     ax[i].pcolor(recs[0].distance[i*1000])
#     ax[i].set_aspect('equal')
# plt.show()   
