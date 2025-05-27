import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import hashlib
import time
from typing import Optional


import ilm
import data_manager
# シミュレーション番号を引数として受け取る
import sys


# 引数の定義
SAVE_RESULT = False  # Set to True to save simulation results
LOAD_RESULT = True
PLOT_RESULT = True



PLOT_SCALE = True  # Set to True to scale the plot. 
PLOT_SCALE_TYPE = "linear"  # Options: "linear" or "log"



PLOT_STYLE = "grid"  # Options: "grid" or "line"
# PLOT_STYLE = "comulative_average"  # Options: "grid" or "line"
PLOT_OBJS = "oldness"  # Options: "distance" or "oldness"
PLOT_OBJS = "expected_distance"  # Options: "distance" or "oldness"
# PLOT_OBJS = "expected_oldness"  # Options: "distance" or "oldness"

# PLOT_OBJS = "oldness_sampled"  # Options: "distance" or "oldness"
# PLOT_OBJS = ["expected_oldness", "variance_oldness"]  # Options: "distance" or "oldness"
# PLOT_OBJS = "distance"  # Options: "distance" or "oldness"
# PLOT_OBJS = 'comulative_average'
PLOT_DISTANCE_FROM_ONE = True
# 起点となるエージェントの番号（PLOT_DISTANCE_FROM_ONEが有効な場合に使用）
DISTANCE_BASE_AGENT_INDEX = 3  # 例: 2番目のエージェントを起点にする


DATA_DIR = os.path.dirname(__file__) + "/../data"
# 事前に環境変数ONEDRIVEを設定しておく
DATA_DIR = os.environ['ONEDRIVE'] + "/SekiLabo/res_language/ilm/data"
# DATA_DIR = '~/Downloads'
# DATA_DIR = '/Users/hachimaruseo/Downloads'
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
# Add after the imports
def compute_stationary_state(W, mu):
    """
    Computes stationary state a*
    """
    n = W.shape[0]
    I = np.eye(n)
    M = np.diag(1 - mu)
    inv_matrix = np.linalg.inv(I - W @ M)
    ones_vec = np.ones(n)
    b = inv_matrix @ ones_vec
    a_star = b - ones_vec
    return a_star



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
    if LOAD_RESULT and  os.path.exists(dir_path):
        print('load', setting_name)
        rec = []
        simulation_version = 0
        simulation_count = 0
        while True:
            if simulation_count >= 200:
                break
            try:
                rec.append(data_manager.load_obj(DATA_DIR + '/raw/' + setting_name, PLOT_OBJS, number=simulation_version)[PLOT_OBJS])
                simulation_count += 1
            except:
                pass
            simulation_version += 1
    else:

        print('no simulation',dir_path)


    print('simulation_num:',len(rec))

    plt_data.append(rec)
    rec = None


def is_concentric_distribution(expected_distance):
    center = len(expected_distance) // 2
    for base in range(len(expected_distance)):
        found = False
        if base == center:
            continue
        for reference in range(len(expected_distance)):
            # referenceが中心より遠く、かつbaseから見て中心より言語的に近い
            if abs(reference - base) > abs(center - base) and expected_distance[base][reference] < expected_distance[base][center]:
                found = True
                break
        if not found:
            # print("not concentric:", base)
            return False
    return True
def is_concentric_distribution(expected_distance):
    center = len(expected_distance) // 2
    for base in range(len(expected_distance)):
        found = []
        if base == center:
            continue
        for reference in range(len(expected_distance)):
            # referenceが中心より遠く、かつbaseから見て中心より言語的に近い
            if abs(reference - base) > abs(center - base) and expected_distance[base][reference] < expected_distance[base][center]:
                found.append([base, reference])
                
        if not found:
            # print("not concentric:", base)
            return False
    return True

# print(recs[0].distance)

def plot_distance(ax, distance):
    distance = np.array(distance)
    if len(distance.shape) == 3:
        distance = distance.sum(axis=0)/distance.shape[0]
    if PLOT_DISTANCE_FROM_ONE:
        ax.bar(range(len(distance)), distance[:, DISTANCE_BASE_AGENT_INDEX])
        ax.bar(len(distance)//2, distance[len(distance)//2, DISTANCE_BASE_AGENT_INDEX], color="red")
        ax.set_ylim(bottom=0)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    else:
        im = ax.pcolor(distance)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.colorbar(im, ax=ax)

def plot_oldness(ax, oldness, min_oldness=None, max_oldness=None, scale_interval=None, plot_mean=True, plot_variance_only=False, analytical_solution=None):
    print("plot oldness")
    if type(oldness) == list:
        if plot_mean:
            mean = np.mean(oldness, axis=0)
            sv = np.sqrt(np.var(oldness, axis=0))
            if plot_variance_only:
                ax.plot(sv)
                ax.set_ylim(bottom=0)
            else:
                plot_oldness(ax, mean, min_oldness, max_oldness, scale_interval, plot_mean=False, plot_variance_only=plot_variance_only, analytical_solution=analytical_solution)
                ax.fill_between(range(len(mean)), mean - sv, mean + sv, alpha=0.3)
        else:
            for i in range(len(oldness)):
                plot_oldness(ax, oldness[i], min_oldness, max_oldness, scale_interval, analytical_solution=analytical_solution)
    else:
        variance_oldness = None
        if type(oldness) == list:
            oldness, variance_oldness = oldness
        if max_oldness is None:
            if variance_oldness is not None:
                max_oldness = np.max(oldness + np.sqrt(variance_oldness))*1.1
            else:
                max_oldness = np.max(oldness)*1.1
        if min_oldness is None:
            if PLOT_SCALE_TYPE == "log":
                min_oldness = 0.1
            else:
                min_oldness = 0
        if plot_variance_only and variance_oldness is not None:
            ax.plot(np.sqrt(variance_oldness))
        else:
            ax.plot(oldness, label='Simulation')
            if variance_oldness is not None:
                ax.fill_between(range(len(oldness)), oldness - np.sqrt(variance_oldness), oldness + np.sqrt(variance_oldness), alpha=0.3)
            ax.scatter(range(len(oldness)), oldness, s=5)
            
            # Plot analytical solution if provided
            if analytical_solution is not None:
                ax.plot(analytical_solution, '--', color='red', label='Analytical', linewidth=2)
                # ax.legend()
            
            ax.set_ylim(top=max_oldness)
            ax.set_yscale(PLOT_SCALE_TYPE)
            if scale_interval is not None:
                ax.set_yticks(np.arange(min_oldness, max_oldness, scale_interval))
            ax.ticklabel_format(style='plain', axis='y', useOffset=False)
            ax.tick_params(axis='y', labelsize=7)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
            ax.set_ylim(bottom=min_oldness)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(PLOT_SCALE)
        





def comulative_average(data):
    cumsum = np.cumsum(data, axis=0)
    counts = np.arange(1, data.shape[0] +1)
    return cumsum / counts

if PLOT_STYLE == "grid":
    setting_counts = [len(unique_args[key]) for key in unique_args.keys() if len(unique_args[key])>1]
    fig, ax = plt.subplots(*setting_counts, figsize=(5, 5) if setting_count > 1 else (5, 5))
    fig.subplots_adjust(wspace=0.3, hspace=0.2)
    
    if setting_count == 1:
        ax = np.array([ax])
    for i, j in np.ndindex(ax.shape):
        current_setting_idx = j*ax.shape[1]+i
        current_args = args[current_setting_idx]
        
        # Calculate analytical solution
        analytical_solution = None
        if PLOT_OBJS == "oldness" or PLOT_OBJS == "expected_oldness":
            # Get network parameters from current settings
            network_args = current_args["network_args"]
            agents_count = current_args["agents_count"]
            W = ilm.networks.network(agents_count, network_args)
            
            # Get agent parameters
            agent_args = current_args["agents_arguments"]
            alpha = agent_args["alpha"]
            data_size = agent_args["data_size"]
            nonzero_alpha = agent_args["nonzero_alpha"]
            
            # Calculate mu based on nonzero_alpha setting
            mu = np.zeros(agents_count)
            if nonzero_alpha == "evely":
                mu = np.ones(agents_count) * alpha / (data_size + alpha)
            elif nonzero_alpha == "center":
                mu[agents_count // 2] = alpha / (data_size + alpha)
            
            # Compute analytical solution
            analytical_solution = compute_stationary_state(W, mu)
            print(f"Setting {current_setting_idx}: alpha={alpha}, data_size={data_size}, nonzero_alpha={nonzero_alpha}")
            print("Analytical solution:", analytical_solution)
        
        if PLOT_OBJS == "expected_distance":
            print("fignum",i,j)
            print("is_concentric_distribution",is_concentric_distribution(np.mean(plt_data[current_setting_idx], axis=0)))
            concentric_distribution_count = 0
            for d in plt_data[current_setting_idx]:
                if is_concentric_distribution(d):
                    concentric_distribution_count += 1
            print("concentric_distribution_count", concentric_distribution_count)
            plot_distance(ax[i, j], plt_data[current_setting_idx])
        elif PLOT_OBJS == "oldness" or PLOT_OBJS == "expected_oldness" or "variance_oldness" in PLOT_OBJS:
            max_oldness = np.max([np.max(d) for d in plt_data])
            min_oldness = np.min([np.min(d) for d in plt_data])
            if i==0 and j==1:
                max_oldness = 32000
                min_oldness = 10000
                scale_interval = 2500
            elif j ==0:
                max_oldness = 1050
                min_oldness = 950
                scale_interval = 25
            else:
                max_oldness = 1750
                min_oldness = 750
                scale_interval = 250
            plot_oldness(ax[i, j], plt_data[current_setting_idx], min_oldness, max_oldness, scale_interval, analytical_solution=analytical_solution)
            # plot_oldness(ax[i, j], plt_data[j*ax.shape[1]+i], min_oldness,max_oldness)
        elif PLOT_OBJS == "distance_sampled" or PLOT_OBJS == "distance" or PLOT_OBJS =='oldness_sampled' or PLOT_OBJS == 'oldness':
            ax[i, j].plot(np.array(plt_data[j*ax.shape[1]+i]).reshape(10000,-1)[-500:])
        else:
            raise ValueError("invalid PLOT_OBJS")
    save_name = ''
    
    if PLOT_OBJS == "expected_distance":
        save_name += "_distance"
        if PLOT_DISTANCE_FROM_ONE:
            save_name += "_distance_from_one"
            save_name += f"_{DISTANCE_BASE_AGENT_INDEX}"
    elif PLOT_OBJS == "oldness" or PLOT_OBJS == "expected_oldness":
        save_name += "_oldness"
    elif "variance_oldness" in PLOT_OBJS:
        save_name += "_variance"
        
    if PLOT_SCALE:
        save_name += f"_{PLOT_SCALE_TYPE}_scale"
        
    save_name += "_with_numeric.png"
    print(f'save {save_name}')
    plt.savefig(os.path.join(DATA_DIR, 'fig', save_name))
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
        elif PLOT_OBJS == "oldness" or PLOT_OBJS == "expected_oldness":
            ax[i].plot(plt_data[i])
            ax[i].set_ylim(bottom=0)

        else:
            raise ValueError("invalid PLOT_OBJS")
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)
            # Save figure with appropriate name based on plot settings
    save_name = ''
    
    if PLOT_OBJS == "expected_distance":
        save_name += "_distance"
    elif PLOT_OBJS == "oldness" or PLOT_OBJS == "expected_oldness":
        save_name += "_oldness"
    elif "variance_oldness" in PLOT_OBJS:
        save_name += "_variance"
        
    if PLOT_SCALE:
        save_name += f"_{PLOT_SCALE_TYPE}_scale"
        
    save_name += ".png"
    
    plt.savefig(os.path.join(DATA_DIR, 'fig', save_name))
    print(f"save{save_name}")
    plt.show()

if PLOT_STYLE == "comulative_average":
    fig, ax = plt.subplots(setting_count)
    if setting_count == 1:
        ax = [ax]
    for i in range(setting_count):
        # Compute the relative change in quantity between consecutive time steps
        comulative_ave = np.zeros(plt_data[i].shape)
        relative_change = np.zeros(plt_data[i].shape[1:])
        for ai in range(plt_data[i].shape[1]):
            for aj in range(ai, plt_data[i].shape[2]):
                comulative_ave[ai,aj] = comulative_average(plt_data[i][:, ai, aj])
                relative_change[ai, aj] = (comulative_ave[ai, aj, -1] - comulative_ave[ai, aj, -2]) / comulative_ave[ai, aj , -1]
        # Find the indices of the largest relative changes at the final time step
        num_to_plot = 5  # Number of lines to plot
        largest_indices = np.unravel_index(np.argsort(relative_change, axis=None)[-num_to_plot:], relative_change.shape)
        
        # Plot the lines with the largest relative changes at the final time step
        for ai, aj in zip(*largest_indices):
            ax[i].plot(comulative_ave[:, ai, aj], label=f"({ai}, {aj})")
        
        ax[i].legend()
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)
    plt.savefig(DATA_DIR + '/fig/' + setting_name + "comulative_average.png")
    plt.show()
# fig_num = 10
# fig, ax = plt.subplots(fig_num)
# for i in range(fig_num):
#     ax[i].invert_yaxis()
#     ax[i].pcolor(recs[0].distance[i*1000])
#     ax[i].set_aspect('equal')
# plt.show()

