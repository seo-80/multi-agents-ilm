import numpy as np
import matplotlib.pyplot as plt
import os
import pickle



import ilm
import data_manager
DATA_DIR = os.environ['ONEDRIVE'] + "/SekiLabo/res_language/ilm/data"
PLOT_OBJS = "distance"

def sample_data(data, span):
    sampled_data = []
    for i in range(0, len(data), span):
        sampled_data.append(data[i])
    return sampled_data

# サンプリングされたデータを保存する関数
def save_sampled_data(sampled_data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(sampled_data, f)

# メイン処理
if __name__ == '__main__':
    agents_count = 7
    data_size = 1
    variants_count = 2
    alpha = 1
    simulation_count = 10000
    nonzero_alpha = "evely"  # "evely" or "center"
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
    args = [defalut_args.copy() for _ in range(setting_count)]
    for i in range(setting_count):
        for ki, key in enumerate(unique_args.keys()):
            args[i][key] = unique_args[key][int(i // np.prod([len(unique_args[key]) for key in list(unique_args.keys())[ki+1:]])) % len(unique_args[key])]


    for i in range(setting_count):
        setting_name = ''
        for key in unique_args.keys():
            setting_name += f"{key}_{args[i][key]}_"
        setting_name = setting_name.replace(":", "_")
        file_path = DATA_DIR + '/raw/' + setting_name + ".pkl"
        dir_path = DATA_DIR + '/raw/' + setting_name

        if (os.path.exists(file_path) or os.path.exists(dir_path)):
            print('load', setting_name)
            rec = data_manager.load_obj(DATA_DIR + '/raw/' + setting_name, PLOT_OBJS, number=None)

            # データをサンプリング
            sampled_data = sample_data(rec.distance, 100)  # スパンを100に設定

            # サンプリングされたデータを保存
            sampled_file_path = DATA_DIR + '/sampled/' + setting_name + "_sampled.pkl"
            save_sampled_data(sampled_data, sampled_file_path)
        else:
            print('Warning: No such file or directory:', setting_name)