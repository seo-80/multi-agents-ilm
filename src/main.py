import numpy as np
import matplotlib.pyplot as plt
import os
import pickle


import ilm

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
        "is_torus": False
    },
    "recorder": "data",
    "simulate_type": "markov_chain"
}

agents_arguments = {
    "alpha":alpha,
    "data_size":data_size,
    "variants_count":variants_count,
    "nonzero_alpha":"center"
}


unique_args = {
    # "simulation_count": [100, 100000, 10000000],
    # "simulate_type":["monte_carlo"],
    "agents_arguments": [agents_arguments],
}
setting_count = np.prod([len(unique_args[key]) for key in unique_args.keys()])

args = [defalut_args.copy() for _ in range(setting_count)]
for i in range(setting_count):
    for ki, key in enumerate(unique_args.keys()):
        args[i][key] = unique_args[key][int(i // np.prod([len(unique_args[key]) for key in list(unique_args.keys())[ki+1:]])) % len(unique_args[key])]



print(args)

recorder = "data"


setting_count = len(args)
recs = []
for i in range(setting_count):
    setting_name = ''
    for key in unique_args.keys():
            setting_name += f"{key}_{args[i][key]}_"
    setting_name = setting_name.replace(":","_")
    file_path = DATA_DIR + '/raw/' +setting_name +".pkl"

    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            recs.append(pickle.load(f))
    else:
        rec = ilm.simulate(
            args[i]
        )
        rec.compute_distance()
        recs.append(rec)
        with open(file_path, "wb") as f:
            pickle.dump(recs[i], f)

# recs = [
#     ilm.simulate(
#         args[i]
#     ) for i in range(setting_count)
# ]
# pickle.dump(rec, open(DATA_DIR + "/distance/finite_variants.pkl", "wb"))
# pickle.dump(rec2, open(DATA_DIR + "/distance/infinite_variants.pkl", "wb"))





plt_data = []
for i in range(setting_count):
    if args[i]["agent"] == "BayesianFiniteVariantsAgent":
        if args[i]["simulate_type"] == "markov_chain":
            plt_data.append(recs[i].distance[-1]*variants_count)
        elif args[i]["simulate_type"] == "monte_carlo":
            plt_data.append(np.mean(recs[i].distance[recs[i].simulation_count//10:], axis=0))
        else:
            raise ValueError("simulate_type must be 'markov_chain' or 'monte_carlo'")
    elif args[i]["agent"] == "BayesianInfiniteVariantsAgent":
        if args[i]["simulate_type"] == "markov_chain":
            plt_data.append(np.sum(recs[i].distance, axis=0))
        else:
            raise ValueError("simulate_type must be 'monte_carlo'")
    else:
        raise ValueError("agent must be 'BayesianFiniteVariantsAgent' or 'BayesianInfiniteVariantsAgent'")


# print(recs[0].distance)
# fig, ax = plt.subplots(setting_count)
# if setting_count == 1:
#     ax = [ax]
# for i in range(setting_count):
#     ax[i].invert_yaxis()
#     ax[i].pcolor(plt_data[i])
#     ax[i].set_aspect('equal')
# ax[0].invert_yaxis()
# ax[0].pcolor(rec[1000:].mean(axis=0))
# ax[0].set_aspect('equal')
# ax[1].invert_yaxis()
# ax[1].pcolor(rec2.sum(axis=0))
# ax[1].set_aspect('equal')
# plt.savefig(DATA_DIR + "/distance/finite_vs_infinite_variants.png")
# plt.show()

fig_num = 10
fig, ax = plt.subplots(fig_num)
for i in range(fig_num):
    ax[i].invert_yaxis()
    ax[i].pcolor(recs[0].distance[i*1000])
    ax[i].set_aspect('equal')
plt.show()   