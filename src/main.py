import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

import ilm

DATA_DIR = os.path.dirname(__file__) + "/../data"

# 引数の定義
simulation_count = 100000
simulation_count = 100
agent1 = "BayesianFiniteVariantsAgent"
agent2 = "BayesianInfiniteVariantsAgent"
simulate_type = "markov_chain"
args = [{
    "agent": "BayesianFiniteVariantsAgent",
    "simulate_type": "monte_carlo",
    "simulation_count": 100000
},
{
    "agent": "BayesianFiniteVariantsAgent",
    "simulate_type": "markov_chain",
    "simulation_count": 100000
}]
# args =[{
#     "agent": "BayesianFiniteVariantsAgent",
#     "simulate_type": "markov_chain",
#     "simulation_count": 1000
# },]
network_args = {
    # "outward_flow_rate": 0.1,
    "bidirectional_flow_rate": 0.005,
    "is_torus": False
}
variants_count = 2
agents_count = 7
alpha = 1
data_size = 1
nonzero_alpha = "center"  # "evely" or "center"


if nonzero_alpha == "evely":
    agents_arguments = [{"alpha": alpha, "data_size": data_size, "variants_count": variants_count} for _ in range(agents_count)]
elif nonzero_alpha == "center":
    agents_arguments = [{"alpha": 0, "data_size": data_size, "variants_count": variants_count} for _ in range(agents_count)]
    agents_arguments[agents_count // 2]["alpha"] = alpha

recorder = "data"


condition_count = len(args)
recs = [
    ilm.simulate(
        simulation_count=args[i]["simulation_count"],
        agent=args[i]["agent"],
        network_args=network_args,
        agents_count=agents_count,
        agents_arguments=agents_arguments,
        recorder=recorder,
        simulate_type=args[i]["simulate_type"]
    ) for i in range(condition_count)
]
print(recs[0].record)
# pickle.dump(rec, open(DATA_DIR + "/distance/finite_variants.pkl", "wb"))
# pickle.dump(rec2, open(DATA_DIR + "/distance/infinite_variants.pkl", "wb"))





plt_data = []
for i in range(condition_count):
    if args[i]["agent"] == "BayesianFiniteVariantsAgent":
        if args[i]["simulate_type"] == "markov_chain":
            plt_data.append(recs[i].distance[-1]*variants_count)
        elif args[i]["simulate_type"] == "monte_carlo":
            print(recs[i].distance.shape)
            plt_data.append(np.mean(recs[i].distance, axis=0))
        else:
            raise ValueError("simulate_type must be 'markov_chain' or 'monte_carlo'")
    elif args[i]["agent"] == "BayesianInfiniteVariantsAgent":
        if args[i]["simulate_type"] == "markov_chain":
            plt_data.append(np.sum(recs[i].distance, axis=0))
        else:
            raise ValueError("simulate_type must be 'monte_carlo'")
    else:
        raise ValueError("agent must be 'BayesianFiniteVariantsAgent' or 'BayesianInfiniteVariantsAgent'")

    print(i,plt_data[-1])
# print(recs[0].distance)
fig, ax = plt.subplots(condition_count)
if condition_count == 1:
    ax = [ax]
for i in range(condition_count):
    ax[i].invert_yaxis()
    ax[i].pcolor(plt_data[i])
    ax[i].set_aspect('equal')
# ax[0].invert_yaxis()
# ax[0].pcolor(rec[1000:].mean(axis=0))
# ax[0].set_aspect('equal')
# ax[1].invert_yaxis()
# ax[1].pcolor(rec2.sum(axis=0))
# ax[1].set_aspect('equal')
plt.savefig(DATA_DIR + "/distance/finite_vs_infinite_variants.png")
plt.show()

for arg in args:
    np.save(DATA_DIR + f"/raw/{arg['agent']}_{arg['simulate_type']}_{network_args}.npy", plt_data)
