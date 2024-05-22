import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

import ilm

DATA_DIR = os.path.dirname(__file__) + "/../data"

# 引数の定義
simulation_count = 100000
agent = ilm.agents.BayesianFiniteVariantsAgent
network_args = {
    # "outward_flow_rate": 0.1,
    "bidirectional_flow_rate": 0.005,
    "is_torus": False
}
variants_count = 2
agents_count = 7
alphaperk = 0.2
data_size = 1
nonzero_alpha = "center"  # "evely" or "center"


alpha = alphaperk * agents_count

if nonzero_alpha == "evely":
    agents_arguments = [{"alpha": alpha, "data_size": data_size, "variants_count": variants_count} for _ in range(agents_count)]
elif nonzero_alpha == "center":
    agents_arguments = [{"alpha": 0, "data_size": data_size, "variants_count": variants_count} for _ in range(agents_count)]
    agents_arguments[agents_count // 2]["alpha"] = alpha

recorder = "data"


rec = ilm.simulate(
    simulation_count=simulation_count,
    agent=agent,
    network_args=network_args,
    agents_count=agents_count,
    agents_arguments=agents_arguments,
    recorder=recorder
)
rec2 = ilm.simulate_markov_chain(
    simulation_count=simulation_count,
    agent=agent,
    network_args=network_args,
    agents_count=agents_count,
    agents_arguments=agents_arguments,
    recorder=recorder
)
pickle.dump(rec, open(DATA_DIR + "/distance/finite_variants.pkl", "wb"))
pickle.dump(rec2, open(DATA_DIR + "/distance/infinite_variants.pkl", "wb"))
rec = rec.distance
rec2 = rec2.distance
print(rec.shape)
print(rec.sum(axis=0))
print(rec2.sum(axis=0))


np.save(DATA_DIR + "/distance/finite_variants.npy", rec)
# print(rec.mean(axis=0))
# print(rec2.sum(axis=0))
fig, ax = plt.subplots(2)
ax[0].invert_yaxis()
ax[0].pcolor(rec.mean(axis=0))
ax[0].set_aspect('equal')
ax[1].invert_yaxis()
ax[1].pcolor(rec2.sum(axis=0))
ax[1].set_aspect('equal')
plt.savefig(DATA_DIR + "/distance/finite_vs_infinite_variants.png")
plt.show()


np.savetxt(DATA_DIR + "/distance/finite_variants.csv", rec.mean(axis=0))
np.savetxt(DATA_DIR + "/distance/infinite_variants.csv", rec2.mean(axis=0))
