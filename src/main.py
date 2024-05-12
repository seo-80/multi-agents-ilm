import numpy as np
import ilm
import matplotlib.pyplot as plt

# 引数の定義
simulation_count = 100000
agent = ilm.agents.BayesianFiniteVariantsAgent
network_args = {
    "outward_flow_rate": 0.01,
    "is_torus": False
}
agents_count = 7
agents_arguments = [
    {"alpha": 0.1, "data_size": 1},
    {"alpha": 0.1, "data_size": 1},
    {"alpha": 0.1, "data_size": 1},
    {"alpha": 0.1, "data_size": 1},
    {"alpha": 0.1, "data_size": 1},
    {"alpha": 0.1, "data_size": 1},
    {"alpha": 0.1, "data_size": 1},
]
recorder = "distance"


rec = ilm.simulate(
    simulation_count=simulation_count,
    agent=agent,
    network_args=network_args,
    agents_count=agents_count,
    agents_arguments=agents_arguments,
    recorder=recorder
)

fig, ax = plt.subplots()
ax.pcolor(rec.record.mean(axis=0))
plt.show()

