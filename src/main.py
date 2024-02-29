import numpy as np
import ilm
import matplotlib.pyplot as plt



rec=ilm.simulate(
    simulation_count=1000,
    agent=ilm.agents.BayesianFiniteVariantsAgent,
    agents_count=4,
    agents_arguments={
        "alpha":1,
        "data_size":100,
    }
)
print(rec.record)