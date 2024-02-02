import numpy as np
import ilm
import matplotlib.pyplot as plt



data=np.array([0,100,100])
data_size=100
data=np.zeros(data_size)
# agent=ilm.agents.BayesianFiniteVariantsAgent(data=data,alpha=10)
agent=ilm.agents.BayesianInfiniteVariantsAgent(data=data,alpha=1)
sc=1000

record=np.empty((sc,data_size,3),dtype=int)

for i in range(sc):
    data=agent.produce()
    agent.learn(data)
    record[i]=data

plt_data=np.empty((sc,sc))
for t,rc in enumerate(record):
    for di in range(sc):
        plt_data[t,di]=np.count_nonzero(rc[:,0]==di)
print(plt_data)
plt.plot(plt_data)
plt.show()
print(agent.hypothesis)