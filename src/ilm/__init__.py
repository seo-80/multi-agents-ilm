import numpy
from . import agents, networks, recorders



def simulate(
    simulation_count,
    agent,
    agents_count=None,
    agents_arguments=None,
    network=None,
    recorder="data",
):
    if agents_count is None:
        agents_count=1

    #init network
    if type(network) == str or network is None:
        network = networks.network(network_type=network,agents_count=agents_count)

    #init recorder
    if type(recorder) == str:
        recorder=recorders.recorder(recorder_type=recorder,simulation_count=simulation_count,agents_count=agents_count)
    
    #init agents
    if type(agents_arguments) == dict:
        agents_arguments=[agents_arguments for _ in range(agents_count)]
    agents=[agent(**arg) for arg in agents_arguments]
    total_data_counts=[agent.data_size for agent in agents]
    for si in range(simulation_count):
        recorder(agents=agents)
        data_flow_count=networks.generate_data_flow_count(network,total_data_counts=total_data_counts)
        datas=generate_datas(agents=agents,data_flow_counts=data_flow_count)
        for ai, agent in enumerate(agents):
            agent.learn(datas[ai])
        


    


    return recorder

def generate_datas(agents,data_flow_counts):
    datas=numpy.zeros((len(agents),agents[0].variants_count))

    for i,_ in enumerate(agents):
        for j,learner in enumerate(agents):
            datas[i]+=agents[j].produce(n=data_flow_counts[i][j])
    return datas
        
        